#!/usr/bin/env python3
"""Phase IV-C raw E(x) relational-offset analysis.

Consumes one or more complete caches produced by Phase_IV_C1. Run one
model/channel at a time into the same output directory; completed combinations
are checkpointed and all requested summary files are rebuilt from checkpoints.
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, platform, time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION="5.0.0"
EXPECTED_EDGE_SHA256="1e7c972c120bec39d59998b97f45a8dd3c584c37cf888a5f85f43d06087b9f39"
EXPECTED_CENSUS_SHA256="19192ccdda875f755abfb83f2deba9d5884c3163691c733d49b4f72aae712cb0"
MODELS=["qwen3_0.6b_1024d","bge_m3_1024d","gte_multilingual_base_768d","qwen3_4b_2560d_int8"]
CHANNELS=["zh_char","en_gloss_unihan","cn_def","en_def_translated_cn_def","en_gloss_google"]
TEXT_COL={"zh_char":"embed_text_zh_char","en_gloss_unihan":"embed_text_en_unihan","cn_def":"embed_text_cn_moe",
          "en_def_translated_cn_def":"embed_text_en_google_from_moe","en_gloss_google":"embed_text_en_google_character"}
ARROW_RE=r"^\s*(?:→|->)"

def now(): return datetime.now(timezone.utc).isoformat()
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()
def norm(x):
    n=np.linalg.norm(x,axis=-1,keepdims=True); return x/np.maximum(n,1e-12)
def cos_rows(a,b): return np.sum(norm(a)*norm(b),axis=1)
def safe_mean(x): return float(np.mean(x)) if len(x) else np.nan
def ids_layout(x):
    x=str(x).strip()
    names={"⿰":"left-right","⿱":"top-bottom","⿲":"left-middle-right","⿳":"top-middle-bottom",
      "⿴":"full-surround","⿵":"surround-from-above","⿶":"surround-from-below","⿷":"surround-from-left",
      "⿸":"surround-upper-left","⿹":"surround-upper-right","⿺":"surround-lower-left","⿻":"overlay"}
    return names.get(x[0],"non-IDS") if x else "non-IDS"
def slug(x): return hashlib.sha1(x.encode("utf-8")).hexdigest()[:12]
def atomic_csv(df,path):
    tmp=path.with_suffix(path.suffix+".tmp"); df.to_csv(tmp,index=False); os.replace(tmp,path)
def atomic_json(x,path):
    tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8"); os.replace(tmp,path)
def pairwise_mean_cos(x):
    if len(x)<2:return np.nan
    z=norm(x); s=z@z.T; return float((s.sum()-len(x))/(len(x)*(len(x)-1)))
def rank_metrics(ranks,prefix=""):
    a=np.asarray([r for r in ranks if np.isfinite(r)],float)
    return {prefix+"n":len(a),prefix+"recall_at_1":float(np.mean(a<=1)) if len(a) else np.nan,
      prefix+"recall_at_5":float(np.mean(a<=5)) if len(a) else np.nan,
      prefix+"recall_at_10":float(np.mean(a<=10)) if len(a) else np.nan,
      prefix+"mrr":float(np.mean(1/a)) if len(a) else np.nan,prefix+"median_rank":float(np.median(a)) if len(a) else np.nan}

def identify_batch(queries,candidates,candidate_names,true_idx,exclude_idx=None,batch=128):
    """Exact argmax and ranks; ties resolve by frozen candidate order."""
    cand=norm(np.asarray(candidates,dtype=np.float32)); names=np.asarray(candidate_names); out=[]
    for start in range(0,len(queries),batch):
        q=norm(np.asarray(queries[start:start+batch],dtype=np.float32)); scores=q@cand.T
        for j,row in enumerate(scores):
            k=start+j; ti=int(true_idx[k])
            if exclude_idx is not None and int(exclude_idx[k])>=0: row[int(exclude_idx[k])]=-np.inf
            if not np.isfinite(row[ti]): raise RuntimeError("True target excluded or nonfinite")
            pred=int(np.argmax(row)); ts=row[ti]
            rank=int(1+np.count_nonzero(row>ts)+np.count_nonzero((row[:ti]==ts)))
            out.append((str(names[pred]),rank))
    return out

def add_identification(frame,prefix,results,true_names,candidate_count):
    frame[prefix+"predicted_character"]=[x[0] for x in results]
    frame[prefix+"true_rank"]=[x[1] for x in results]
    for k in (1,5,10): frame[f"{prefix}top{k}_correct"]=(frame[prefix+"true_rank"]<=k).astype(int)
    # Top-1 is explicitly identity, and QC below proves equivalence with rank 1.
    frame[prefix+"top1_correct"]=(frame[prefix+"predicted_character"].to_numpy()==np.asarray(true_names)).astype(int)
    frame[prefix+"candidate_count"]=candidate_count

def qc_identification(df,kind):
    ncol="candidate_anchor_count" if kind=="inverse" else "candidate_character_count"
    true="anchor_character" if kind=="inverse" else "derived_character"
    for p in ("baseline_","offset_"):
        rank=df[p+"true_"+("anchor_rank" if kind=="inverse" else "character_rank")]
        pred=df[p+"predicted_"+("anchor" if kind=="inverse" else "character")]
        top=df[p+"top1_correct"].astype(bool)
        if not ((rank>=1)&(rank<=df[ncol])).all(): raise RuntimeError(f"{kind} rank outside candidate bounds")
        if not ((rank.eq(1))==top).all(): raise RuntimeError(f"{kind} rank/top1 invariant failed")
        if not (top==(pred==df[true])).all(): raise RuntimeError(f"{kind} argmax identity invariant failed")

EDGE_REQUIRED=["edge_id","anchor_char","derived_char","added_component","phonophore_root",
 "recursive_depth","family_id_root","family_id_immediate","edge_status","edge_confidence",
 "edge_dispute_flag","anchor_in_population","normalization_applied","normalization_collapse_flag",
 "anchor_char_normalized","added_component_normalized","phonophore_root_normalized",
 "edge_structure_pattern","edge_structure_pattern_normalized","layout"]

def validate_edge_universe(edges,master):
    missing=[c for c in EDGE_REQUIRED if c not in edges]
    if missing: raise ValueError(f"Authoritative edge table missing columns: {missing}")
    if edges["edge_id"].eq("").any() or edges.edge_id.duplicated().any(): raise ValueError("Blank/duplicate edge_id")
    key=["anchor_char","derived_char","added_component"]
    dup=edges.duplicated(key,keep=False)
    if dup.any(): raise ValueError(f"Duplicate canonical edges rejected: {int(dup.sum())} rows")
    if (edges.anchor_char==edges.derived_char).any(): raise ValueError("Self-edges are forbidden")
    for c in ["anchor_char","derived_char","added_component","phonophore_root","family_id_root","family_id_immediate"]:
        if edges[c].eq("").any(): raise ValueError(f"Blank required P/C/S/family field: {c}")
    if not edges.recursive_depth.str.fullmatch(r"[1-9]\d*").all(): raise ValueError("Invalid recursive_depth")
    valid_status={"eligible_direct","missing_anchor_from_frozen_population","required_diagnostic_candidate"}
    bad=set(edges.edge_status)-valid_status
    if bad: raise ValueError(f"Invalid direct edge_status values: {sorted(bad)}")
    for canonical,normalized in [("anchor_char","anchor_char_normalized"),("added_component","added_component_normalized"),
                                  ("phonophore_root","phonophore_root_normalized"),("edge_structure_pattern","edge_structure_pattern_normalized")]:
        if not (edges[canonical]==edges[normalized]).all():
            raise ValueError(f"Hidden/unfrozen normalization detected: {canonical} != {normalized}")
    pop=set(master.original_character)
    if not edges.derived_char.isin(pop).all(): raise ValueError("At least one derived endpoint does not resolve to master")
    actual=edges.anchor_char.isin(pop)
    declared=edges.anchor_in_population.str.lower().map({"true":True,"false":False})
    if declared.isna().any() or not (actual.to_numpy()==declared.to_numpy()).all(): raise ValueError("anchor_in_population disagrees with master identity")
    if edges.normalization_collapse_flag.str.lower().ne("false").any(): raise ValueError("Unresolved normalization collapse flag")
    # Directed forest validation: unique derived endpoints and no ancestry cycle.
    if edges.derived_char.duplicated().any(): raise ValueError("Graph invalid: derived endpoint has multiple immediate anchors")
    parent=dict(zip(edges.derived_char,edges.anchor_char))
    for node in parent:
        seen=set(); cur=node
        while cur in parent:
            if cur in seen: raise ValueError(f"Graph cycle detected at {cur}")
            seen.add(cur);cur=parent[cur]
    return {"duplicate_canonical_edges_rejected":0,"self_edges":0,"cycles":0,"lineage_graph_valid":True,
      "derived_ids_resolve":True,"anchor_population_flags_consistent":True}

def freeze_inputs(out,master_path,edges_path,census_path,qc_path,sha_manifest_path,cache_meta_path,tag):
    path=out/"input_fingerprints_v5.json"
    current={"master":{"filename":master_path.name,"sha256":sha256(master_path)},
      "edges":{"filename":edges_path.name,"sha256":sha256(edges_path)},
      "census":{"filename":census_path.name,"sha256":sha256(census_path)},
      "normalization_qc":{"filename":qc_path.name,"sha256":sha256(qc_path)},
      "sha256_manifest":{"filename":sha_manifest_path.name,"sha256":sha256(sha_manifest_path)}}
    state=json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"frozen":current,"cache_metadata":{}}
    if state.get("frozen")!=current: raise RuntimeError("Frozen master/edges/census changed under existing v5 result directory")
    cm_hash=sha256(cache_meta_path)
    prior=state["cache_metadata"].get(tag)
    if prior and prior!=cm_hash: raise RuntimeError(f"Cache metadata changed for {tag}")
    state["cache_metadata"][tag]=cm_hash;atomic_json(state,path)
    return state

def build_edges(master,index,E,edges):
    lookup={c:i for i,c in enumerate(index.original_character.astype(str))}
    eligible=set(lookup)
    e=edges.copy()
    e["derived_character"]=e.derived_char
    e["anchor_character"]=e.anchor_char
    e["relation_class"]=e.added_component
    e["anchor_family"]=e.family_id_root
    e["immediate_family"]=e.family_id_immediate
    e["layout_class"]=e.layout
    meta=master[["original_character","char_id","mandarin_pinyin_toneless"]].rename(columns={"char_id":"derived_char_id"})
    e=e.merge(meta,left_on="derived_character",right_on="original_character",how="left",validate="one_to_one").drop(columns=["original_character"])
    e["derived_in_cache"]=e.derived_character.isin(eligible)
    e["anchor_in_cache"]=e.anchor_character.isin(eligible)
    e=e[e.derived_in_cache&e.anchor_in_cache].copy().reset_index(drop=True)
    e["c_idx"]=e.derived_character.map(lookup); e["p_idx"]=e.anchor_character.map(lookup)
    delta=E[e.c_idx.to_numpy()]-E[e.p_idx.to_numpy()]
    return e,delta,lookup

def family_relation_means(e,delta):
    rows=[]; vectors=[]
    for (s,f),idx in e.groupby(["relation_class","anchor_family"],sort=True).groups.items():
        ids=np.asarray(list(idx),int); rows.append({"relation_class":s,"anchor_family":f,"n_edges":len(ids)})
        vectors.append(delta[ids].mean(axis=0))
    return pd.DataFrame(rows),np.asarray(vectors,dtype=np.float32)

def leave_family_prototypes(e,delta,min_train_families):
    fr,fv=family_relation_means(e,delta); byrel={s:g.index.to_numpy() for s,g in fr.groupby("relation_class")}
    proto=np.zeros_like(delta,dtype=np.float32); ok=np.zeros(len(e),bool); train_n=np.zeros(len(e),int)
    direction=np.zeros_like(delta,dtype=np.float32)
    f_lookup={(r.relation_class,r.anchor_family):i for i,r in fr.iterrows()}
    for i,r in e.iterrows():
        ids=byrel[r.relation_class]; held=f_lookup[(r.relation_class,r.anchor_family)]
        train=ids[ids!=held]; train_n[i]=len(train)
        if len(train)<min_train_families: continue
        proto[i]=fv[train].mean(axis=0)
        direction[i]=norm(norm(fv[train]).mean(axis=0,keepdims=True))[0]
        ok[i]=np.linalg.norm(proto[i])>0
    return proto,direction,ok,train_n,fr,fv

def structural_ranks(queries,e,index,E,true_chars,exclude_chars):
    layouts=index.decomposition_normalized.map(ids_layout).to_numpy(); chars=index.original_character.astype(str).to_numpy()
    lookup={c:i for i,c in enumerate(chars)}; ranks=[]; sizes=[]
    for q,layout,t,x in zip(queries,e.layout_class,true_chars,exclude_chars):
        ids=np.flatnonzero(layouts==layout); ex=lookup.get(x,-1); ti=lookup[t]
        ids=ids[ids!=ex]
        scores=norm(q[None,:])[0]@norm(E[ids]).T; target=np.where(ids==ti)[0]
        if len(target)==0:ranks.append(np.nan);sizes.append(len(ids));continue
        ranks.append(int(1+np.count_nonzero(scores>scores[target[0]])));sizes.append(len(ids))
    return ranks,sizes

def controls(e,E,lookup,residual,rng):
    all_chars=np.array(list(lookup)); records=[]
    byS={k:g.index.to_numpy() for k,g in e.groupby("relation_class")}
    byPy={k:g.index.to_numpy() for k,g in e.groupby("mandarin_pinyin_toneless") if k}
    byLay={k:g.index.to_numpy() for k,g in e.groupby("layout_class")}
    byBoth={k:g.index.to_numpy() for k,g in e.groupby(["relation_class","mandarin_pinyin_toneless"]) if k[1]}
    pools=[("random_anchor",None),("same_S_wrong_phonophore",byS),("pronunciation_matched_wrong_phonophore",byPy),
           ("layout_matched_wrong_phonophore",byLay),("pronunciation_plus_S_wrong_phonophore",byBoth)]
    for i,r in e.iterrows():
        for name,mapping in pools:
            if mapping is None: candidates=all_chars[all_chars!=r.anchor_character]
            else:
                key=(r.relation_class,r.mandarin_pinyin_toneless) if name.startswith("pronunciation_plus") else \
                    (r.relation_class if name.startswith("same_S") else r.mandarin_pinyin_toneless if name.startswith("pronunciation_") else r.layout_class)
                ids=mapping.get(key,np.array([],int)); candidates=e.loc[ids,"anchor_character"].drop_duplicates().to_numpy()
                candidates=candidates[candidates!=r.anchor_character]
            if len(candidates)==0: records.append({"edge_row":i,"control":name,"available":False});continue
            c=str(rng.choice(candidates)); records.append({"edge_row":i,"control":name,"available":True,"control_anchor":c,
                "residual_control_cosine":float(norm(residual[i:i+1])[0]@norm(E[lookup[c]:lookup[c]+1])[0])})
    return pd.DataFrame(records)

def learning_curves(fr,fv,min_train,rng,reps):
    rows=[]
    for s,g in fr.groupby("relation_class"):
        ids=g.index.to_numpy(); m=len(ids)
        for n in [1,2,4,8,16,m]:
            if n>=m or n<1: continue
            for rep in range(reps):
                test=int(rng.choice(ids)); pool=ids[ids!=test]
                if len(pool)<n:continue
                train=rng.choice(pool,n,replace=False); p=fv[train].mean(axis=0)
                rows.append({"relation_class":s,"n_training_families":n,"replicate":rep,
                  "held_out_family":fr.at[test,"anchor_family"],"heldout_offset_cosine":float(norm(p[None])[0]@norm(fv[test:test+1])[0])})
    return pd.DataFrame(rows)

def relation_nulls(fr,fv,rng,reps):
    observed=[]
    for s,g in fr.groupby("relation_class"):
        ids=g.index.to_numpy(); observed.append({"null_type":"observed","replicate":-1,"relation_class":s,"n_families":len(ids),"mean_pairwise_cosine":pairwise_mean_cos(fv[ids])})
    rows=observed
    labels=fr.relation_class.to_numpy().copy()
    families=fr.anchor_family.to_numpy()
    for rep in range(reps):
        # Family is the atomic row here: each family×relation offset was already averaged.
        shuffled=rng.permutation(labels)
        for s in np.unique(labels):
            ids=np.flatnonzero(shuffled==s)
            if len(ids)>=2: rows.append({"null_type":"relation_label_permutation_family_unit","replicate":rep,"relation_class":s,
                "n_families":len(ids),"mean_pairwise_cosine":pairwise_mean_cos(fv[ids])})
        # Random groupings preserve each observed relation class size.
        perm=rng.permutation(len(fr)); cursor=0
        for s,g in fr.groupby("relation_class"):
            n=len(g); ids=perm[cursor:cursor+n];cursor+=n
            if n>=2: rows.append({"null_type":"random_offset_group_same_size","replicate":rep,"relation_class":s,
                "n_families":n,"mean_pairwise_cosine":pairwise_mean_cos(fv[ids])})
    return pd.DataFrame(rows)

def analyze(master,index,E,edges,model,channel,args):
    e,delta,lookup=build_edges(master,index,E,edges)
    proto,dproto,ok,train_n,fr,fv=leave_family_prototypes(e,delta,args.min_train_families)
    e["model"]=model;e["channel"]=channel;e["edge_row"]=np.arange(len(e));e["training_family_count"]=train_n;e["prototype_available"]=ok
    e["offset_norm"]=np.linalg.norm(delta,axis=1);e["raw_anchor_derived_cosine"]=cos_rows(E[e.c_idx],E[e.p_idx])
    e["offset_prototype_cosine"]=np.nan;e.loc[ok,"offset_prototype_cosine"]=cos_rows(delta[ok],proto[ok])
    use=e[ok].copy(); ids=np.flatnonzero(ok); qf=E[use.p_idx.to_numpy()]+proto[ids]; qi=E[use.c_idx.to_numpy()]-proto[ids]
    chars=index.original_character.astype(str).tolist(); char_lookup={c:i for i,c in enumerate(chars)}
    true_c=[char_lookup[c] for c in use.derived_character]; ex_p=[char_lookup[c] for c in use.anchor_character]
    foff=identify_batch(qf,E,chars,true_c,ex_p,args.retrieval_batch)
    fbase=identify_batch(E[use.p_idx.to_numpy()],E,chars,true_c,ex_p,args.retrieval_batch)
    f_struct,f_sizes=structural_ranks(qf,use,index,E,use.derived_character,use.anchor_character)
    forward=use[["model","channel","edge_row","derived_character","anchor_character","relation_class","anchor_family","training_family_count"]].copy()
    add_identification(forward,"baseline_",fbase,use.derived_character,len(E)-1)
    add_identification(forward,"offset_",foff,use.derived_character,len(E)-1)
    forward=forward.rename(columns={"baseline_true_rank":"baseline_true_character_rank","offset_true_rank":"offset_true_character_rank",
      "baseline_candidate_count":"candidate_character_count"}).drop(columns=["offset_candidate_count"])
    forward["rank_change"]=forward.baseline_true_character_rank-forward.offset_true_character_rank
    forward["top1_rescued"]=((forward.baseline_top1_correct==0)&(forward.offset_top1_correct==1)).astype(int)
    forward["top1_lost"]=((forward.baseline_top1_correct==1)&(forward.offset_top1_correct==0)).astype(int)
    forward["true_rank_unrestricted"]=forward.offset_true_character_rank;forward["candidate_count_unrestricted"]=len(E)-1
    forward["true_rank_layout_constrained"]=f_struct;forward["candidate_count_layout_constrained"]=f_sizes
    # Candidate anchors are independent characters documented as immediate anchors.
    anchor_set=set(edges.anchor_char)&set(char_lookup);anchor_chars=[c for c in chars if c in anchor_set]
    anchor_ids=np.array([char_lookup[c] for c in anchor_chars]); amap={c:i for i,c in enumerate(anchor_chars)}
    ioff=identify_batch(qi,E[anchor_ids],anchor_chars,[amap[c] for c in use.anchor_character],None,args.retrieval_batch)
    ibase=identify_batch(E[use.c_idx.to_numpy()],E[anchor_ids],anchor_chars,[amap[c] for c in use.anchor_character],None,args.retrieval_batch)
    inverse=use[["model","channel","edge_row","derived_character","anchor_character","relation_class","anchor_family","training_family_count"]].copy()
    add_identification(inverse,"baseline_",ibase,use.anchor_character,len(anchor_chars));add_identification(inverse,"offset_",ioff,use.anchor_character,len(anchor_chars))
    inverse=inverse.rename(columns={"baseline_predicted_character":"baseline_predicted_anchor","baseline_true_rank":"baseline_true_anchor_rank",
      "offset_predicted_character":"offset_predicted_anchor","offset_true_rank":"offset_true_anchor_rank",
      "baseline_candidate_count":"candidate_anchor_count"}).drop(columns=["offset_candidate_count"])
    inverse["rank_change"]=inverse.baseline_true_anchor_rank-inverse.offset_true_anchor_rank
    inverse["top1_rescued"]=((inverse.baseline_top1_correct==0)&(inverse.offset_top1_correct==1)).astype(int)
    inverse["top1_lost"]=((inverse.baseline_top1_correct==1)&(inverse.offset_top1_correct==0)).astype(int)
    inverse["true_anchor_rank"]=inverse.offset_true_anchor_rank
    residual=np.zeros_like(delta);residual[ok]=E[e.loc[ok,"c_idx"]]-proto[ok]
    e["residual_to_anchor_cosine"]=np.nan;e.loc[ok,"residual_to_anchor_cosine"]=cos_rows(residual[ok],E[e.loc[ok,"p_idx"]])
    e["residual_improvement_over_raw_cosine"]=e.residual_to_anchor_cosine-e.raw_anchor_derived_cosine
    inverse=inverse.merge(e.loc[ok,["edge_row","raw_anchor_derived_cosine","residual_to_anchor_cosine","residual_improvement_over_raw_cosine"]],on="edge_row",validate="one_to_one")
    qc_identification(inverse,"inverse");qc_identification(forward,"forward")
    fam=[]
    for family,g in e[ok].groupby("anchor_family"):
        x=g.index.to_numpy()
        if len(x)<2:continue
        fam.append({"model":model,"channel":channel,"anchor_family":family,"n_edges":len(x),
          "ordinary_whole_vector_coherence":pairwise_mean_cos(E[e.loc[x,"c_idx"]]),
          "residual_family_coherence":pairwise_mean_cos(residual[x]),
          "mean_residual_to_anchor_cosine":float(e.loc[x,"residual_to_anchor_cosine"].mean())})
    fam=pd.DataFrame(fam)
    if len(fam):fam["residual_coherence_improvement"]=fam.residual_family_coherence-fam.ordinary_whole_vector_coherence
    rng=np.random.default_rng(args.seed)
    control=controls(e[ok].reset_index(drop=True),E,lookup,residual[ok],rng);control["model"]=model;control["channel"]=channel
    learn=learning_curves(fr,fv,args.min_train_families,rng,args.learning_reps);learn["model"]=model;learn["channel"]=channel
    nulls=relation_nulls(fr,fv,rng,args.null_reps);nulls["model"]=model;nulls["channel"]=channel
    rel=[]
    for s,g in e.groupby("relation_class"):
        family_ids=fr.index[fr.relation_class==s].to_numpy(); valid=g[g.prototype_available]
        row={"model":model,"channel":channel,"relation_class":s,"n_edges":len(g),"n_anchor_families":g.anchor_family.nunique(),
          "within_relation_family_offset_cosine":pairwise_mean_cos(fv[family_ids]),"n_heldout_test_edges":len(valid),
          "n_heldout_test_edges_ge5":int((g.training_family_count>=5).sum()),"n_heldout_test_edges_ge8":int((g.training_family_count>=8).sum()),
          "mean_heldout_offset_prototype_cosine":valid.offset_prototype_cosine.mean() if len(valid) else np.nan,
          "mean_heldout_offset_prototype_cosine_ge5":g.loc[g.training_family_count>=5,"offset_prototype_cosine"].mean(),
          "mean_heldout_offset_prototype_cosine_ge8":g.loc[g.training_family_count>=8,"offset_prototype_cosine"].mean()}
        row.update(rank_metrics(forward.loc[forward.relation_class==s,"offset_true_character_rank"],"forward_"))
        row.update(rank_metrics(inverse.loc[inverse.relation_class==s,"offset_true_anchor_rank"],"inverse_"));rel.append(row)
    rel=pd.DataFrame(rel)
    prototype_full=[]; keys=[]
    for s,g in fr.groupby("relation_class"):
        keys.append(s);prototype_full.append(fv[g.index].mean(axis=0))
    return e,delta,np.asarray(prototype_full),keys,rel,forward,inverse,fam,control,nulls,learn

def identification_summary(inv,fwd):
    rows=[]
    for (model,channel),ig in inv.groupby(["model","channel"]):
        fg=fwd[(fwd.model==model)&(fwd.channel==channel)]
        for threshold in (3,5,8):
            i=ig[ig.training_family_count>=threshold];f=fg[fg.training_family_count>=threshold]
            row={"model":model,"channel":channel,"minimum_other_training_families":threshold,
              "eligible_heldout_edges":len(i),"candidate_anchor_count":int(i.candidate_anchor_count.iloc[0]) if len(i) else np.nan,
              "candidate_character_count":int(f.candidate_character_count.iloc[0]) if len(f) else np.nan,
              "inverse_percent_rank_improved":100*safe_mean(i.rank_change>0),"inverse_percent_rank_worsened":100*safe_mean(i.rank_change<0),
              "inverse_top1_rescues":int(i.top1_rescued.sum()),"inverse_top1_losses":int(i.top1_lost.sum()),
              "inverse_net_top1_gain":int(i.top1_rescued.sum()-i.top1_lost.sum())}
            for label,g,col in [("inverse",i,"anchor"),("forward",f,"character")]:
                for p in ("baseline","offset"):
                    r=g[f"{p}_true_{col}_rank"]
                    row.update({f"{label}_{p}_recall_at_1":safe_mean(r<=1),f"{label}_{p}_recall_at_5":safe_mean(r<=5),
                      f"{label}_{p}_recall_at_10":safe_mean(r<=10),f"{label}_{p}_mrr":safe_mean(1/r),
                      f"{label}_{p}_median_rank":float(r.median()) if len(r) else np.nan})
                if label=="inverse" and len(g):
                    n=g.candidate_anchor_count.astype(float);row["inverse_baseline_mean_percentile_rank"]=safe_mean(1-(g.baseline_true_anchor_rank-1)/np.maximum(n-1,1));row["inverse_offset_mean_percentile_rank"]=safe_mean(1-(g.offset_true_anchor_rank-1)/np.maximum(n-1,1))
            if threshold==3 and len(i):
                row["human_interpretation"]=(f"Across {len(i):,} held-out cases, the correct phonophore/source anchor was rank 1 in "
                  f"{100*row['inverse_baseline_recall_at_1']:.2f}% before relation subtraction and {100*row['inverse_offset_recall_at_1']:.2f}% afterward.")
                row["uniform_top1_reference"]=1/row["candidate_anchor_count"]
            rows.append(row)
    return pd.DataFrame(rows)

def consolidate(out):
    ck=out/"checkpoints"; mapping={
      "edge_offset_scores_v5.csv":"edge.csv","semantic_relation_summary_v5.csv":"relation.csv",
      "forward_analogy_retrieval_v5.csv":"forward.csv","inverse_anchor_recovery_v5.csv":"inverse.csv",
      "residual_family_coherence_v5.csv":"residual.csv","null_model_results_v5.csv":"null.csv",
      "relation_learning_curves_v5.csv":"learning.csv","matched_control_results_v5.csv":"control.csv"}
    combined={}
    for final,suffix in mapping.items():
        files=sorted(ck.glob(f"*__{suffix}")); frames=[pd.read_csv(f) for f in files]
        x=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame();atomic_csv(x,out/final);combined[final]=x
    rel=combined["semantic_relation_summary_v5.csv"]
    cross_model=rel.groupby(["channel","relation_class"],as_index=False).agg(models=("model","nunique"),mean_offset_consistency=("mean_heldout_offset_prototype_cosine","mean"),mean_forward_mrr=("forward_mrr","mean"),mean_inverse_mrr=("inverse_mrr","mean")) if len(rel) else pd.DataFrame()
    atomic_csv(cross_model,out/"cross_model_relation_summary_v5.csv")
    cross_channel=rel.groupby(["model","relation_class"],as_index=False).agg(channels=("channel","nunique"),mean_offset_consistency=("mean_heldout_offset_prototype_cosine","mean"),mean_forward_mrr=("forward_mrr","mean"),mean_inverse_mrr=("inverse_mrr","mean")) if len(rel) else pd.DataFrame()
    atomic_csv(cross_channel,out/"cross_channel_relation_summary_v5.csv")
    # Matched evaluation uses exact direct-edge intersections. Prototypes remain
    # independently cross-validated inside each channel's eligible population.
    pairs=[]
    edge=combined["edge_offset_scores_v5.csv"]
    for a,b in [("zh_char","en_gloss_unihan"),("zh_char","cn_def"),("zh_char","en_def_translated_cn_def"),("cn_def","en_def_translated_cn_def")]:
        if len(edge):
            keys=["model","derived_character","anchor_character","relation_class","anchor_family"]
            keep=keys+["offset_prototype_cosine","residual_to_anchor_cosine","prototype_available"]
            x=edge.loc[edge.channel==a,keep].merge(edge.loc[edge.channel==b,keep],on=keys,suffixes=("_a","_b"))
            x=x[x.prototype_available_a.astype(bool)&x.prototype_available_b.astype(bool)]
            if len(x):
                y=x.groupby(["model","relation_class"],as_index=False).agg(
                    matched_edges=("derived_character","size"),
                    mean_offset_cosine_a=("offset_prototype_cosine_a","mean"),mean_offset_cosine_b=("offset_prototype_cosine_b","mean"),
                    mean_residual_anchor_cosine_a=("residual_to_anchor_cosine_a","mean"),mean_residual_anchor_cosine_b=("residual_to_anchor_cosine_b","mean"))
                y["channel_a"]=a;y["channel_b"]=b;pairs.append(y)
                y["subset_id"]=f"{a}__INTERSECT__{b}"
    if len(edge):
        keys=["model","derived_character","anchor_character","relation_class","anchor_family"]
        metric=["offset_prototype_cosine","residual_to_anchor_cosine","prototype_available"]
        chans=["zh_char","cn_def","en_def_translated_cn_def"]
        parts=[]
        for j,c in enumerate(chans):
            z=edge.loc[edge.channel==c,keys+metric].rename(columns={m:f"{m}_{j}" for m in metric});parts.append(z)
        x=parts[0].merge(parts[1],on=keys).merge(parts[2],on=keys)
        if len(x):
            x=x[x.prototype_available_0.astype(bool)&x.prototype_available_1.astype(bool)&x.prototype_available_2.astype(bool)]
            y=x.groupby(["model","relation_class"],as_index=False).agg(
                matched_edges=("derived_character","size"),mean_offset_cosine_a=("offset_prototype_cosine_0","mean"),
                mean_offset_cosine_b=("offset_prototype_cosine_1","mean"),mean_offset_cosine_c=("offset_prototype_cosine_2","mean"),
                mean_residual_anchor_cosine_a=("residual_to_anchor_cosine_0","mean"),
                mean_residual_anchor_cosine_b=("residual_to_anchor_cosine_1","mean"),
                mean_residual_anchor_cosine_c=("residual_to_anchor_cosine_2","mean"))
            y["channel_a"],y["channel_b"],y["channel_c"]=chans;y["subset_id"]="zh_char__INTERSECT__cn_def__INTERSECT__en_def_translated_cn_def";pairs.append(y)
    matched=pd.concat(pairs,ignore_index=True) if pairs else pd.DataFrame();atomic_csv(matched,out/"matched_subset_relation_summary_v5.csv")
    atomic_csv(matched[(matched.get("channel_a",pd.Series(dtype=str))=="cn_def")&(matched.get("channel_b",pd.Series(dtype=str))=="en_def_translated_cn_def")] if len(matched) else pd.DataFrame(),out/"cross_language_relation_summary_v5.csv")
    inv=combined["inverse_anchor_recovery_v5.csv"];fwd=combined["forward_analogy_retrieval_v5.csv"]
    summary=identification_summary(inv,fwd) if len(inv) else pd.DataFrame();atomic_csv(summary,out/"retrieval_identification_summary_v5.csv")
    # Predeclared diagnostics: two named v5 families plus the ten frozen Phase-II roots.
    roots={"青","堯","店","余","元","帝","闌","辰","曷","賓","圭","台","亥"}
    diag=inv[(inv.anchor_character.isin(roots))|(inv.derived_character.isin({"掂","惦"}))].copy() if len(inv) else pd.DataFrame()
    if len(diag):
        diag=diag.rename(columns={"anchor_character":"P","derived_character":"C","relation_class":"S",
          "training_family_count":"other_training_family_count","baseline_predicted_anchor":"baseline_predicted_P",
          "baseline_true_anchor_rank":"baseline_rank_true_P","offset_predicted_anchor":"offset_predicted_P",
          "offset_true_anchor_rank":"offset_rank_true_P","offset_top1_correct":"true_P_offset_top1",
          "offset_top5_correct":"true_P_offset_top5","offset_top10_correct":"true_P_offset_top10",
          "raw_anchor_derived_cosine":"baseline_cosine","residual_to_anchor_cosine":"adjusted_cosine"})
    atomic_csv(diag,out/"diagnostic_identification_cases_v5.csv")
    matched_rows=[]
    if len(inv):
        keys=["model","edge_row","derived_character","anchor_character"]
        for model,g in inv.groupby("model"):
            chans=sorted(g.channel.unique())
            for x in range(len(chans)):
                for y in range(x+1,len(chans)):
                    a,b=chans[x],chans[y];z=g[g.channel==a][keys+["baseline_top1_correct","offset_top1_correct"]].merge(g[g.channel==b][keys+["baseline_top1_correct","offset_top1_correct"]],on=keys,suffixes=("_a","_b"))
                    if len(z): matched_rows.append({"model":model,"channel_a":a,"channel_b":b,"matched_edges":len(z),"baseline_recall_at_1_a":z.baseline_top1_correct_a.mean(),"offset_recall_at_1_a":z.offset_top1_correct_a.mean(),"baseline_recall_at_1_b":z.baseline_top1_correct_b.mean(),"offset_recall_at_1_b":z.offset_top1_correct_b.mean()})
    atomic_csv(pd.DataFrame(matched_rows),out/"matched_cross_channel_inverse_recall_v5.csv")

def plots(out):
    import matplotlib.pyplot as plt
    pdir=out/"plots";pdir.mkdir(exist_ok=True)
    null=pd.read_csv(out/"null_model_results_v5.csv"); rel=pd.read_csv(out/"semantic_relation_summary_v5.csv"); learn=pd.read_csv(out/"relation_learning_curves_v5.csv");res=pd.read_csv(out/"residual_family_coherence_v5.csv")
    if len(null):
        plt.figure(figsize=(8,5));
        for name,g in null.groupby("null_type"): plt.hist(g.mean_pairwise_cosine.dropna(),bins=40,alpha=.45,label=name,density=True)
        plt.legend(fontsize=7);plt.xlabel("Family-unit offset cosine");plt.tight_layout();plt.savefig(pdir/"within_relation_vs_null_offset_cosine.png",dpi=160);plt.close()
    if len(rel):
        g=rel.groupby(["model","channel"])[["forward_mrr","inverse_mrr"]].mean().reset_index();x=np.arange(len(g));
        plt.figure(figsize=(max(8,len(g)*.5),5));plt.bar(x-.2,g.forward_mrr,.4,label="forward MRR");plt.bar(x+.2,g.inverse_mrr,.4,label="inverse MRR");plt.xticks(x,g.model+"\n"+g.channel,rotation=70,ha="right",fontsize=7);plt.legend();plt.tight_layout();plt.savefig(pdir/"model_channel_retrieval_comparison.png",dpi=160);plt.close()
    if len(learn):
        g=learn.groupby(["channel","n_training_families"]).heldout_offset_cosine.mean().reset_index();plt.figure(figsize=(8,5))
        for c,z in g.groupby("channel"):plt.plot(z.n_training_families,z.heldout_offset_cosine,marker="o",label=c)
        plt.xscale("log",base=2);plt.xlabel("Independent training families");plt.ylabel("Held-out offset cosine");plt.legend(fontsize=8);plt.tight_layout();plt.savefig(pdir/"relation_learning_curves.png",dpi=160);plt.close()
    if len(res):
        plt.figure(figsize=(8,5));plt.hist(res.residual_coherence_improvement.dropna(),bins=50);plt.axvline(0,color="black",lw=1);plt.xlabel("Residual minus whole-vector family coherence");plt.tight_layout();plt.savefig(pdir/"family_residual_improvement.png",dpi=160);plt.close()
    summary=pd.read_csv(out/"retrieval_identification_summary_v5.csv");inv=pd.read_csv(out/"inverse_anchor_recovery_v5.csv");diag=pd.read_csv(out/"diagnostic_identification_cases_v5.csv")
    primary=summary[summary.minimum_other_training_families==3].copy() if len(summary) else summary
    def paired(metric,title,name):
        if not len(primary): return
        labels=primary.model+"\n"+primary.channel;x=np.arange(len(primary));plt.figure(figsize=(max(9,len(primary)*.55),5));plt.bar(x-.2,primary["inverse_baseline_"+metric],.4,label="baseline");plt.bar(x+.2,primary["inverse_offset_"+metric],.4,label="offset");plt.xticks(x,labels,rotation=70,ha="right",fontsize=7);plt.title(title);plt.legend();plt.tight_layout();plt.savefig(pdir/name,dpi=160);plt.close()
    paired("recall_at_1","Inverse anchor identification: Recall@1","inverse_recall_at_1_baseline_vs_offset.png")
    paired("median_rank","Inverse anchor identification: median true rank","inverse_median_rank_baseline_vs_offset.png")
    if len(primary):
        labels=primary.model+"\n"+primary.channel;x=np.arange(len(primary));plt.figure(figsize=(max(9,len(primary)*.55),5));
        for j,(col,label) in enumerate([("inverse_baseline_recall_at_5","base R@5"),("inverse_offset_recall_at_5","offset R@5"),("inverse_baseline_recall_at_10","base R@10"),("inverse_offset_recall_at_10","offset R@10")]):plt.plot(x,primary[col],marker="o",label=label)
        plt.xticks(x,labels,rotation=70,ha="right",fontsize=7);plt.legend();plt.tight_layout();plt.savefig(pdir/"inverse_recall_at_5_10_baseline_vs_offset.png",dpi=160);plt.close()
        plt.figure(figsize=(9,5));plt.hist(inv.rank_change,bins=60);plt.axvline(0,color="black");plt.xlabel("Baseline rank − offset rank (positive = improvement)");plt.tight_layout();plt.savefig(pdir/"inverse_rank_change_distribution.png",dpi=160);plt.close()
        labels=primary.model+"\n"+primary.channel;x=np.arange(len(primary));plt.figure(figsize=(max(9,len(primary)*.55),5));plt.bar(x-.2,primary.forward_baseline_recall_at_1,.4,label="baseline");plt.bar(x+.2,primary.forward_offset_recall_at_1,.4,label="offset");plt.xticks(x,labels,rotation=70,ha="right",fontsize=7);plt.legend();plt.tight_layout();plt.savefig(pdir/"forward_recall_at_1_baseline_vs_offset.png",dpi=160);plt.close()
        plt.figure(figsize=(9,5));
        for (m,c),g in summary.groupby(["model","channel"]):plt.plot(g.minimum_other_training_families,g.inverse_offset_recall_at_1,marker="o",alpha=.65,label=f"{m}/{c}")
        plt.xticks([3,5,8]);plt.xlabel("Minimum OTHER training families");plt.ylabel("Offset inverse Recall@1");plt.legend(fontsize=5,ncol=2);plt.tight_layout();plt.savefig(pdir/"support_sensitivity_inverse_recall_at_1.png",dpi=160);plt.close()
    matched_path=out/"matched_cross_channel_inverse_recall_v5.csv"
    try: m=pd.read_csv(matched_path)
    except (FileNotFoundError,pd.errors.EmptyDataError): m=pd.DataFrame()
    if len(m):
        plt.figure(figsize=(9,5));plt.scatter(m.baseline_recall_at_1_a,m.offset_recall_at_1_a,label="channel A");plt.scatter(m.baseline_recall_at_1_b,m.offset_recall_at_1_b,label="channel B");plt.plot([0,1],[0,1],color="black",lw=1);plt.xlabel("Matched baseline Recall@1");plt.ylabel("Matched offset Recall@1");plt.legend();plt.tight_layout();plt.savefig(pdir/"matched_cross_channel_inverse_recall_at_1.png",dpi=160);plt.close()
    if len(diag):
        plt.figure(figsize=(9,5));
        for j,r in diag.reset_index(drop=True).iterrows():plt.plot([0,1],[r.baseline_rank_true_P,r.offset_rank_true_P],marker="o",alpha=.5)
        plt.xticks([0,1],["baseline","offset"]);plt.ylabel("True-anchor rank");plt.tight_layout();plt.savefig(pdir/"diagnostic_baseline_to_offset_rank.png",dpi=160);plt.close()

def main():
    p=argparse.ArgumentParser();p.add_argument("--master",type=Path,required=True);p.add_argument("--edges",type=Path,required=True)
    p.add_argument("--census",type=Path,required=True,help="Traditional_Chinese_Component_Family_Universe_11151_v3.xlsx")
    p.add_argument("--normalization-qc",type=Path,required=True);p.add_argument("--sha256-manifest",type=Path,required=True)
    p.add_argument("--cache-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model",choices=MODELS,required=True);p.add_argument("--channel",choices=CHANNELS,required=True);p.add_argument("--min-train-families",type=int,default=3)
    p.add_argument("--null-reps",type=int,default=100);p.add_argument("--learning-reps",type=int,default=30);p.add_argument("--retrieval-batch",type=int,default=128);p.add_argument("--seed",type=int,default=20260903)
    p.add_argument("--allow-google-exploratory",action="store_true");p.add_argument("--force",action="store_true")
    p.add_argument("--centered-sensitivity",action="store_true",help="Run separately after raw primary; subtract channel mean and label outputs as sensitivity")
    a=p.parse_args()
    if a.min_train_families!=3: raise ValueError("V5 primary threshold is frozen at exactly 3 other root families")
    if a.channel=="en_gloss_google" and not a.allow_google_exploratory:raise SystemExit("Google direct gloss is audit-demoted; explicit --allow-google-exploratory required")
    analysis_channel=a.channel+("__centered_sensitivity" if a.centered_sensitivity else "")
    a.output.mkdir(parents=True,exist_ok=True);ck=a.output/"checkpoints";ck.mkdir(exist_ok=True); tag=f"{a.model}__{analysis_channel}"
    edge_hash=sha256(a.edges);census_hash=sha256(a.census)
    if edge_hash!=EXPECTED_EDGE_SHA256: raise RuntimeError(f"Edge v2 SHA-256 mismatch: {edge_hash}")
    if census_hash!=EXPECTED_CENSUS_SHA256: raise RuntimeError(f"Census v3 SHA-256 mismatch: {census_hash}")
    declared=a.sha256_manifest.read_text(encoding="utf-8")
    if EXPECTED_EDGE_SHA256 not in declared or EXPECTED_CENSUS_SHA256 not in declared:
        raise RuntimeError("SHA-256 manifest does not declare both frozen handoff hashes")
    qc_text=a.normalization_qc.read_text(encoding="utf-8")
    if "canonical `青` edge/root records = **18**" not in qc_text or "cycles | 0" not in qc_text:
        raise RuntimeError("Normalization QC does not contain the expected 青/cycle handoff assertions")
    master=pd.read_csv(a.master,dtype=str,keep_default_na=False)
    edges=pd.read_csv(a.edges,dtype=str,keep_default_na=False)
    graph_qc=validate_edge_universe(edges,master)
    qing_edges=int(((edges.anchor_char=="青")&(edges.phonophore_root=="青")).sum())
    if qing_edges!=18 or (edges.anchor_char=="靑").any(): raise RuntimeError("Corrected 青-family diagnostic failed")
    cache=a.cache_root/a.model/a.channel
    index=pd.read_csv(cache/"character_index.csv",dtype=str,keep_default_na=False);E0=np.load(cache/"E_raw.npy",mmap_mode="r")
    meta=json.loads((cache/"metadata.json").read_text());
    if not meta.get("complete"):raise RuntimeError("Embedding cache metadata does not say complete")
    if len(E0)!=len(index):raise RuntimeError("Cache/index row mismatch")
    E=E0
    if a.centered_sensitivity:E=np.asarray(E0,dtype=np.float32)-E0.mean(axis=0,keepdims=True)
    required=["char_id","original_character","decomposition_normalized","mandarin_pinyin_toneless"]
    missing=[c for c in required if c not in master];
    if missing:raise ValueError(f"Missing master columns: {missing}")
    # Old/minimal caches may index only identity and text. Enrich without
    # changing cache order so layout-constrained retrieval uses frozen master metadata.
    if "decomposition_normalized" not in index:
        index=index.merge(master[["char_id","decomposition_normalized"]],on="char_id",how="left",validate="one_to_one")
    fingerprints=freeze_inputs(a.output,a.master,a.edges,a.census,a.normalization_qc,a.sha256_manifest,cache/"metadata.json",tag)
    marker=ck/f"{tag}__complete.json"
    if marker.exists() and not a.force: print(f"Already complete after frozen-input validation: {tag}");consolidate(a.output);plots(a.output);return
    started=time.time();print(f"Analyzing raw E(x): {tag}",flush=True)
    edge,delta,protos,keys,rel,fwd,inv,res,ctrl,null,learn=analyze(master,index,E,edges,a.model,analysis_channel,a)
    for name,x in [("edge",edge),("relation",rel),("forward",fwd),("inverse",inv),("residual",res),("control",ctrl),("null",null),("learning",learn)]:atomic_csv(x,ck/f"{tag}__{name}.csv")
    np.savez_compressed(ck/f"{tag}__prototypes.npz",relation_classes=np.asarray(keys),prototypes=protos)
    # Coverage records both the structural universe and offset-eligible subset.
    structural=edges
    outside=edges[edges.anchor_in_population.str.lower()=="false"]
    outside_available=int(outside.anchor_char.isin(set(index.original_character)).sum())
    coverage=pd.DataFrame([{"model":a.model,"channel":analysis_channel,"eligible_characters":len(index),"documented_direct_edges":len(structural),"eligible_direct_edges":len(edge),
      "immediate_families":edge.immediate_family.nunique(),"root_anchor_families":edge.anchor_family.nunique(),"relation_classes":edge.relation_class.nunique(),
      "heldout_test_edges_ge3":int((edge.training_family_count>=3).sum()),"heldout_test_edges_ge5":int((edge.training_family_count>=5).sum()),
      "heldout_test_edges_ge8":int((edge.training_family_count>=8).sum()),"anchors_outside_population":len(outside),
      "outside_anchors_available_in_channel":outside_available,"normalization_corrected_edges_used":int(edge.normalization_applied.str.lower().eq("true").sum()),
      "disputed_edges_used":int(edge.edge_dispute_flag.str.lower().eq("true").sum()),"duplicate_canonical_edges_rejected":0,
      "lineage_graph_valid":graph_qc["lineage_graph_valid"],"corrected_qing_edges_present":qing_edges,
      "cross_reference_definition_rows":int(master.definition_cn.str.contains(ARROW_RE,regex=True,na=False).sum()) if "definition_cn" in master else np.nan}])
    atomic_csv(coverage,ck/f"{tag}__coverage.csv")
    atomic_json({"model":a.model,"channel":analysis_channel,"completed_utc":now(),"runtime_seconds":time.time()-started,"E_representation":"centered sensitivity E_raw" if a.centered_sensitivity else "E_raw; uncentered; no PCA/CCA/W","prototype_unit":"mean edges within family, then mean independent family vectors","seed":a.seed},marker)
    # Combine prototypes into the required NPZ with namespaced keys.
    arrays={}
    for f in sorted(ck.glob("*__prototypes.npz")):
        z=np.load(f);base=f.name.replace("__prototypes.npz","");arrays[base+"__relation_classes"]=z["relation_classes"];arrays[base+"__prototypes"]=z["prototypes"]
    np.savez_compressed(a.output/"semantic_relation_prototypes_v5.npz",**arrays)
    covs=[pd.read_csv(f) for f in sorted(ck.glob("*__coverage.csv"))];atomic_csv(pd.concat(covs,ignore_index=True),a.output/"channel_coverage_and_eligibility_v5.csv")
    consolidate(a.output);plots(a.output)
    manifest={"schema_version":5,"script_version":VERSION,"created_utc":now(),"master":a.master.name,"master_sha256":sha256(a.master),
      "authoritative_edges":a.edges.name,"edges_sha256":edge_hash,"census":a.census.name,"census_sha256":census_hash,
      "normalization_qc":a.normalization_qc.name,"normalization_qc_sha256":sha256(a.normalization_qc),
      "sha256_manifest":a.sha256_manifest.name,"sha256_manifest_sha256":sha256(a.sha256_manifest),
      "models":MODELS,"channels":CHANNELS,
      "primary_space":"raw E(x), uncentered","normalization":"cosine calculations normalize operands; offset subtraction uses E_raw","leakage_control":"leave-one-root-anchor-family-out",
      "minimum_other_training_root_families":3,"stricter_sensitivity_thresholds":[5,8],"family_identifier":"frozen family_id_root from edge v2",
      "structural_source_rule":"consume edge v2 directly; no edge reconstruction or hidden normalization",
      "candidate_universes":{"forward":"all channel-eligible characters except input anchor","inverse":"unique eligible documented normalized immediate anchors from edge v2"},
      "identification_rule":"predicted identity is numpy.argmax of exact cosine scores",
      "deterministic_tie_policy":"candidate order is frozen character_index.csv order; numpy.argmax selects the first maximum; true rank counts greater scores plus equal-score candidates earlier in that order",
      "null_repetitions":a.null_reps,"learning_repetitions":a.learning_reps,"seed":a.seed,"interpretation_boundary":"zh_char alone may reflect semantic, distributional, graphic, tokenization, or mixed learned structure; semantic channels are independent probes","W_used":False,"PCA_CCA_SVCCA_used":False,"python":platform.python_version()}
    atomic_json(manifest,a.output/"raw_offset_method_manifest_v5.json")
    qc=f"""# Phase IV-C Raw Offset QC v5

Updated UTC: `{now()}`

- Completed model/channel checkpoints: {len(list(ck.glob('*__complete.json')))}
- Current combination: `{tag}`
- Channel-eligible characters: {len(index):,}
- Documented direct structural edges: {len(structural):,}
- Offset-eligible direct edges (both C and P embedded): {len(edge):,}
- Structurally documented but offset-ineligible here: {len(structural)-len(edge):,}
- Immediate families used: {edge.immediate_family.nunique():,}
- Relation classes: {edge.relation_class.nunique():,}
- Root anchor families: {edge.anchor_family.nunique():,}
- Held-out edges with >= 3 other training root families: {int((edge.training_family_count>=3).sum()):,}
- Held-out edges with >= 5 other training root families: {int((edge.training_family_count>=5).sum()):,}
- Held-out edges with >= 8 other training root families: {int((edge.training_family_count>=8).sum()):,}
- Outside-population anchors in frozen v2: {len(outside):,}; available in this channel: {outside_available:,}
- Normalization-corrected edges used: {int(edge.normalization_applied.str.lower().eq('true').sum()):,}
- Disputed edges used: {int(edge.edge_dispute_flag.str.lower().eq('true').sum()):,}
- Duplicate canonical edges rejected: 0
- Graph/lineage validation: PASS (no self-edges, duplicate derived endpoints, or cycles)
- Corrected `青` family: PASS ({qing_edges} canonical edges; no canonical `靑` anchors)
- Frozen edge SHA-256: `{edge_hash}`
- Frozen census SHA-256: `{census_hash}`
- Primary vectors: raw, uncentered `E(x)`; no PCA, CCA, SVCCA, or learned W.
- Prototype weighting: edges averaged within root family, then family means averaged.
- Direct relations come exclusively from `relational_direct_edges_v2.csv`; recursive lineage depth is retained but edges are never collapsed.
- Google direct-character gloss is blocked unless explicitly enabled as exploratory.
- Identification headline: `argmax_P cos(E(P), E(C*) - r_S) = P*`, evaluated on unseen root families.
- Deterministic ties: frozen `character_index.csv` order; first maximum wins; ranks use the identical ordering.
- Baseline and offset candidate universes are identical; exact argmax/rank/Top-1 invariants: PASS.
"""
    (a.output/"raw_offset_QC_v5.md").write_text(qc,encoding="utf-8")
    print(f"Done in {(time.time()-started)/60:.1f} min: {tag}")

if __name__=="__main__":main()
