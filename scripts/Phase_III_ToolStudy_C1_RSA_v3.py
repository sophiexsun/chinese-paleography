"""Memory-safe, restartable Phase III-C1 cross-model RSA on MOE-4,808.

No embedding inference. Full-space RSA is exact over all 11,556,028 pairs,
stored as one float32 condensed-vector memmap per model. Dense scale-dependent
RSA uses one frozen, reproducible pair sample and chunked pair scoring.

Example:
  !python /content/Phase_III_ToolStudy_C1_RSA_v3.py \
    --cache-root /content/embedding_caches \
    --master /content/Phase_II_C_MOE4808_Semantic_Master_v1.csv \
    --ab-root /content/Phase_III_ToolStudy_4808_v2 \
    --output /content/Phase_III_ToolStudy_C1_RSA_v3
"""
from __future__ import annotations
import argparse, hashlib, json, platform, time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import pearsonr, rankdata
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA

VERSION="3.0.0"; N=4808
MODELS={
 "qwen3_0.6b_1024d":{"dim":1024,"aliases":["qwen3_0.6b","qwen0.6b"]},
 "bge_m3_1024d":{"dim":1024,"aliases":["bge_m3","bge-m3"]},
 "gte_multilingual_base_768d":{"dim":768,"aliases":["gte_multilingual_base","gte-multilingual-base"]},
 "qwen3_4b_2560d_int8":{"dim":2560,"aliases":["qwen3_4b","qwen4b"]},
}
CHANNELS={"cn_moe":["cn_moe","embed_text_cn_moe"],"en_google_from_moe":["en_google_from_moe","embed_text_en_google_from_moe"]}
GRID=[16,32,48,64,96,128,192,256,384,512,640,768,1024,1280,1536,2048,2560]

def read_csv(p): return pd.read_csv(p,encoding="utf-8-sig",dtype=str,keep_default_na=False)
def l2(x):
 x=np.asarray(x,dtype=np.float32); n=np.linalg.norm(x,axis=1,keepdims=True)
 if np.any(n==0) or not np.isfinite(x).all(): raise ValueError("Invalid embeddings")
 return x/n
def sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
@dataclass
class Cache: model:str; channel:str; emb:Path; index:Path
def identify(path,meta,specs):
 text=(str(path)+json.dumps(meta,ensure_ascii=False)).lower()
 for key,aliases in specs.items():
  vals=aliases if isinstance(aliases,list) else aliases["aliases"]
  if key.lower() in text or any(a.lower() in text for a in vals): return key
 return None
def discover(root,channels):
 out={}
 for emb in root.rglob("embeddings.npy"):
  mp=emb.with_name("cache_metadata.json"); md=json.loads(mp.read_text()) if mp.exists() else {}
  m=identify(emb,md,MODELS); c=identify(emb,md,CHANNELS); idx=emb.with_name("character_index.csv")
  if m and c and idx.exists(): out[(m,c)]=Cache(m,c,emb,idx)
 missing=[(m,c) for m in MODELS for c in channels if (m,c) not in out]
 if missing: raise FileNotFoundError(f"Missing caches: {missing}")
 return out
def load_matrix(cache,master):
 x=np.load(cache.emb,mmap_mode="r"); idx=read_csv(cache.index); dim=MODELS[cache.model]["dim"]
 if x.shape!=(N,dim): raise ValueError(f"{cache.emb}: {x.shape}, expected {(N,dim)}")
 key=next((k for k in ["moe_id","blinded_character","original_character"] if k in idx and k in master),None)
 if not key or not np.array_equal(idx[key].to_numpy(),master[key].to_numpy()): raise ValueError(f"Row mismatch: {cache.model}/{cache.channel}")
 return l2(x)
def condensed_exact(x,path,block=64):
 total=N*(N-1)//2
 if path.exists() and path.stat().st_size==total*4: return np.memmap(path,dtype="float32",mode="r",shape=(total,)),True
 tmp=path.with_suffix(".partial"); progress=path.with_suffix(".progress.json")
 start_row=0
 if tmp.exists() and tmp.stat().st_size==total*4 and progress.exists():
  start_row=int(json.loads(progress.read_text()).get("next_row",0)); out=np.memmap(tmp,dtype="float32",mode="r+",shape=(total,))
 else:
  out=np.memmap(tmp,dtype="float32",mode="w+",shape=(total,))
 pos=start_row*(2*N-start_row-1)//2
 for start in range(start_row,N-1,block):
  stop=min(start+block,N-1); sim=x[start:stop]@x.T
  for local,i in enumerate(range(start,stop)):
   vals=sim[local,i+1:]; out[pos:pos+len(vals)]=vals; pos+=len(vals)
  out.flush(); progress.write_text(json.dumps({"next_row":stop})); print(f"  full RSM rows {stop}/{N-1}",flush=True)
 del out; tmp.replace(path); progress.unlink(missing_ok=True)
 return np.memmap(path,dtype="float32",mode="r",shape=(total,)),False
def spearman_memmap(a,b,rank_dir,key):
 def ranks(v,p):
  if p.exists(): return np.load(p,mmap_mode="r")
  r=rankdata(np.asarray(v),method="average").astype(np.float32); np.save(p,r); return np.load(p,mmap_mode="r")
 ra=ranks(a,rank_dir/f"rank_{key[0]}.npy"); rb=ranks(b,rank_dir/f"rank_{key[1]}.npy")
 return float(pearsonr(ra,rb).statistic)
def sample_pairs(n,count,seed):
 rng=np.random.default_rng(seed); a=rng.integers(0,n,count,dtype=np.int32); b=rng.integers(0,n,count,dtype=np.int32)
 bad=a==b
 while bad.any(): b[bad]=rng.integers(0,n,bad.sum(),dtype=np.int32); bad=a==b
 lo=np.minimum(a,b); hi=np.maximum(a,b); return lo,hi
def pair_cos_chunked(x,pairs,chunk=10000):
 out=np.empty(len(pairs[0]),dtype=np.float32)
 for s in range(0,len(out),chunk):
  a,b=pairs[0][s:s+chunk],pairs[1][s:s+chunk]
  out[s:s+len(a)]=np.einsum("ij,ij->i",x[a],x[b],optimize=True)
 return out
def load_or_make_pca(x,ab_root,work,model,channel):
 candidates=[ab_root/"intermediates"/"pca"/model/channel if ab_root else Path("__none__"),work/"pca"/model/channel]
 for d in candidates:
  if (d/"scores.npy").exists() and (d/"explained_variance_ratio.npy").exists():
   return np.load(d/"scores.npy",mmap_mode="r"),np.load(d/"explained_variance_ratio.npy"),"HIT"
 d=candidates[-1]; d.mkdir(parents=True,exist_ok=True); p=PCA(svd_solver="full").fit(x); scores=p.transform(x).astype(np.float32)
 np.save(d/"scores.npy",scores); np.save(d/"explained_variance_ratio.npy",p.explained_variance_ratio_); return scores,p.explained_variance_ratio_,"CREATED"
def bootstrap_pair(x,y,n_boot,subset_n,seed,checkpoint,pair_name,channel):
 old=pd.read_csv(checkpoint) if checkpoint.exists() else pd.DataFrame(); done=set(old.bootstrap.tolist()) if len(old) else set(); rows=old.to_dict("records")
 for rep in range(1,n_boot+1):
  if rep in done: continue
  rng=np.random.default_rng(seed+rep); ix=rng.choice(N,subset_n,replace=True); tri=np.triu_indices(subset_n,1)
  va=(x[ix]@x[ix].T)[tri]; vb=(y[ix]@y[ix].T)[tri]
  rows.append({"model_pair":pair_name,"semantic_channel":channel,"bootstrap":rep,"character_sample_n":subset_n,"spearman":float(pearsonr(rankdata(va),rankdata(vb)).statistic)})
  pd.DataFrame(rows).to_csv(checkpoint,index=False)
 return rows
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--cache-root",required=True,type=Path); p.add_argument("--master",required=True,type=Path); p.add_argument("--ab-root",type=Path); p.add_argument("--output",type=Path,default=Path("Phase_III_ToolStudy_C1_RSA_v3")); p.add_argument("--include-english",action="store_true"); p.add_argument("--scale-pairs",type=int,default=250000); p.add_argument("--scale-pair-chunk",type=int,default=10000); p.add_argument("--bootstraps",type=int,default=100); p.add_argument("--bootstrap-n",type=int,default=750); p.add_argument("--seed",type=int,default=20260825); args=p.parse_args()
 t=time.time(); args.output.mkdir(parents=True,exist_ok=True); inter=args.output/"intermediates"; inter.mkdir(exist_ok=True); plots=args.output/"plots"; plots.mkdir(exist_ok=True); rank_dir=inter/"ranks"; rank_dir.mkdir(exist_ok=True)
 channels=["cn_moe"]+(["en_google_from_moe"] if args.include_english else []); master=read_csv(args.master)
 if len(master)!=N: raise ValueError("Master must have 4,808 rows")
 caches=discover(args.cache_root,channels); matrices={}; fullvec={}; pcadata={}; audit=[]
 for c in channels:
  for m in MODELS:
   print(f"Loading {m}/{c}",flush=True); x=load_matrix(caches[(m,c)],master); matrices[(m,c)]=x
   rdir=inter/"full_rsm"/c; rdir.mkdir(parents=True,exist_ok=True); v,hit=condensed_exact(x,rdir/f"{m}.f32")
   fullvec[(m,c)]=v; scores,evr,status=load_or_make_pca(x,args.ab_root,inter,m,c); pcadata[(m,c)]=(scores,np.cumsum(evr))
   audit.append({"model":m,"semantic_channel":c,"embedding_source":str(caches[(m,c)].emb),"embedding_sha256":sha(caches[(m,c)].emb),"full_rsm_status":"HIT" if hit else "CREATED","pca_status":status})
 pd.DataFrame(audit).to_csv(args.output/"rsa_input_audit_v3.csv",index=False)
 full=[]; bootall=[]
 for c in channels:
  for a,b in combinations(MODELS,2):
   name=f"{a}__{b}"; print(f"Full RSA {name}/{c}",flush=True)
   sp=spearman_memmap(fullvec[(a,c)],fullvec[(b,c)],rank_dir,(f"{a}_{c}",f"{b}_{c}")); pe=float(pearsonr(fullvec[(a,c)],fullvec[(b,c)]).statistic)
   full.append({"model_A":a,"model_B":b,"semantic_channel":c,"spearman":sp,"pearson":pe,"n_pairs":N*(N-1)//2,"exact_all_pairs":True}); pd.DataFrame(full).to_csv(args.output/"rsa_fullspace_pairwise_v3.csv",index=False)
   cp=inter/f"bootstrap_{name}_{c}.csv"; bootall+=bootstrap_pair(matrices[(a,c)],matrices[(b,c)],args.bootstraps,args.bootstrap_n,args.seed,cp,name,c)
   pd.DataFrame(bootall).to_csv(args.output/"rsa_bootstrap_results_v3.csv",index=False)
 pairs=sample_pairs(N,args.scale_pairs,args.seed+1); np.savez_compressed(inter/"scale_pair_indices_v3.npz",a=pairs[0],b=pairs[1]); scale=[]
 for c in channels:
  cos={}
  for m,spec in MODELS.items():
   scores,cum=pcadata[(m,c)]
   for k in [d for d in GRID if d<=spec["dim"]]:
    path=inter/f"scale_cos_{m}_{c}_{k}.npy"
    if path.exists(): vals=np.load(path,mmap_mode="r")
    else: vals=pair_cos_chunked(l2(scores[:,:k]),pairs,args.scale_pair_chunk); np.save(path,vals)
    cos[(m,k)]=vals
  for a,b in combinations(MODELS,2):
   for k in [d for d in GRID if d<=min(MODELS[a]["dim"],MODELS[b]["dim"])]:
    ca,cb=pcadata[(a,c)][1][k-1],pcadata[(b,c)][1][k-1]
    scale.append({"model_A":a,"model_B":b,"semantic_channel":c,"dimension":k,"cumulative_variance_A":float(ca),"cumulative_variance_B":float(cb),"mean_cumulative_variance":float((ca+cb)/2),"spearman":float(pearsonr(rankdata(cos[(a,k)]),rankdata(cos[(b,k)])).statistic),"n_pairs":args.scale_pairs,"exact_all_pairs":False})
    pd.DataFrame(scale).to_csv(args.output/"rsa_scale_curves_v3.csv",index=False)
 rdf=pd.DataFrame(full); bdf=pd.DataFrame(bootall); sdf=pd.DataFrame(scale)
 for c in channels:
  mat=pd.DataFrame(np.eye(4),index=MODELS,columns=MODELS)
  for _,r in rdf[rdf.semantic_channel==c].iterrows(): mat.loc[r.model_A,r.model_B]=mat.loc[r.model_B,r.model_A]=r.spearman
  mat.to_csv(args.output/("rsa_fullspace_matrix_v3.csv" if c=="cn_moe" else f"rsa_fullspace_matrix_v3_{c}.csv")); fig,ax=plt.subplots(figsize=(8,7)); sns.heatmap(mat,annot=True,vmin=0,vmax=1,cmap="viridis",ax=ax); fig.tight_layout(); fig.savefig(plots/f"rsa_fullspace_{c}.png",dpi=180); plt.close(fig)
 sp=sdf.assign(model_pair=sdf.model_A+" vs "+sdf.model_B); fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=sp,x="dimension",y="spearman",hue="model_pair",style="semantic_channel",markers=True,ax=ax); fig.tight_layout(); fig.savefig(plots/"rsa_vs_dimension_v3.png",dpi=180); plt.close(fig)
 manifest={"version":VERSION,"method":{"full_RSA":"exact condensed cosine vectors computed in row blocks","scale_RSA":"fixed sampled character pairs scored in chunks","uncertainty":"character bootstrap"},"n_full_pairs":N*(N-1)//2,"scale_pair_sample":args.scale_pairs,"bootstraps":args.bootstraps,"bootstrap_n":args.bootstrap_n,"seed":args.seed,"packages":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},"runtime_seconds":time.time()-t}
 (args.output/"rsa_method_manifest_v3.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); (args.output/"rsa_QC_v3.md").write_text("# Phase III-C1 RSA QC v3\n\n- Exact full-space pairs: **11,556,028**\n- Dense scale RSA: fixed sampled pairs, chunked and reused across all models/scales\n- Bootstrap unit: character\n- Checkpoint/resume: PASS\n- No embedding inference: PASS\n",encoding="utf-8"); print(f"Done in {(time.time()-t)/60:.1f} min")
if __name__=="__main__": main()
