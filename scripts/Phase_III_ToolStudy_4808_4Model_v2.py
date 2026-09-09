"""Phase III tool study on four frozen MOE-4,808 embedding caches.

This script NEVER embeds text. It characterizes cached matrices using effective
dimensionality, anisotropy, PCA/native-MRL/random-projection scale curves,
cross-model RSA, and out-of-sample SVCCA with permutation nulls.

Expected Phase-II layout (auto-discovered recursively):
  CACHE_ROOT/model_key/channel/embeddings.npy
  CACHE_ROOT/model_key/channel/character_index.csv
  CACHE_ROOT/model_key/channel/cache_metadata.json

Install in Colab:
  %pip install -q numpy pandas scipy scikit-learn matplotlib seaborn tqdm

Example:
  !python Phase_III_ToolStudy_4808_4Model_v2.py \
    --cache-root /content/Phase_II_D_results_v1/embedding_caches \
    --master /content/Phase_II_C_MOE4808_Semantic_Master_v1.csv \
    --members /content/Phase_II_C_Frozen10_Test_Members_v1.csv \
    --controls /content/Phase_II_D_results_v1/Phase_II_D_Control_Indices_v1.npz \
    --output /content/Phase_III_ToolStudy_4808_v2

Primary CN-only run: omit --include-english. Add it to replicate the study on
en_google_from_moe when that cache exists.
"""

from __future__ import annotations

import argparse, hashlib, json, math, platform, sys, time, traceback
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.linalg import eigh
from scipy.stats import pearsonr, spearmanr
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.random_projection import SparseRandomProjection
try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs): return iterable

VERSION = "2.0.0"
MODEL_SPECS = {
    "qwen3_0.6b_1024d": {"dim": 1024, "label": "Qwen3-0.6B", "mrl": True,
                          "aliases": ["qwen3_0.6b", "qwen3_0_6b", "qwen0.6b"]},
    "bge_m3_1024d": {"dim": 1024, "label": "BGE-M3", "mrl": False,
                      "aliases": ["bge_m3", "bge-m3"]},
    "gte_multilingual_base_768d": {"dim": 768, "label": "GTE-multilingual-base", "mrl": False,
                                    "aliases": ["gte_multilingual_base", "gte-multilingual-base"]},
    "qwen3_4b_2560d_int8": {"dim": 2560, "label": "Qwen3-4B INT8", "mrl": True,
                             "aliases": ["qwen3_4b", "qwen3-4b", "qwen4b"]},
}
CHANNEL_SPECS = {
    "cn_moe": ["cn_moe", "embed_text_cn_moe", "cn_definition"],
    "en_google_from_moe": ["en_google_from_moe", "embed_text_en_google_from_moe", "en_definition"],
}
PCA_GRID = [16, 32, 48, 64, 96, 128, 192, 256, 384, 512, 640,
            768, 1024, 1280, 1536, 2048, 2560]
RANDOM_PROJECTION_DIMS = [32, 64, 128, 256, 384, 512, 768, 1024]
VARIANCE_SCALE_CHECKPOINTS = [0.50, 0.60, 0.70, 0.80, 0.85,
                              0.90, 0.95, 0.975, 0.99, 1.00]
KNN_KS = [10, 25, 50]
VAR_THRESHOLDS = [0.90, 0.95, 0.99]

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p, encoding="utf-8-sig", dtype=str, keep_default_na=False)

def l2(x):
    x = np.asarray(x, dtype=np.float32)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(n == 0) or not np.isfinite(x).all(): raise ValueError("Invalid/zero embedding rows")
    return x / n

def safe_name(s): return "".join(c if c.isalnum() or c in "._-" else "_" for c in s)

@dataclass
class Cache:
    model: str; channel: str; emb: Path; index: Path; meta: Path | None

def identify_model(path: Path, meta: dict) -> str | None:
    text = (str(path) + " " + json.dumps(meta, ensure_ascii=False)).lower()
    for key, spec in MODEL_SPECS.items():
        if key.lower() in text or any(a.lower() in text for a in spec["aliases"]): return key
    dim = meta.get("dimension") or meta.get("embedding_dimension")
    return None if dim is None else None

def identify_channel(path: Path, meta: dict) -> str | None:
    text = (str(path) + " " + json.dumps(meta, ensure_ascii=False)).lower()
    for key, aliases in CHANNEL_SPECS.items():
        if key in text or any(a.lower() in text for a in aliases): return key
    return None

def discover_caches(root: Path, channels: list[str], overrides: Path | None) -> dict:
    found = {}
    if overrides:
        cfg = json.loads(overrides.read_text(encoding="utf-8"))
        for model, cmap in cfg.items():
            for channel, item in cmap.items():
                emb = Path(item["embeddings"]); idx = Path(item["character_index"])
                meta = Path(item["metadata"]) if item.get("metadata") else None
                found[(model, channel)] = Cache(model, channel, emb, idx, meta)
    for emb in root.rglob("embeddings.npy"):
        meta = emb.with_name("cache_metadata.json")
        md = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
        model, channel = identify_model(emb, md), identify_channel(emb, md)
        indices = [emb.with_name("character_index.csv"), emb.with_name("validated_semantic_master_index_v1.csv")]
        idx = next((p for p in indices if p.exists()), None)
        if model and channel and idx and (model, channel) not in found:
            found[(model, channel)] = Cache(model, channel, emb, idx, meta if meta.exists() else None)
    missing = [(m, c) for m in MODEL_SPECS for c in channels if (m, c) not in found]
    if missing:
        template = {m: {c: {"embeddings": "/path/embeddings.npy", "character_index": "/path/character_index.csv", "metadata": "/path/cache_metadata.json"} for c in channels} for m in MODEL_SPECS}
        (root / "toolstudy_cache_override_template.json").write_text(json.dumps(template, indent=2), encoding="utf-8")
        raise FileNotFoundError(f"Missing caches {missing}. Edit {root/'toolstudy_cache_override_template.json'} and pass --cache-overrides.")
    return found

def load_master(master_path: Path):
    m = read_csv(master_path)
    if len(m) != 4808: raise ValueError(f"Expected 4808 master rows, found {len(m)}")
    for k in ["moe_id", "blinded_character", "original_character"]:
        if k in m and m[k].duplicated().any(): raise ValueError(f"Master {k} is not unique")
    return m

def load_cache(cache: Cache, master: pd.DataFrame, expected_dim: int):
    x = np.load(cache.emb, mmap_mode="r")
    idx = read_csv(cache.index)
    if x.shape != (4808, expected_dim): raise ValueError(f"{cache.emb}: {x.shape}, expected (4808,{expected_dim})")
    if len(idx) != 4808: raise ValueError(f"{cache.index}: expected 4808 rows")
    key = next((k for k in ["moe_id", "blinded_character", "original_character"] if k in idx and k in master), None)
    if not key or not np.array_equal(idx[key].to_numpy(), master[key].to_numpy()):
        raise ValueError(f"Row identity mismatch for {cache.model}/{cache.channel}; checked {key}")
    arr = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(arr)
    norms = np.linalg.norm(arr, axis=1)
    audit = {"model": cache.model, "semantic_channel": cache.channel, "n_rows": x.shape[0],
             "native_dim": x.shape[1], "dtype": str(x.dtype),
             "l2_normalized_input": bool(np.allclose(norms, 1, atol=2e-4)),
             "n_missing": int(np.isnan(arr).sum()), "n_nonfinite": int((~finite).sum()),
             "cache_source": str(cache.emb), "index_source": str(cache.index),
             "cache_sha256": sha256_file(cache.emb), "index_sha256": sha256_file(cache.index)}
    if audit["n_nonfinite"]: raise ValueError(f"Nonfinite values in {cache.emb}")
    return arr, audit

def fit_pca(x):
    # Exact full spectrum; fit once and reuse. PCA centers observations internally.
    p = PCA(n_components=min(x.shape), svd_solver="full")
    scores = p.fit_transform(x).astype(np.float32)
    return p, scores

def fit_or_load_pca(x, cache_dir: Path):
    cache_dir.mkdir(parents=True,exist_ok=True)
    paths={k:cache_dir/f"{k}.npy" for k in ["scores","explained_variance","explained_variance_ratio","singular_values"]}
    meta=cache_dir/"pca_metadata.json"
    signature={"shape":list(x.shape),"dtype":str(x.dtype),"method":"sklearn PCA full exact SVD; observation centering"}
    if meta.exists() and all(p.exists() for p in paths.values()):
        old=json.loads(meta.read_text(encoding="utf-8"))
        if old.get("signature")==signature:
            return ({k:np.load(p,mmap_mode="r") for k,p in paths.items()},True)
        raise RuntimeError(f"Stale PCA intermediate at {cache_dir}; use a new output directory")
    p,scores=fit_pca(x)
    arrays={"scores":scores,"explained_variance":p.explained_variance_,"explained_variance_ratio":p.explained_variance_ratio_,"singular_values":p.singular_values_}
    for k,v in arrays.items(): np.save(paths[k],v,allow_pickle=False)
    meta.write_text(json.dumps({"signature":signature,"created_by_script":VERSION},indent=2),encoding="utf-8")
    return arrays,False

def pc_count(cum, threshold): return int(np.searchsorted(cum, threshold, side="left") + 1)

def pair_sample(n, count, seed):
    rng = np.random.default_rng(seed); pairs = set()
    while len(pairs) < count:
        a = rng.integers(0, n, count); b = rng.integers(0, n, count)
        good = a != b
        for i, j in zip(a[good], b[good]): pairs.add((min(int(i),int(j)), max(int(i),int(j))))
        if count >= n*(n-1)//2: break
    z = np.array(list(pairs)[:count], dtype=np.int32); return z[:,0], z[:,1]

def pair_cos(x, pairs): return np.einsum("ij,ij->i", x[pairs[0]], x[pairs[1]])

def anisotropy(x, pairs):
    xn = l2(x); vals = pair_cos(xn, pairs); mean = x.mean(axis=0); mn = np.linalg.norm(mean)
    direction = mean / mn if mn else mean
    return {"mean_pairwise_cosine": float(vals.mean()), "median_pairwise_cosine": float(np.median(vals)),
            "sd_pairwise_cosine": float(vals.std(ddof=1)),
            "mean_cosine_to_global_mean_direction": float((xn @ direction).mean()),
            "norm_global_mean_embedding": float(mn)}

def neighbors(x, max_k=50):
    return NearestNeighbors(n_neighbors=max_k+1, metric="cosine", algorithm="brute", n_jobs=-1).fit(x).kneighbors(return_distance=False)[:,1:]

def knn_overlap(ref, cur, k):
    return float(np.mean([len(set(a[:k]).intersection(b[:k]))/k for a,b in zip(ref,cur)]))

def load_families(master, members_path, controls_path):
    members = read_csv(members_path)
    members["master_row_index"] = pd.to_numeric(members.get("master_row_index", ""), errors="coerce")
    if members.master_row_index.isna().any():
        lookup = pd.Series(np.arange(len(master)), index=master.moe_id)
        members["master_row_index"] = members.moe_id.map(lookup)
    members["master_row_index"] = members.master_row_index.astype(int)
    controls = {k: v.astype(np.int64) for k,v in np.load(controls_path).items()}
    return members, controls

def fsc_many(x, ix, chunk=250):
    out=[]
    for s in range(0,len(ix),chunk):
        z=x[ix[s:s+chunk]]; g=z@z.swapaxes(1,2); n=z.shape[1]
        out.extend(((g.sum((1,2),dtype=np.float64)-np.trace(g,axis1=1,axis2=2))/(n*(n-1))).tolist())
    return np.asarray(out)

def family_metrics(x, members, controls, identity):
    rows=[]
    for anchor,g in members.groupby("family_anchor",sort=False):
        fid=g.family_id.iloc[0]; ix=g.sort_values("member_order").master_row_index.to_numpy(int)
        test=float(((x[ix]@x[ix].T).sum()-len(ix))/(len(ix)*(len(ix)-1)))
        null=fsc_many(x,controls[fid]); mean=float(null.mean()); sd=float(null.std(ddof=1)); ge=int((null>=test).sum())
        rows.append({**identity,"family_id":fid,"family_anchor":anchor,"family_size":len(ix),
                     "FSC_test":test,"control_mean":mean,"control_sd":sd,"delta_FSC":test-mean,
                     "z_score":(test-mean)/sd if sd else np.nan,"empirical_upper_tail_p":(1+ge)/(1+len(null))})
    return rows

def geometry_metrics(z, full_cos, full_nn, pairs, identity):
    zn=l2(z); cos=pair_cos(zn,pairs); rows=[]
    rows.append({**identity,"metric":"RSA_vs_full_spearman","value":float(spearmanr(full_cos,cos).statistic)})
    rows.append({**identity,"metric":"RSA_vs_full_pearson","value":float(pearsonr(full_cos,cos).statistic)})
    cur_nn=neighbors(zn,max(KNN_KS))
    for k in KNN_KS: rows.append({**identity,"metric":f"knn_overlap_{k}","value":knn_overlap(full_nn,cur_nn,k)})
    return rows

def scales_for(model, dim):
    del model
    return sorted(set([k for k in PCA_GRID if k <= dim] + [dim]))

def common_pair_scales(model_a, model_b):
    ceiling=min(MODEL_SPECS[model_a]["dim"],MODEL_SPECS[model_b]["dim"])
    return [k for k in PCA_GRID if k <= ceiling]

def save_spectrum_plot(spectrum, out, model, channel):
    d=spectrum[(spectrum.model==model)&(spectrum.semantic_channel==channel)]
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(d.pc,d.explained_variance_ratio,lw=1)
    ax.set(xlabel="Principal component",ylabel="Explained variance ratio",title=f"Scree — {MODEL_SPECS[model]['label']} — {channel}")
    ax.set_yscale("log"); fig.tight_layout(); fig.savefig(out/f"scree_{safe_name(model)}_{channel}.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(d.pc,d.cumulative_variance,lw=1.5)
    for y in [.5,.75,.8,.9,.95,.99]: ax.axhline(y,color="gray",lw=.5,alpha=.4)
    ax.set(xlabel="Principal component",ylabel="Cumulative explained variance",ylim=(0,1.01),title=f"Cumulative variance — {MODEL_SPECS[model]['label']} — {channel}")
    fig.tight_layout(); fig.savefig(out/f"cumulative_variance_{safe_name(model)}_{channel}.png",dpi=180); plt.close(fig)

def corr_rsm(a,b,pairs,method="spearman"):
    x,y=pair_cos(l2(a),pairs),pair_cos(l2(b),pairs)
    return float((spearmanr(x,y) if method=="spearman" else pearsonr(x,y)).statistic)

def rsa_bootstrap(a,b,n_boot,subset_n,seed):
    rng=np.random.default_rng(seed); out=[]; n=len(a)
    for i in range(n_boot):
        ix=rng.choice(n,subset_n,replace=True); tri=np.triu_indices(subset_n,1)
        va=(l2(a[ix])@l2(a[ix]).T)[tri]; vb=(l2(b[ix])@l2(b[ix]).T)[tri]
        out.append(float(spearmanr(va,vb).statistic))
    return out

def invsqrt_cov(c, ridge):
    vals,vecs=eigh(c); vals=np.maximum(vals,ridge); return (vecs*(1/np.sqrt(vals)))@vecs.T

def cca_fit(x,y,ridge=1e-4):
    mx,my=x.mean(0),y.mean(0); xc=x-mx; yc=y-my; n=len(x)-1
    wx=invsqrt_cov(xc.T@xc/n,ridge); wy=invsqrt_cov(yc.T@yc/n,ridge)
    u,s,vt=np.linalg.svd(wx@(xc.T@yc/n)@wy,full_matrices=False)
    return mx,my,wx@u,wy@vt.T,s

def component_corr(x,y):
    k=min(x.shape[1],y.shape[1]); out=np.empty(k)
    for j in range(k):
        sx,sy=x[:,j].std(),y[:,j].std(); out[j]=np.corrcoef(x[:,j],y[:,j])[0,1] if sx and sy else np.nan
    return out

def svcca_pair(a,b,pair_name,channel,folds,thresholds,null_reps,seed,ridge):
    fold_rows=[]; null_rows=[]; component_rows=[]
    kf=KFold(folds,shuffle=True,random_state=seed)
    for fold,(tr,te) in enumerate(kf.split(a),1):
        pa=PCA(svd_solver="full").fit(a[tr]); pb=PCA(svd_solver="full").fit(b[tr])
        for threshold in thresholds:
            da=pc_count(np.cumsum(pa.explained_variance_ratio_),threshold); db=pc_count(np.cumsum(pb.explained_variance_ratio_),threshold)
            # Keep CCA below train rank and cap only when explicitly requested via numerical rank.
            da=min(da,len(tr)-2); db=min(db,len(tr)-2)
            atr=pa.transform(a[tr])[:,:da]; ate=pa.transform(a[te])[:,:da]
            btr=pb.transform(b[tr])[:,:db]; bte=pb.transform(b[te])[:,:db]
            mx,my,wa,wb,train_s=cca_fit(atr,btr,ridge)
            test_c=component_corr((ate-mx)@wa,(bte-my)@wb)
            for j,c in enumerate(test_c,1): component_rows.append({"model_pair":pair_name,"semantic_channel":channel,"fold":fold,"variance_threshold":threshold,"component":j,"test_canonical_correlation":c,"train_canonical_correlation":train_s[j-1]})
            fold_rows.append({"model_pair":pair_name,"semantic_channel":channel,"fold":fold,"variance_threshold":threshold,"retained_A":da,"retained_B":db,"n_canonical":len(test_c),"mean_test_canonical_correlation":float(np.nanmean(test_c)),"median_test_canonical_correlation":float(np.nanmedian(test_c))})
            for rep in range(null_reps):
                rng=np.random.default_rng(seed+fold*100000+rep*101+int(threshold*100))
                ptr=rng.permutation(len(tr)); pte=rng.permutation(len(te))
                nmx,nmy,nwa,nwb,_=cca_fit(atr,btr[ptr],ridge)
                nc=component_corr((ate-nmx)@nwa,(bte[pte]-nmy)@nwb)
                null_rows.append({"model_pair":pair_name,"semantic_channel":channel,"fold":fold,"variance_threshold":threshold,"null_rep":rep+1,"mean_test_canonical_correlation":float(np.nanmean(nc)),"median_test_canonical_correlation":float(np.nanmedian(nc))})
    return fold_rows,null_rows,component_rows

def main():
    p=argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,description=__doc__)
    p.add_argument("--cache-root",required=True,type=Path); p.add_argument("--cache-overrides",type=Path)
    p.add_argument("--master",required=True,type=Path); p.add_argument("--members",required=True,type=Path); p.add_argument("--controls",required=True,type=Path)
    p.add_argument("--output",type=Path,default=Path("Phase_III_ToolStudy_4808_v2")); p.add_argument("--include-english",action="store_true")
    p.add_argument("--seed",type=int,default=20260825); p.add_argument("--pair-sample",type=int,default=250000,help="Pair sample used only for anisotropy and within-model scale preservation")
    p.add_argument("--rsa-pair-sample",type=int,default=0,help="0 = exact all 11.56M upper-triangle pairs (default); positive = documented approximation")
    p.add_argument("--random-projections",type=int,default=20); p.add_argument("--rsa-bootstraps",type=int,default=200); p.add_argument("--rsa-bootstrap-n",type=int,default=750)
    p.add_argument("--svcca-folds",type=int,default=5); p.add_argument("--svcca-null-reps",type=int,default=20); p.add_argument("--cca-ridge",type=float,default=1e-4)
    p.add_argument("--skip-random-projection",action="store_true"); p.add_argument("--skip-svcca",action="store_true")
    args=p.parse_args(); started=time.time(); args.output.mkdir(parents=True,exist_ok=True); plots=args.output/"plots"; plots.mkdir(exist_ok=True)
    channels=["cn_moe"]+(["en_google_from_moe"] if args.include_english else [])
    master=load_master(args.master); members,controls=load_families(master,args.members,args.controls)
    caches=discover_caches(args.cache_root,channels,args.cache_overrides)
    pairs=pair_sample(4808,args.pair_sample,args.seed)
    rsa_pairs=(np.triu_indices(4808,1) if args.rsa_pair_sample==0 else pair_sample(4808,args.rsa_pair_sample,args.seed+1))
    matrices={}; audits=[]; spectra=[]; eff=[]; anis=[]; scale=[]; fam=[]; rand=[]; scores={}; projections={}; variance_map=[]; spec_cum={}
    for channel in channels:
      for model,spec in MODEL_SPECS.items():
        x,audit=load_cache(caches[(model,channel)],master,spec["dim"]); matrices[(model,channel)]=x; audits.append(audit)
        pa,pcahit=fit_or_load_pca(x,args.output/"intermediates"/"pca"/model/channel); score=np.asarray(pa["scores"]); scores[(model,channel)]=score
        ev=np.asarray(pa["explained_variance"]); evr=np.asarray(pa["explained_variance_ratio"]); cum=np.cumsum(evr); spec_cum[(model,channel)]=cum; svals=np.asarray(pa["singular_values"])
        audit["pca_intermediate_status"]="HIT" if pcahit else "CREATED"
        for i in range(len(ev)): spectra.append({"model":model,"semantic_channel":channel,"pc":i+1,"singular_value":svals[i],"eigenvalue":ev[i],"explained_variance_ratio":evr[i],"cumulative_variance":cum[i]})
        deff=float(ev.sum()**2/(ev@ev)); row={"model":model,"semantic_channel":channel,"native_dim":spec["dim"],"participation_ratio_D_eff":deff,"D_eff_over_native_dim":deff/spec["dim"]}
        for t in VARIANCE_SCALE_CHECKPOINTS:
            k=min(pc_count(cum,t),spec["dim"])
            label=str(t*100).rstrip("0").rstrip(".").replace(".","_")
            row[f"PCs_for_{label}pct"]=k
            variance_map.append({"model":model,"semantic_channel":channel,"target_cumulative_variance":t,
                                 "minimum_dimension":k,"actual_cumulative_variance":float(cum[k-1])})
        eff.append(row); anis.append({"model":model,"semantic_channel":channel,"pair_sample_n":len(pairs[0]),**anisotropy(x,pairs)})
        full=l2(x); full_cos=pair_cos(full,pairs); full_nn=neighbors(full,max(KNN_KS)); projections[(model,channel,"full",spec["dim"])]=full
        for k in scales_for(model,spec["dim"]):
            z=full if k==spec["dim"] else l2(score[:,:k]); projections[(model,channel,"PCA",k)]=z
            ident={"model":model,"semantic_channel":channel,"reduction_method":"full" if k==spec["dim"] else "PCA","dimension":k,
                   "cumulative_variance_retained":float(cum[k-1]),"replicate":0}
            scale += geometry_metrics(z,full_cos,full_nn,pairs,ident); fam += family_metrics(z,members,controls,ident)
        if spec["mrl"]:
            for k in [d for d in scales_for(model,spec["dim"]) if 32<=d<spec["dim"]]:
                z=l2(x[:,:k]); ident={"model":model,"semantic_channel":channel,"reduction_method":"native_MRL","dimension":k,
                                      "cumulative_variance_retained":np.nan,"replicate":0}
                scale += geometry_metrics(z,full_cos,full_nn,pairs,ident); fam += family_metrics(z,members,controls,ident)
        if not args.skip_random_projection:
            for k in [d for d in RANDOM_PROJECTION_DIMS if d<=spec["dim"]]:
              for rep in tqdm(range(args.random_projections),desc=f"RP {model}/{channel}/{k}"):
                rp=SparseRandomProjection(n_components=k,density="auto",random_state=args.seed+rep+10000*k)
                z=l2(rp.fit_transform(x)); ident={"model":model,"semantic_channel":channel,"reduction_method":"random_projection","dimension":k,
                                                  "cumulative_variance_retained":np.nan,"replicate":rep+1}
                gm=geometry_metrics(z,full_cos,full_nn,pairs,ident); fr=family_metrics(z,members,controls,ident); scale+=gm; fam+=fr; rand+=gm
    audit_df=pd.DataFrame(audits); spec_df=pd.DataFrame(spectra); eff_df=pd.DataFrame(eff); anis_df=pd.DataFrame(anis); scale_df=pd.DataFrame(scale); fam_df=pd.DataFrame(fam); rand_df=pd.DataFrame(rand); variance_df=pd.DataFrame(variance_map)
    audit_df.to_csv(args.output/"toolstudy_input_audit_v2.csv",index=False); spec_df.to_csv(args.output/"eigenvalue_spectra_v2.csv",index=False); eff_df.to_csv(args.output/"effective_dimensionality_summary_v2.csv",index=False); anis_df.to_csv(args.output/"anisotropy_summary_v2.csv",index=False); scale_df.to_csv(args.output/"scale_curve_all_models_v2.csv",index=False); fam_df.to_csv(args.output/"scale_curve_family_results_v2.csv",index=False); rand_df.to_csv(args.output/"random_projection_controls_v2.csv",index=False); variance_df.to_csv(args.output/"scale_variance_checkpoints_v2.csv",index=False)
    for channel in channels:
      for model in MODEL_SPECS: save_spectrum_plot(spec_df,plots,model,channel)
    # Full-space and scale-dependent cross-model RSA.
    rsa_rows=[]; rsa_scale=[]; boot=[]
    for channel in channels:
      for a,b in combinations(MODEL_SPECS,2):
        pair=f"{a}__{b}"; va=matrices[(a,channel)]; vb=matrices[(b,channel)]
        sp=corr_rsm(va,vb,rsa_pairs,"spearman"); pe=corr_rsm(va,vb,rsa_pairs,"pearson"); rsa_rows.append({"model_A":a,"model_B":b,"semantic_channel":channel,"spearman":sp,"pearson":pe,"pair_sample_n":len(rsa_pairs[0]),"exact_all_pairs":args.rsa_pair_sample==0})
        vals=rsa_bootstrap(va,vb,args.rsa_bootstraps,args.rsa_bootstrap_n,args.seed)
        for i,v in enumerate(vals,1): boot.append({"model_A":a,"model_B":b,"semantic_channel":channel,"bootstrap":i,"spearman":v,"character_sample_n":args.rsa_bootstrap_n})
        for k in common_pair_scales(a,b):
            xa=projections[(a,channel,"PCA",k)]; xb=projections[(b,channel,"PCA",k)]
            va_ret=float(spec_cum[(a,channel)][k-1])
            vb_ret=float(spec_cum[(b,channel)][k-1])
            rsa_scale.append({"model_A":a,"model_B":b,"semantic_channel":channel,"dimension":k,
                              "cumulative_variance_A":va_ret,"cumulative_variance_B":vb_ret,
                              "mean_cumulative_variance":(va_ret+vb_ret)/2,
                              "spearman":corr_rsm(xa,xb,rsa_pairs),"pair_sample_n":len(rsa_pairs[0]),"exact_all_pairs":args.rsa_pair_sample==0})
    rdf=pd.DataFrame(rsa_rows); bdf=pd.DataFrame(boot); rsdf=pd.DataFrame(rsa_scale)
    rdf.to_csv(args.output/"rsa_fullspace_pairwise_v2.csv",index=False); bdf.to_csv(args.output/"rsa_bootstrap_results_v2.csv",index=False); rsdf.to_csv(args.output/"rsa_scale_curves_v2.csv",index=False)
    for channel in channels:
        mat=pd.DataFrame(np.eye(4),index=MODEL_SPECS,columns=MODEL_SPECS)
        for _,r in rdf[rdf.semantic_channel==channel].iterrows(): mat.loc[r.model_A,r.model_B]=mat.loc[r.model_B,r.model_A]=r.spearman
        mat.to_csv(args.output/f"rsa_fullspace_matrix_v2_{channel}.csv")
        if channel=="cn_moe": mat.to_csv(args.output/"rsa_fullspace_matrix_v2.csv")
        fig,ax=plt.subplots(figsize=(8,7)); sns.heatmap(mat,annot=True,vmin=0,vmax=1,cmap="viridis",ax=ax); ax.set_title(f"Full-space RSA (Spearman) — {channel}"); fig.tight_layout(); fig.savefig(plots/f"rsa_fullspace_{channel}.png",dpi=180); plt.close(fig)
    # Out-of-sample SVCCA.
    sfold=[]; snull=[]; scomp=[]
    if not args.skip_svcca:
      for channel in channels:
       for a,b in combinations(MODEL_SPECS,2):
        f,n,c=svcca_pair(matrices[(a,channel)],matrices[(b,channel)],f"{a}__{b}",channel,args.svcca_folds,VAR_THRESHOLDS,args.svcca_null_reps,args.seed,args.cca_ridge); sfold+=f; snull+=n; scomp+=c
    sf=pd.DataFrame(sfold); sn=pd.DataFrame(snull); sc=pd.DataFrame(scomp)
    sf.to_csv(args.output/"svcca_fold_results_v2.csv",index=False); sn.to_csv(args.output/"svcca_null_results_v2.csv",index=False); sc.to_csv(args.output/"svcca_canonical_components_v2.csv",index=False)
    if len(sf): sf.groupby(["model_pair","semantic_channel","variance_threshold"],as_index=False).agg(mean_test_canonical_correlation=("mean_test_canonical_correlation","mean"),sd_across_folds=("mean_test_canonical_correlation","std"),median_test_canonical_correlation=("median_test_canonical_correlation","mean"),mean_retained_A=("retained_A","mean"),mean_retained_B=("retained_B","mean")).to_csv(args.output/"svcca_pairwise_results_v2.csv",index=False)
    else: pd.DataFrame().to_csv(args.output/"svcca_pairwise_results_v2.csv",index=False)
    # Core comparison plots.
    fig,ax=plt.subplots(figsize=(9,5)); sns.barplot(data=eff_df,x="model",y="participation_ratio_D_eff",hue="semantic_channel",ax=ax); ax.tick_params(axis="x",rotation=25); fig.tight_layout(); fig.savefig(plots/"effective_dimensionality_comparison.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); sns.barplot(data=anis_df,x="model",y="mean_pairwise_cosine",hue="semantic_channel",ax=ax); ax.tick_params(axis="x",rotation=25); ax.set_title("Embedding anisotropy / concentration"); fig.tight_layout(); fig.savefig(plots/"anisotropy_comparison.png",dpi=180); plt.close(fig)
    for metric in ["RSA_vs_full_spearman","knn_overlap_25"]:
      d=scale_df[(scale_df.metric==metric)&(scale_df.reduction_method!="random_projection")]
      fig,ax=plt.subplots(figsize=(9,5)); sns.lineplot(data=d,x="dimension",y="value",hue="model",style="reduction_method",markers=True,ax=ax); ax.set_title(metric); fig.tight_layout(); fig.savefig(plots/f"scale_curve_{metric}.png",dpi=180); plt.close(fig)
      pv=d[d.reduction_method.isin(["PCA","full"])].dropna(subset=["cumulative_variance_retained"])
      fig,ax=plt.subplots(figsize=(9,5)); sns.lineplot(data=pv,x="cumulative_variance_retained",y="value",hue="model",markers=True,ax=ax); ax.set_title(f"{metric} by cumulative PCA variance"); ax.set_xlabel("Cumulative PCA variance retained V(k)"); fig.tight_layout(); fig.savefig(plots/f"scale_curve_variance_{metric}.png",dpi=180); plt.close(fig)
      rp=scale_df[(scale_df.metric==metric)&(scale_df.reduction_method=="random_projection")]
      if len(rp):
        fig,ax=plt.subplots(figsize=(9,5)); sns.lineplot(data=rp,x="dimension",y="value",hue="model",estimator="mean",errorbar=("pi",95),marker="o",ax=ax); ax.set_title(f"Random-projection control — {metric} (95% interval)"); fig.tight_layout(); fig.savefig(plots/f"random_projection_{metric}.png",dpi=180); plt.close(fig)
    # Family z-score curves and RSA dimension curves are separate y-axes/figures.
    fd=fam_df[(fam_df.reduction_method!="random_projection")]
    fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=fd,x="dimension",y="z_score",hue="model",style="reduction_method",units="family_id",estimator=None,alpha=.45,ax=ax); ax.axhline(0,color="gray",lw=.7); ax.set_title("Frozen Phase II family signal across scale"); fig.tight_layout(); fig.savefig(plots/"scale_curve_family_z_scores.png",dpi=180); plt.close(fig)
    fdv=fd[fd.reduction_method.isin(["PCA","full"])].dropna(subset=["cumulative_variance_retained"])
    fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=fdv,x="cumulative_variance_retained",y="z_score",hue="model",units="family_id",estimator=None,alpha=.45,ax=ax); ax.axhline(0,color="gray",lw=.7); ax.set_title("Frozen family signal by cumulative PCA variance"); ax.set_xlabel("Cumulative PCA variance retained V(k)"); fig.tight_layout(); fig.savefig(plots/"scale_curve_variance_family_z_scores.png",dpi=180); plt.close(fig)
    frp=fam_df[fam_df.reduction_method=="random_projection"]
    if len(frp):
        fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=frp,x="dimension",y="z_score",hue="model",estimator="mean",errorbar=("pi",95),marker="o",ax=ax); ax.axhline(0,color="gray",lw=.7); ax.set_title("Random-projection family-signal control (families × projections)"); fig.tight_layout(); fig.savefig(plots/"random_projection_family_z_scores.png",dpi=180); plt.close(fig)
    rsplot=rsdf.assign(model_pair=rsdf.model_A+" vs "+rsdf.model_B)
    fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=rsplot,x="dimension",y="spearman",hue="model_pair",style="semantic_channel",markers=True,ax=ax); ax.set_title("Cross-model RSA across common PCA scales"); fig.tight_layout(); fig.savefig(plots/"rsa_vs_dimension.png",dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=rsplot,x="mean_cumulative_variance",y="spearman",hue="model_pair",style="semantic_channel",markers=True,ax=ax); ax.set_title("Cross-model RSA by mean cumulative PCA variance"); ax.set_xlabel("Mean V(k) across model pair"); fig.tight_layout(); fig.savefig(plots/"rsa_vs_cumulative_variance.png",dpi=180); plt.close(fig)
    if len(sc):
        primary=sc[sc.variance_threshold==.99]
        fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=primary,x="component",y="test_canonical_correlation",hue="model_pair",estimator="mean",errorbar="sd",ax=ax); ax.set_title("SVCCA held-out canonical-correlation profiles (99%)"); fig.tight_layout(); fig.savefig(plots/"svcca_canonical_profiles.png",dpi=180); plt.close(fig)
        tv=sf.groupby(["model_pair","semantic_channel","variance_threshold"],as_index=False).mean(numeric_only=True).assign(kind="true")
        nv=sn.groupby(["model_pair","semantic_channel","variance_threshold"],as_index=False).mean(numeric_only=True).assign(kind="null")
        comp=pd.concat([tv,nv],ignore_index=True)
        fig,ax=plt.subplots(figsize=(11,6)); sns.barplot(data=comp[comp.variance_threshold==.99],x="model_pair",y="mean_test_canonical_correlation",hue="kind",ax=ax); ax.tick_params(axis="x",rotation=35); ax.set_title("SVCCA held-out true vs permutation null (99%)"); fig.tight_layout(); fig.savefig(plots/"svcca_true_vs_null.png",dpi=180); plt.close(fig)
    manifest={"script_version":VERSION,"source_dataset":str(args.master),"source_embedding_caches":audit_df[["model","semantic_channel","cache_source","index_source","cache_sha256"]].to_dict("records"),"row_id_validation":"first common key among moe_id, blinded_character, original_character; exact order required","semantic_channels":channels,"models":MODEL_SPECS,"preprocessing":{"PCA":"center original cached rows; exact full SVD; project; L2-renormalize before cosine","native_MRL":"truncate original coordinate prefix only for Qwen checkpoints at supported k>=32; L2-renormalize","random_projection":"sklearn SparseRandomProjection; L2-renormalize"},"dimensions":{"pca_grid":PCA_GRID,"random_projection_subset":RANDOM_PROJECTION_DIMS,"rule":"omit k above native dimension; always include exact full dimension"},"scale_variance_checkpoints":VARIANCE_SCALE_CHECKPOINTS,"svcca_variance_thresholds":VAR_THRESHOLDS,"random_projection":{"n":args.random_projections,"seed_formula":"master_seed + replicate + 10000*k","interpretation":"which subspace at fixed k; not a substitute for dense scale sampling"},"RSA":{"similarity":"cosine","primary":"Spearman","secondary":"Pearson","full_cross_model_pairs":"all upper-triangle pairs" if args.rsa_pair_sample==0 else args.rsa_pair_sample,"within_model_scale_pair_sample":args.pair_sample,"character_bootstraps":args.rsa_bootstraps,"bootstrap_n":args.rsa_bootstrap_n,"scale_axes":["raw dimension k","mean cumulative PCA variance V(k) across pair"]},"SVCCA":{"folds":args.svcca_folds,"thresholds":VAR_THRESHOLDS,"ridge":args.cca_ridge,"null_reps":args.svcca_null_reps,"train_test":"PCA and CCA fit TRAIN only; frozen transforms evaluated TEST"},"seeds":{"master":args.seed},"packages":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},"runtime_seconds":time.time()-started}
    (args.output/"toolstudy_method_manifest_v2.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    qc=["# Phase III Tool Study QC v2","",f"- Matched master rows: **{len(master)}**",f"- Models: **{len(MODEL_SPECS)}**",f"- Channels: **{', '.join(channels)}**",f"- Dense PCA dimension grid: **{PCA_GRID}**",f"- Raw-k and cumulative-variance scale axes recorded: **PASS**",f"- Cache row/dimension/order validation: **PASS**",f"- Original cache matrices overwritten: **NO**",f"- PCA centering and re-normalization documented: **PASS**",f"- Phase II memberships/control indices reused: **PASS**",f"- RSA uncertainty resampled at character level: **PASS**",f"- SVCCA reduction/alignment fitted train-only: **{'SKIPPED' if args.skip_svcca else 'PASS'}**","","## Guardrail","","This run characterizes embedding instruments. It does not test or prove 右文說, 聲符示源, semantic inheritance, or latent component pathways. Dimensional checkpoints identify broad regions or plateaus, not an optimal latent layer."]
    (args.output/"toolstudy_QC_v2.md").write_text("\n".join(qc)+"\n",encoding="utf-8")
    print(f"Completed in {(time.time()-started)/60:.1f} min: {args.output.resolve()}")

if __name__=="__main__": main()
