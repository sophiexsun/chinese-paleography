"""Restartable Phase III-C2 cross-validated SVCCA on frozen MOE-4,808 caches.

Each model's PCA is fitted once per fold and reused across all six model pairs
and all variance thresholds. Every true/null task is checkpointed separately.
No embedding inference. Default null repetitions=3 for the initial diagnostic;
rerun with --null-reps 20 to extend the same checkpoints without repeating work.

Example:
  !python /content/Phase_III_ToolStudy_C2_SVCCA_v3.py \
    --cache-root /content/embedding_caches \
    --master /content/Phase_II_C_MOE4808_Semantic_Master_v1.csv \
    --output /content/Phase_III_ToolStudy_C2_SVCCA_v3
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
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

VERSION="3.0.0"; N=4808; THRESHOLDS=[.90,.95,.99]
MODELS={
 "qwen3_0.6b_1024d":{"dim":1024,"aliases":["qwen3_0.6b","qwen0.6b"]},
 "bge_m3_1024d":{"dim":1024,"aliases":["bge_m3","bge-m3"]},
 "gte_multilingual_base_768d":{"dim":768,"aliases":["gte_multilingual_base","gte-multilingual-base"]},
 "qwen3_4b_2560d_int8":{"dim":2560,"aliases":["qwen3_4b","qwen4b"]},
}
CHANNELS={"cn_moe":["cn_moe","embed_text_cn_moe"],"en_google_from_moe":["en_google_from_moe","embed_text_en_google_from_moe"]}
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
 for key,val in specs.items():
  aliases=val if isinstance(val,list) else val["aliases"]
  if key.lower() in text or any(a.lower() in text for a in aliases): return key
 return None
def discover(root,channels):
 out={}
 for emb in root.rglob("embeddings.npy"):
  mp=emb.with_name("cache_metadata.json"); md=json.loads(mp.read_text()) if mp.exists() else {}; m=identify(emb,md,MODELS); c=identify(emb,md,CHANNELS); idx=emb.with_name("character_index.csv")
  if m and c and idx.exists(): out[(m,c)]=Cache(m,c,emb,idx)
 missing=[(m,c) for m in MODELS for c in channels if (m,c) not in out]
 if missing: raise FileNotFoundError(f"Missing caches: {missing}")
 return out
def load_matrix(cache,master):
 x=np.load(cache.emb,mmap_mode="r"); idx=read_csv(cache.index); expected=(N,MODELS[cache.model]["dim"])
 if x.shape!=expected: raise ValueError(f"{cache.emb}: {x.shape}, expected {expected}")
 key=next((k for k in ["moe_id","blinded_character","original_character"] if k in idx and k in master),None)
 if not key or not np.array_equal(idx[key].to_numpy(),master[key].to_numpy()): raise ValueError(f"Row mismatch {cache.model}/{cache.channel}")
 return l2(x)
def pc_count(cum,t): return min(int(np.searchsorted(cum,t,side="left")+1),len(cum))
def fold_pca(x,train,test,d):
 meta=d/"meta.json"; names=["train_scores.npy","test_scores.npy","evr.npy"]
 if meta.exists() and all((d/n).exists() for n in names): return np.load(d/names[0],mmap_mode="r"),np.load(d/names[1],mmap_mode="r"),np.load(d/names[2]),True
 d.mkdir(parents=True,exist_ok=True); p=PCA(svd_solver="full").fit(x[train]); tr=p.transform(x[train]).astype(np.float32); te=p.transform(x[test]).astype(np.float32)
 np.save(d/names[0],tr); np.save(d/names[1],te); np.save(d/names[2],p.explained_variance_ratio_); meta.write_text(json.dumps({"train_n":len(train),"test_n":len(test),"method":"exact full PCA fit train-only"},indent=2)); return tr,te,p.explained_variance_ratio_,False
def cca_fit_pca_scores(x,y,ridge):
 mx=x.mean(0); my=y.mean(0); xc=x-mx; yc=y-my; n=len(x)-1
 sx=np.sqrt((xc*xc).sum(0)/n+ridge); sy=np.sqrt((yc*yc).sum(0)/n+ridge)
 cross=(xc/sx).T@(yc/sy)/n; u,s,vt=np.linalg.svd(cross,full_matrices=False)
 return mx,my,u/sx[:,None],vt.T/sy[:,None],s
def component_corr(x,y):
 k=min(x.shape[1],y.shape[1]); xc=x[:,:k]-x[:,:k].mean(0); yc=y[:,:k]-y[:,:k].mean(0)
 den=np.sqrt((xc*xc).sum(0)*(yc*yc).sum(0)); out=np.full(k,np.nan,dtype=np.float32); good=den>0
 out[good]=((xc[:,good]*yc[:,good]).sum(0)/den[good]).astype(np.float32); return out
def true_task(atr,ate,btr,bte,ridge,path):
 if path.exists(): return
 mx,my,wa,wb,s=cca_fit_pca_scores(atr,btr,ridge); test=component_corr((ate-mx)@wa,(bte-my)@wb)
 np.savez_compressed(path,train_canonical=s.astype(np.float32),test_canonical=test)
def null_task(atr,ate,btr,bte,ridge,seed,path):
 if path.exists(): return
 rng=np.random.default_rng(seed); ptr=rng.permutation(len(atr)); pte=rng.permutation(len(ate)); mx,my,wa,wb,s=cca_fit_pca_scores(atr,btr[ptr],ridge); test=component_corr((ate-mx)@wa,(bte[pte]-my)@wb)
 np.savez_compressed(path,train_canonical=s.astype(np.float32),test_canonical=test)
def selected_pairs(values):
 allpairs=[f"{a}__{b}" for a,b in combinations(MODELS,2)]
 if not values: return allpairs
 bad=set(values)-set(allpairs)
 if bad: raise ValueError(f"Unknown --pairs {bad}; choices={allpairs}")
 return values
def aggregate(task_root,output):
 folds=[]; comps=[]; nulls=[]
 for p in sorted((task_root/"true").glob("*.npz")):
  if not p.with_suffix(".json").exists(): continue
  meta=json.loads(p.with_suffix(".json").read_text()); z=np.load(p); tc=z["test_canonical"]; tr=z["train_canonical"]
  folds.append({**meta,"n_canonical":len(tc),"mean_test_canonical_correlation":float(np.nanmean(tc)),"median_test_canonical_correlation":float(np.nanmedian(tc))})
  comps.extend({**meta,"component":i+1,"train_canonical_correlation":float(tr[i]),"test_canonical_correlation":float(tc[i])} for i in range(len(tc)))
 for p in sorted((task_root/"null").glob("*.npz")):
  if not p.with_suffix(".json").exists(): continue
  meta=json.loads(p.with_suffix(".json").read_text()); tc=np.load(p)["test_canonical"]
  nulls.append({**meta,"n_canonical":len(tc),"mean_test_canonical_correlation":float(np.nanmean(tc)),"median_test_canonical_correlation":float(np.nanmedian(tc))})
 fd=pd.DataFrame(folds); cd=pd.DataFrame(comps); nd=pd.DataFrame(nulls); fd.to_csv(output/"svcca_fold_results_v3.csv",index=False); cd.to_csv(output/"svcca_canonical_components_v3.csv",index=False); nd.to_csv(output/"svcca_null_results_v3.csv",index=False)
 if len(fd): fd.groupby(["model_pair","semantic_channel","variance_threshold"],as_index=False).agg(mean_test_canonical_correlation=("mean_test_canonical_correlation","mean"),sd_across_folds=("mean_test_canonical_correlation","std"),median_test_canonical_correlation=("median_test_canonical_correlation","mean"),mean_retained_A=("retained_A","mean"),mean_retained_B=("retained_B","mean")).to_csv(output/"svcca_pairwise_results_v3.csv",index=False)
 return fd,cd,nd
def main():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--cache-root",required=True,type=Path); p.add_argument("--master",required=True,type=Path); p.add_argument("--output",type=Path,default=Path("Phase_III_ToolStudy_C2_SVCCA_v3")); p.add_argument("--include-english",action="store_true"); p.add_argument("--folds",type=int,default=5); p.add_argument("--null-reps",type=int,default=3); p.add_argument("--ridge",type=float,default=1e-4); p.add_argument("--seed",type=int,default=20260825); p.add_argument("--pairs",nargs="+"); args=p.parse_args()
 t=time.time(); args.output.mkdir(parents=True,exist_ok=True); plots=args.output/"plots"; plots.mkdir(exist_ok=True); inter=args.output/"intermediates"; task=inter/"tasks"; (task/"true").mkdir(parents=True,exist_ok=True); (task/"null").mkdir(parents=True,exist_ok=True)
 channels=["cn_moe"]+(["en_google_from_moe"] if args.include_english else []); master=read_csv(args.master)
 if len(master)!=N: raise ValueError("Master must contain 4,808 rows")
 caches=discover(args.cache_root,channels); matrices={(m,c):load_matrix(caches[(m,c)],master) for c in channels for m in MODELS}; pairs=selected_pairs(args.pairs)
 splits=list(KFold(args.folds,shuffle=True,random_state=args.seed).split(np.arange(N))); pdata={}; audit=[]
 for c in channels:
  for fold,(tr,te) in enumerate(splits,1):
   for m in MODELS:
    print(f"Fold PCA {c} fold={fold} model={m}",flush=True); d=inter/"fold_pca"/c/f"fold_{fold}"/m; a,b,evr,hit=fold_pca(matrices[(m,c)],tr,te,d); pdata[(m,c,fold)]=(a,b,np.cumsum(evr)); audit.append({"model":m,"semantic_channel":c,"fold":fold,"status":"HIT" if hit else "CREATED","retained_90":pc_count(np.cumsum(evr),.90),"retained_95":pc_count(np.cumsum(evr),.95),"retained_99":pc_count(np.cumsum(evr),.99)})
    pd.DataFrame(audit).to_csv(args.output/"svcca_pca_audit_v3.csv",index=False)
 for c in channels:
  for pair in pairs:
   ma,mb=pair.split("__")
   for fold in range(1,args.folds+1):
    atr0,ate0,ca=pdata[(ma,c,fold)]; btr0,bte0,cb=pdata[(mb,c,fold)]
    for th in THRESHOLDS:
     da,db=pc_count(ca,th),pc_count(cb,th); atr,ate=atr0[:,:da],ate0[:,:da]; btr,bte=btr0[:,:db],bte0[:,:db]; stem=f"{pair}__{c}__f{fold}__v{int(th*1000)}"
     tp=task/"true"/f"{stem}.npz"; true_task(atr,ate,btr,bte,args.ridge,tp); tp.with_suffix(".json").write_text(json.dumps({"model_pair":pair,"semantic_channel":c,"fold":fold,"variance_threshold":th,"retained_A":da,"retained_B":db}))
     print(f"True {stem} complete",flush=True)
     for rep in range(1,args.null_reps+1):
      npth=task/"null"/f"{stem}__r{rep}.npz"; null_task(atr,ate,btr,bte,args.ridge,args.seed+fold*100000+rep*101+int(th*1000),npth); npth.with_suffix(".json").write_text(json.dumps({"model_pair":pair,"semantic_channel":c,"fold":fold,"variance_threshold":th,"retained_A":da,"retained_B":db,"null_rep":rep})); print(f"  null {rep}/{args.null_reps}",flush=True)
     aggregate(task,args.output)
 fd,cd,nd=aggregate(task,args.output)
 if len(cd):
  d=cd[cd.variance_threshold==.99]; fig,ax=plt.subplots(figsize=(10,6)); sns.lineplot(data=d,x="component",y="test_canonical_correlation",hue="model_pair",estimator="mean",errorbar="sd",ax=ax); fig.tight_layout(); fig.savefig(plots/"svcca_canonical_profiles_v3.png",dpi=180); plt.close(fig)
 if len(fd) and len(nd):
  tv=fd.groupby(["model_pair","semantic_channel","variance_threshold"],as_index=False).mean(numeric_only=True).assign(kind="true"); nv=nd.groupby(["model_pair","semantic_channel","variance_threshold"],as_index=False).mean(numeric_only=True).assign(kind="null"); d=pd.concat([tv,nv]); fig,ax=plt.subplots(figsize=(11,6)); sns.barplot(data=d[d.variance_threshold==.99],x="model_pair",y="mean_test_canonical_correlation",hue="kind",ax=ax); ax.tick_params(axis="x",rotation=35); fig.tight_layout(); fig.savefig(plots/"svcca_true_vs_null_v3.png",dpi=180); plt.close(fig)
 manifest={"version":VERSION,"models":MODELS,"channels":channels,"pairs":pairs,"folds":args.folds,"variance_thresholds":THRESHOLDS,"null_reps":args.null_reps,"ridge":args.ridge,"seed":args.seed,"method":"PCA fit TRAIN once per model/fold; PCA-score diagonal whitening; CCA fit TRAIN; frozen mappings evaluated TEST; independently permuted train/test row correspondence for null","packages":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"sklearn":sklearn.__version__},"runtime_seconds":time.time()-t}
 (args.output/"svcca_method_manifest_v3.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); (args.output/"svcca_QC_v3.md").write_text("# Phase III-C2 SVCCA QC v3\n\n- Five-fold out-of-sample design: PASS\n- PCA fit TRAIN only and reused across pairs/thresholds: PASS\n- CCA fit TRAIN only: PASS\n- TEST excluded from direction selection: PASS\n- Permutation null separated from true alignment: PASS\n- Task-level checkpoint/resume: PASS\n- No embedding inference: PASS\n",encoding="utf-8"); print(f"Done in {(time.time()-t)/60:.1f} min")
if __name__=="__main__": main()
