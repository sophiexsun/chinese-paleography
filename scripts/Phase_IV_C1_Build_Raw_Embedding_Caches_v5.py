#!/usr/bin/env python3
"""Build restartable raw and L2-normalized Phase IV embedding caches.

Run one model/channel per Colab job. Examples are printed by --help.
The embedding text is passed exactly as stored: no prompt or context is added.
"""
from __future__ import annotations

import argparse, hashlib, json, os, platform, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERSION = "5.0.0"
CHANNELS = {
    "zh_char": "embed_text_zh_char",
    "en_gloss_unihan": "embed_text_en_unihan",
    "cn_def": "embed_text_cn_moe",
    "en_def_translated_cn_def": "embed_text_en_google_from_moe",
    "en_gloss_google": "embed_text_en_google_character",
}
MODELS = {
    "qwen3_0.6b_1024d": ("Qwen/Qwen3-Embedding-0.6B", 1024, "qwen"),
    "bge_m3_1024d": ("BAAI/bge-m3", 1024, "sentence_transformer"),
    "gte_multilingual_base_768d": ("Alibaba-NLP/gte-multilingual-base", 768, "sentence_transformer"),
    "qwen3_4b_2560d_int8": ("Qwen/Qwen3-Embedding-4B", 2560, "qwen_int8"),
}

def now(): return datetime.now(timezone.utc).isoformat()
def sha256(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(8<<20),b""): h.update(b)
    return h.hexdigest()

def atomic_json(x,path):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding="utf-8")
    os.replace(tmp,path)

def load_table(path):
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)

def google_audit_mask(df, args):
    if not args.google_audit:
        raise ValueError("Exploratory Google gloss requires --google-audit; nonblank alone is not eligibility")
    audit=load_table(args.google_audit)
    if not args.google_audit_key or not args.google_eligibility_column:
        raise ValueError("Provide --google-audit-key and --google-eligibility-column explicitly")
    for c in (args.google_audit_key,args.google_eligibility_column):
        if c not in audit: raise ValueError(f"Google audit missing declared column: {c}")
    master_key=args.google_master_key
    if master_key not in df: raise ValueError(f"Master missing Google audit key: {master_key}")
    if audit[args.google_audit_key].duplicated().any(): raise ValueError("Google audit key is not unique")
    allowed={x.strip() for x in args.google_eligible_values.split(",") if x.strip()}
    if not allowed: raise ValueError("No --google-eligible-values supplied")
    status=dict(zip(audit[args.google_audit_key].astype(str),audit[args.google_eligibility_column].astype(str)))
    mapped=df[master_key].astype(str).map(status).fillna("")
    return mapped.isin(allowed), allowed

def last_token_pool(hidden, attention_mask):
    import torch
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding: return hidden[:, -1]
    seq_lens = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_lens]

class Encoder:
    def __init__(self,key,batch_size,max_length):
        self.key=key; self.name,self.dim,self.kind=MODELS[key]
        self.batch_size=batch_size; self.max_length=max_length
        self.pooling=""
        if self.kind=="sentence_transformer":
            from sentence_transformers import SentenceTransformer
            self.model=SentenceTransformer(self.name, trust_remote_code=True)
            self.pooling="SentenceTransformer model-configured pooling; normalize_embeddings=False"
        else:
            import torch
            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
            self.tok=AutoTokenizer.from_pretrained(self.name, padding_side="left", trust_remote_code=True)
            kwargs={"trust_remote_code":True,"device_map":"auto","torch_dtype":"auto"}
            if self.kind=="qwen_int8":
                kwargs["quantization_config"]=BitsAndBytesConfig(load_in_8bit=True)
            self.model=AutoModel.from_pretrained(self.name,**kwargs).eval()
            self.pooling="last non-padding token"

    def encode(self,texts):
        if self.kind=="sentence_transformer":
            x=self.model.encode(texts,batch_size=self.batch_size,convert_to_numpy=True,
                normalize_embeddings=False,show_progress_bar=False)
            return np.asarray(x,dtype=np.float32)
        import torch
        batch=self.tok(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt")
        device=next(self.model.parameters()).device
        batch={k:v.to(device) for k,v in batch.items()}
        with torch.no_grad(): out=self.model(**batch)
        return last_token_pool(out.last_hidden_state,batch["attention_mask"]).float().cpu().numpy()

def main():
    p=argparse.ArgumentParser(epilog="Example: python %(prog)s --master /content/relational_character_master_5channel_translated_v2.csv --cache-root /content/Phase_IV_Relational_11151_v1/embedding_caches --model qwen3_0.6b_1024d --channel zh_char")
    p.add_argument("--master",type=Path,required=True); p.add_argument("--cache-root",type=Path,required=True)
    p.add_argument("--model",choices=MODELS,required=True); p.add_argument("--channel",choices=CHANNELS,required=True)
    p.add_argument("--batch-size",type=int,default=16); p.add_argument("--max-length",type=int,default=512)
    p.add_argument("--allow-google-exploratory",action="store_true")
    p.add_argument("--google-audit",type=Path)
    p.add_argument("--google-master-key",default="char_id")
    p.add_argument("--google-audit-key",default="")
    p.add_argument("--google-eligibility-column",default="")
    p.add_argument("--google-eligible-values",default="mechanically_eligible_english_looking",
                   help="Comma-separated exact audit statuses; surviving rows remain semantically unvalidated")
    a=p.parse_args()
    if a.channel=="en_gloss_google" and not a.allow_google_exploratory:
        raise SystemExit("en_gloss_google is audit-demoted; pass --allow-google-exploratory explicitly.")
    df=pd.read_csv(a.master,dtype=str,keep_default_na=False)
    for c in ("char_id","original_character",CHANNELS[a.channel]):
        if c not in df: raise ValueError(f"Missing column: {c}")
    if df.char_id.duplicated().any(): raise ValueError("Duplicate char_id")
    text_col=CHANNELS[a.channel]
    eligible=df[text_col].astype(str).str.strip().ne("")
    eligibility_rule="nonblank frozen channel text"
    audit_hash=None
    if a.channel=="en_gloss_google":
        audit_eligible, allowed=google_audit_mask(df,a)
        eligible &= audit_eligible
        eligibility_rule=f"nonblank and audit {a.google_eligibility_column} in {sorted(allowed)}; mechanically eligible, semantically unvalidated"
        audit_hash=sha256(a.google_audit)
    sub=df.loc[eligible,["char_id","original_character",text_col]].copy().reset_index().rename(columns={"index":"master_row"})
    out=a.cache_root/a.model/a.channel; out.mkdir(parents=True,exist_ok=True)
    index_path=out/"character_index.csv"; raw_path=out/"E_raw.npy"; norm_path=out/"E_norm.npy"; done_path=out/"completed.npy"
    previous_meta={}
    if (out/"metadata.json").exists(): previous_meta=json.loads((out/"metadata.json").read_text(encoding="utf-8"))
    if previous_meta:
        if not previous_meta.get("complete",False):
            print("Existing cache is incomplete; resuming pending rows.")
        if not str(previous_meta.get("pooling","")).strip():
            raise RuntimeError("Existing cache metadata lacks pooling provenance; cannot declare v4-compatible reuse")
        checks={"master_sha256":sha256(a.master),"model_key":a.model,"channel":a.channel,
                "native_dimension":MODELS[a.model][1],"max_length":a.max_length}
        for key,want in checks.items():
            if key in previous_meta and previous_meta[key]!=want:
                raise RuntimeError(f"Existing cache metadata mismatch for {key}: {previous_meta[key]!r} != {want!r}")
    if index_path.exists():
        prior=pd.read_csv(index_path,dtype=str,keep_default_na=False)
        if prior[["char_id",text_col]].astype(str).values.tolist()!=sub[["char_id",text_col]].astype(str).values.tolist():
            raise RuntimeError("Frozen eligible rows/text changed; use a new cache directory")
    else: sub.to_csv(index_path,index=False)
    n=len(sub); dim=MODELS[a.model][1]
    if raw_path.exists():
        raw=np.lib.format.open_memmap(raw_path,mode="r+")
        norm=np.lib.format.open_memmap(norm_path,mode="r+")
        done=np.load(done_path,mmap_mode="r+")
        if raw.shape!=(n,dim) or norm.shape!=(n,dim) or done.shape!=(n,): raise RuntimeError("Cache shape mismatch")
    else:
        raw=np.lib.format.open_memmap(raw_path,mode="w+",dtype="float32",shape=(n,dim))
        norm=np.lib.format.open_memmap(norm_path,mode="w+",dtype="float32",shape=(n,dim))
        done=np.lib.format.open_memmap(done_path,mode="w+",dtype="uint8",shape=(n,)); done[:]=0; done.flush()
    started=time.time(); pending=np.flatnonzero(np.asarray(done)==0)
    print(f"{a.model}/{a.channel}: {n:,} eligible; {len(pending):,} pending",flush=True)
    reused_complete_cache=len(pending)==0
    enc=None
    if len(pending): enc=Encoder(a.model,a.batch_size,a.max_length)
    # Contiguous batches make memmap writes simple and restart-safe. Gaps from a
    # rare interrupted flush are handled by selecting only still-pending rows.
    for pos in range(0,len(pending),a.batch_size):
        idx=pending[pos:pos+a.batch_size]; texts=sub.iloc[idx][text_col].astype(str).tolist()
        x=enc.encode(texts)
        if x.shape!=(len(idx),dim): raise RuntimeError(f"Unexpected embedding shape {x.shape}; expected {(len(idx),dim)}")
        if not np.isfinite(x).all(): raise RuntimeError("Nonfinite embedding returned")
        lengths=np.linalg.norm(x,axis=1)
        if np.any(lengths<=0): raise RuntimeError("Zero-norm embedding returned")
        raw[idx]=x; norm[idx]=x/lengths[:,None]; raw.flush(); norm.flush()
        done[idx]=1; done.flush()
        if pos==0 or (pos//a.batch_size+1)%25==0 or pos+len(idx)>=len(pending):
            print(f"  {min(pos+len(idx),len(pending)):,}/{len(pending):,} newly cached",flush=True)
    # Compatibility alias requested by the specification: raw learned E(x).
    alias=out/"embeddings.npy"
    if not alias.exists():
        try: os.link(raw_path,alias)
        except OSError:
            import shutil; shutil.copy2(raw_path,alias)
    expected_pooling=(enc.pooling if enc is not None else previous_meta.get("pooling","validated cached model pooling"))
    meta={"schema_version":5,"script_version":VERSION,"created_or_updated_utc":now(),
      "master":a.master.name,"master_sha256":sha256(a.master),"model_key":a.model,"model_name":MODELS[a.model][0],
      "channel":a.channel,"text_column":text_col,"eligible_characters":n,"native_dimension":dim,
      "eligibility_rule":eligibility_rule,"google_audit_filename":a.google_audit.name if a.google_audit else None,
      "google_audit_sha256":audit_hash,"pooling":expected_pooling,"raw_behavior":"model output before caller L2 normalization; no centering",
      "normalized_behavior":"rowwise L2 normalization of E_raw","dtype":"float32","batch_size":a.batch_size,
      "raw_norm_sample_mean":float(np.linalg.norm(np.asarray(raw[:min(n,1000)]),axis=1).mean()),
      "raw_output_already_unit_normalized":bool(np.allclose(np.linalg.norm(np.asarray(raw[:min(n,1000)]),axis=1),1.0,atol=2e-4)),
      "max_length":a.max_length,"prompt_added":False,"complete":bool(np.asarray(done).all()),
      "python":platform.python_version(),"runtime_seconds_this_invocation":time.time()-started,
      "reused_complete_v4_compatible_cache":reused_complete_cache,
      "reuse_rule":"master hash, exact channel index/text, pooling provenance, raw/normalized shapes, model key/dimension, cache metadata, and completion mask must match"}
    atomic_json(meta,out/"metadata.json")
    print(f"Done: {out}")

if __name__=="__main__": main()
