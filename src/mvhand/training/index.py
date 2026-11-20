from __future__ import annotations
import json, sqlite3, tarfile
from pathlib import Path
from typing import Mapping

SCHEMA="FORM2FRAME_WDS_INDEX_V1"
def build_index(db_path,dataset_root,dataset_sha,shards):
    db=sqlite3.connect(db_path); db.executescript("DROP TABLE IF EXISTS members; DROP TABLE IF EXISTS meta; CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT); CREATE TABLE members(split TEXT,key TEXT,suffix TEXT,shard TEXT,off INTEGER,size INTEGER,PRIMARY KEY(split,key,suffix));")
    db.execute("INSERT INTO meta VALUES(?,?)",("schema",SCHEMA)); db.execute("INSERT INTO meta VALUES(?,?)",("dataset_sha256",dataset_sha))
    for split,path,sha,size in shards:
        p=Path(dataset_root)/path
        if p.stat().st_size!=size: raise ValueError("shard size mismatch")
        with tarfile.open(p,"r:") as tar:
            for m in tar:
                if not m.isfile(): continue
                name=Path(m.name).name; key,suffix=name.split(".",1)
                db.execute("INSERT INTO members VALUES(?,?,?,?,?,?)",(split,key,suffix,path,m.offset_data,m.size))
        db.execute("INSERT INTO meta VALUES(?,?)",("shard:"+path,json.dumps({"sha256":sha,"size":size},sort_keys=True)))
    db.commit(); db.close()

class IndexedWDS:
    def __init__(self,db_path,dataset_root,dataset_sha):
        self.db=sqlite3.connect(db_path); self.root=Path(dataset_root); self.open_counts={"train":0,"val":0,"test":0}
        row=self.db.execute("SELECT v FROM meta WHERE k='dataset_sha256'").fetchone()
        if not row or row[0]!=dataset_sha: raise ValueError("index dataset binding mismatch")
    def get(self,split,qualified_key):
        if split == "test": raise PermissionError("TEST access is prohibited in training control")
        return self._get_authorized(split, qualified_key)

    def _get_authorized(self,split,qualified_key):
        shard,key=qualified_key.split("::",1)
        if not shard.startswith(f"DexYCB_mv_{split}-"): raise ValueError("qualified key split mismatch")
        rows=self.db.execute("SELECT suffix,shard,off,size FROM members WHERE split=? AND key=?",(split,key)).fetchall()
        if len(rows)!=9: raise KeyError(qualified_key)
        sample={"__key__":key}
        for suffix,stored,off,size in rows:
            if stored!=shard: continue
            with (self.root/stored).open("rb") as f: f.seek(off); sample[suffix]=f.read(size)
            self.open_counts[split] += 1
        if len(sample)!=10: raise KeyError(qualified_key)
        return sample

    def get_terminal(self, qualified_key, authorization: Mapping[str, object]):
        """Read TEST only through a passed, zero-prior-evaluation gate audit."""
        if (
            authorization.get("schema")
            != "form2frame.three-dataset-r-preterminal-gate-audit.v1"
            or authorization.get("dataset") != "dexycb"
            or not authorization.get("passed")
            or authorization.get("selection") != "seed1 epoch10 HW_GV"
            or int(authorization.get("terminal_model_evaluations_before_audit", -1)) != 0
        ):
            raise PermissionError("TEST reader requires a passed DexYCB preterminal audit")
        return self._get_authorized("test", qualified_key)

    def get_group(self, split, qualified_key, hamer_source):
        from mvhand.data.poem_wds_runtime import decode_canonical_sample
        return decode_canonical_sample(self.get(split,qualified_key), hamer_source, qualified_key=qualified_key)

    def get_terminal_group(self, qualified_key, hamer_source, authorization):
        from mvhand.data.poem_wds_runtime import decode_canonical_sample
        return decode_canonical_sample(
            self.get_terminal(qualified_key, authorization),
            hamer_source,
            qualified_key=qualified_key,
        )
