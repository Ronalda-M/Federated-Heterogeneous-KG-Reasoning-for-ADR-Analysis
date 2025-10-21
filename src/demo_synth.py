import os, argparse, numpy as np
def main(out):
    rng = np.random.default_rng(42)
    os.makedirs(out, exist_ok=True)
    node_sizes = {"Drug":2000,"ADR":1500,"Outcome":200,"Symptom":600,"Gender":2}
    np.savez_compressed(os.path.join(out,"nodes.npz"), **node_sizes)
    def make_edges(src_n,dst_n,m): 
        src = rng.integers(0,src_n,size=m); dst = rng.integers(0,dst_n,size=m); 
        return np.vstack([src,dst]).astype(np.int64)
    edges = {
      ("Drug","causes","ADR"): make_edges(2000,1500,6000),
      ("ADR","resulted_in","Outcome"): make_edges(1500,200,2000),
      ("Drug","consumed_for","Symptom"): make_edges(2000,600,4000),
      ("Drug","consumed_by","Gender"): make_edges(2000,2,2500),
      ("Gender","affected_by","ADR"): make_edges(2,1500,2000),
    }
    def make_attr(m):
        rng = np.random.default_rng(43)
        count = np.clip(np.round(np.abs(rng.normal(3,2,size=m))),1,50)
        conf = np.clip(rng.normal(0.8,0.1,size=m),0.1,0.99)
        recency = np.clip(rng.exponential(365,size=m),1,2000)
        dosage = rng.integers(0,3,size=m); route = rng.integers(0,3,size=m); source = rng.integers(0,3,size=m)
        return np.vstack([count,conf,recency,dosage,route,source]).T.astype(np.float32)
    for (s,r,t), ei in edges.items():
        base = f"edge__{s}__{r}__{t}"
        np.savez_compressed(os.path.join(out,f"{base}.npz"), edge_index=ei, edge_attr=make_attr(ei.shape[1]))
    M = edges[("Drug","causes","ADR")].shape[1]
    perm = np.random.permutation(M); tr, va = int(0.7*M), int(0.85*M)
    np.savez_compressed(os.path.join(out,"split_causes.npz"), train=perm[:tr], val=perm[tr:va], test=perm[va:])
    for k,n in node_sizes.items():
        x = np.random.normal(size=(n,64 if k in ("Drug","ADR") else 16)).astype(np.float32)
        np.savez_compressed(os.path.join(out,f"feat__{k}.npz"), x=x)
    print("synthetic data at", out)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); a=ap.parse_args(); main(a.out)
