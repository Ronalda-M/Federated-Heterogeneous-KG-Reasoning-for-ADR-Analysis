import json, argparse, numpy as np
def temperature_scale(p, y):
    import numpy as np
    eps=1e-6; z=np.log(np.clip(p,eps,1-eps)/np.clip(1-p,eps,1-eps))
    Ts=np.linspace(0.5,5.0,60); bestT=1.0; best=1e9
    for T in Ts:
        q=1/(1+np.exp(-z/T)); brier=np.mean((q-y)**2)
        if brier<best: best, bestT=brier, T
    return bestT
def main(pred, labels, out):
    p=np.load(pred); y=np.load(labels); T=temperature_scale(p,y); json.dump({"method":"temperature","T":float(T)}, open(out,"w"), indent=2)
if __name__=="__main__":
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--pred",required=True); ap.add_argument("--labels",required=True); ap.add_argument("--out",required=True); a=ap.parse_args(); main(a.pred,a.labels,a.out)
