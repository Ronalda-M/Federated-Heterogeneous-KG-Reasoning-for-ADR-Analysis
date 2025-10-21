import os, json, argparse, numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score, roc_auc_score
COLORS={"Proposed":"#17becf"}
def apply_temp(p, cal_json):
    cal=json.load(open(cal_json)); T=cal.get("T",1.0); eps=1e-6; z=np.log(np.clip(p,eps,1-eps)/np.clip(1-p,eps,1-eps)); return 1/(1+np.exp(-z/T))
def main(pred, labels, out, cal=None):
    os.makedirs(out, exist_ok=True); p=np.load(pred); y=np.load(labels); 
    if cal: p=apply_temp(p, cal)
    # ROC
    fpr,tpr,_=roc_curve(y,p); plt.figure(figsize=(5,4)); plt.plot(fpr,tpr,label=f"Proposed (AUC={auc(fpr,tpr):.3f})", color=COLORS["Proposed"]); plt.plot([0,1],[0,1],'--',color='#999'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC — Proposed'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out,'roc.png'))
    # PR
    P,R,_=precision_recall_curve(y,p); plt.figure(figsize=(5,4)); plt.plot(R,P,label=f"Proposed (AP={auc(R,P):.3f})", color=COLORS["Proposed"]); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('Precision–Recall — Proposed'); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out,'pr.png'))
    # Calibration
    def cal_curve(y,s,bins=10):
        edges=np.linspace(0,1,bins+1); idx=np.digitize(s,edges)-1; xs=[]; ys=[]
        for b in range(bins):
            m=idx==b
            if m.sum()==0: continue
            xs.append(s[m].mean()); ys.append(y[m].mean())
        return np.array(xs), np.array(ys)
    xs,ys=cal_curve(y,p,10); plt.figure(figsize=(5,4)); plt.plot(xs,ys,marker='o',color=COLORS['Proposed']); plt.plot([0,1],[0,1],'--',color='#999'); plt.xlabel('Mean predicted'); plt.ylabel('Observed rate'); plt.title('Calibration — Proposed'); plt.tight_layout(); plt.savefig(os.path.join(out,'cal.png'))
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--pred",required=True); ap.add_argument("--labels",required=True); ap.add_argument("--out",required=True); ap.add_argument("--cal"); a=ap.parse_args(); main(a.pred,a.labels,a.out,a.cal)
