import os, argparse, yaml, numpy as np, torch, torch.nn as nn
from .utils import set_seed, get_device
from .model_gnn import ADRGNN
from sklearn.metrics import average_precision_score, roc_auc_score
def load_demo_npz(root):
    feats={}
    for t in ["Drug","ADR","Outcome","Symptom","Gender"]:
        feats[t]=np.load(os.path.join(root,f"feat__{t}.npz"))["x"]
    dat=np.load(os.path.join(root,"edge__Drug__causes__ADR.npz")); E=dat["edge_index"].T
    split=np.load(os.path.join(root,"split_causes.npz")); return feats,E,split
def sample_neg(n_src,n_dst,k):
    src=np.random.randint(0,n_src,size=k); dst=np.random.randint(0,n_dst,size=k); return np.stack([src,dst],axis=1)
def main(config,data_root,out):
    set_seed(config["seed"]); device=get_device(config["device"]); os.makedirs(out,exist_ok=True)
    feats,E,split=load_demo_npz(data_root); x=np.concatenate([feats["Drug"],feats["ADR"]],axis=0); x=torch.from_numpy(x).float().to(device)
    n_drug,n_adr=feats["Drug"].shape[0],feats["ADR"].shape[0]; idx_tr,idx_va,idx_te=split["train"],split["val"],split["test"]
    model=ADRGNN(in_dim=x.size(1),hidden=config["model"]["hidden_dim"],num_layers=config["model"]["num_layers"],num_rels=5,edge_dim=0).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=config["opt"]["lr"],weight_decay=config["opt"]["weight_decay"]); bce=nn.BCELoss()
    def epoch(indices,train=True):
        pos=E[indices]; neg=sample_neg(n_drug,n_adr,len(indices)*config["opt"]["negative_samples"])
        triples=np.vstack([pos, neg + [0,n_drug]]); labels=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))]).astype(np.float32)
        edge_index=torch.from_numpy(np.vstack([E[:,0],E[:,1]+n_drug])).long().to(device); edge_type=torch.zeros(edge_index.size(1),dtype=torch.long,device=device)
        triples=torch.from_numpy(triples).long().to(device); labels=torch.from_numpy(labels).to(device)
        model.train() if train else model.eval()
        with torch.set_grad_enabled(train):
            pred=model(x,edge_index,edge_type,None,triples,rel_id=0); loss=bce(pred,labels)
            if train: opt.zero_grad(); loss.backward(); opt.step()
        return loss.item(), pred.detach().cpu().numpy(), labels.detach().cpu().numpy()
    for ep in range(config["opt"]["epochs"]):
        tr_loss,_,_=epoch(idx_tr,True); va_loss,va_p,va_y=epoch(idx_va,False)
        if ep%5==0: print(f"ep {ep:02d} | train {tr_loss:.3f} | val {va_loss:.3f} | AP {average_precision_score(va_y,va_p):.3f} | ROC {roc_auc_score(va_y,va_p):.3f}")
    _,te_p,te_y=epoch(idx_te,False); np.save(os.path.join(out,"preds_test.npy"),te_p); np.save(os.path.join(out,"labels_test.npy"),te_y); print("saved test predictions", out)
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); ap.add_argument("--data",required=True); ap.add_argument("--out",required=True); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config)); main(cfg,a.data,a.out)
