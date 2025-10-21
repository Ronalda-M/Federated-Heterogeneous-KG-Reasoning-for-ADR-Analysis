import torch, torch.nn as nn, torch.nn.functional as F
class EdgeMLP(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,out_dim),nn.GELU(),nn.Dropout(0.2),nn.Linear(out_dim,out_dim))
    def forward(self,e): return self.net(e)
class RGCNLayer(nn.Module):
    def __init__(self,in_dim,out_dim,num_rels):
        super().__init__(); self.weight=nn.Parameter(torch.randn(num_rels,in_dim,out_dim)*0.02); self.self_lin=nn.Linear(in_dim,out_dim); self.norm=nn.LayerNorm(out_dim)
    def forward(self,x,edge_index,edge_type,edge_emb=None):
        src,dst=edge_index; out=torch.zeros_like(self.self_lin(x))
        for r in range(self.weight.size(0)):
            mask=(edge_type==r)
            if mask.sum()==0: continue
            W=self.weight[r]; m=torch.matmul(x[src[mask]],W)
            if edge_emb is not None: m=m+edge_emb[mask]
            out.index_add_(0,dst[mask],m)
        deg=torch.zeros(x.size(0),device=x.device); deg.index_add_(0,dst,torch.ones_like(dst,dtype=torch.float)); deg=torch.clamp(deg,min=1.).unsqueeze(-1)
        out=out/deg; out=out+self.self_lin(x); return self.norm(F.gelu(out))
class LinkDecoder(nn.Module):
    def __init__(self,dim,num_rels,edge_dim=0):
        super().__init__(); self.R=nn.Parameter(torch.randn(num_rels,dim,dim)*0.02); self.edge_mlp=EdgeMLP(edge_dim,dim) if edge_dim>0 else None
    def forward(self,zs,zt,rel_id,edge_attr=None):
        score=torch.einsum('bd,drd,br->b',zs,self.R[rel_id],zt)
        if self.edge_mlp is not None and edge_attr is not None:
            score=score+(self.edge_mlp(edge_attr)*(zs+zt)/2).sum(-1)
        return torch.sigmoid(score)
class ADRGNN(nn.Module):
    def __init__(self,in_dim,hidden,num_layers,num_rels,edge_dim=0):
        super().__init__(); self.enc=nn.Linear(in_dim,hidden); self.layers=nn.ModuleList([RGCNLayer(hidden,hidden,num_rels) for _ in range(num_layers)]); self.dec=LinkDecoder(hidden,num_rels,edge_dim)
    def forward(self,x,edge_index,edge_type,edge_attr,triples,rel_id):
        h=torch.relu(self.enc(x)); eemb=edge_attr if edge_attr is None else edge_attr
        for g in self.layers: h=g(h,edge_index,edge_type,eemb)
        zs=h[triples[:,0]]; zt=h[triples[:,1]]
        return self.dec(zs,zt,rel_id,None)
