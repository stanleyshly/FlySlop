import sys, time, torch
sys.path.insert(0,'.')
from backend.connectome.runtime import SparseRecurrent
torch.set_num_threads(int(sys.argv[1]))
m=SparseRecurrent('data/connectome/thinker_circuit_small.npz',spectral_radius=4.0)
n=m.n;B=32
w=m.edge_weights().detach()
R=torch.randn(n,B);G=torch.randn(n,B)
def t(f,k=20):
    f();s=time.perf_counter()
    for _ in range(k):f()
    return (time.perf_counter()-s)/k*1e3
W=torch.sparse_csr_tensor(m.crow,m.pre,w,size=(n,n))
print("mm csr",t(lambda:torch.sparse.mm(W,R)))
print("build",t(lambda:torch.sparse_csr_tensor(m.crow,m.pre,w,size=(n,n))))
print("W@R op",t(lambda:W@R))
Rt=R.T.contiguous()
print("mm B-major?", t(lambda:torch.sparse.mm(W,R)))
WT=torch.sparse_csr_tensor(m.crowT,m.post_T,w[m.perm],size=(n,n))
print("mm T",t(lambda:torch.sparse.mm(WT,G)))
print("w[perm]",t(lambda:w[m.perm]))
print("gvals naive",t(lambda:(G[m.post]*R[m.pre]).sum(1)))
try:
    inp=torch.sparse_csr_tensor(m.crow,m.pre,torch.ones_like(w),size=(n,n))
    Rt=R.T.contiguous()
    print("sampled_addmm",t(lambda:torch.sparse.sampled_addmm(inp,G,Rt)))
    a=torch.sparse.sampled_addmm(inp,G,Rt).values(); b=(G[m.post]*R[m.pre]).sum(1)
    print((a-b).abs().max())
except Exception as e: print("sampled fail",e)
def chunked(cs):
    out=torch.empty_like(w)
    for i in range(0,w.numel(),cs):
        out[i:i+cs]=(G[m.post[i:i+cs]]*R[m.pre[i:i+cs]]).sum(1)
    return out
for cs in (8192,32768,131072): print("chunk",cs,t(lambda:chunked(cs)))
# einsum with bmm trick: gather rows [E,B]
Gt=G.contiguous()
print("gather G",t(lambda:G[m.post]), "gather R",t(lambda:R[m.pre]), "index_select",t(lambda:R.index_select(0,m.pre)))
