import sys, time, torch
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'.')
from backend.connectome.runtime import SparseRecurrent
torch.set_num_threads(1)
m=SparseRecurrent('data/connectome/thinker_circuit_small.npz',spectral_radius=4.0)
n=m.n;B=32
w=m.edge_weights().detach()
R=torch.randn(n,B)
def t(f,k=30):
    f();s=time.perf_counter()
    for _ in range(k):f()
    return (time.perf_counter()-s)/k*1e3
W=torch.sparse_csr_tensor(m.crow,m.pre,w,size=(n,n))
ref=torch.sparse.mm(W,R)
print("single",t(lambda:torch.sparse.mm(W,R)))
for P in (2,4,8):
    # balanced row splits by edge count
    crow=m.crow; E=w.numel()
    bounds=[0]+[int(torch.searchsorted(crow,torch.tensor(E*i//P))) for i in range(1,P)]+[n]
    ex=ThreadPoolExecutor(P)
    def blk(i):
        a,b=bounds[i],bounds[i+1]; e0,e1=int(crow[a]),int(crow[b])
        S=torch.sparse_csr_tensor(crow[a:b+1]-e0,m.pre[e0:e1],w[e0:e1],size=(b-a,n))
        return torch.sparse.mm(S,R)
    f=lambda:torch.cat(list(ex.map(blk,range(P))))
    print(P,t(f),(f()-ref).abs().max().item())
    # preallocated out
    out=torch.empty(n,B)
    blks=[]
    for i in range(P):
        a,b=bounds[i],bounds[i+1]; e0,e1=int(crow[a]),int(crow[b])
        blks.append((a,b,torch.sparse_csr_tensor(crow[a:b+1]-e0,m.pre[e0:e1],w[e0:e1],size=(b-a,n))))
    def blk2(x):
        a,b,S=x; torch.sparse.mm(S,R) 
    print(P,"prebuilt",t(lambda:list(ex.map(blk2,blks))))
