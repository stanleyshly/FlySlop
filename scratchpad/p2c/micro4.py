import sys, time, torch
sys.path.insert(0,'.')
from backend.connectome.runtime import SparseRecurrent
torch.set_num_threads(4)
m=SparseRecurrent('data/connectome/typing_circuit.npz',backend='csr')
n=m.n;w=m.edge_weights().detach()
def t(f,k=300):
    f();s=time.perf_counter()
    for _ in range(k):f()
    return (time.perf_counter()-s)/k*1e3
W=torch.sparse_csr_tensor(m.crow,m.pre,w,size=(n,n))
for B in (1,16):
    R=torch.randn(n,B)
    print(B,"sparse.mm",t(lambda:torch.sparse.mm(W,R)),"build+mm",t(lambda:torch.sparse.mm(torch.sparse_csr_tensor(m.crow,m.pre,w,size=(n,n)),R)))
    print(B,"addmm",t(lambda:torch.addmm(torch.zeros(n,B),W,R)), "matmul",t(lambda:W@R))
v=torch.randn(n)
print("mv",t(lambda:torch.mv(W,v)), "W@v",t(lambda:W@v))
torch.set_num_threads(1)
print("1thr mv",t(lambda:torch.mv(W,v)),"mm",t(lambda:torch.sparse.mm(W,torch.randn(n,1))))
Wc=W.to_sparse_coo() if False else None
r=torch.randn(1,n)
print("index_add", t(lambda: torch.zeros(1,n).index_add_(1,m.post,r[:,m.pre]*w)))
print("gather", t(lambda: r[:,m.pre]))
