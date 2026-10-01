import sys, time, torch
sys.path.insert(0,'.')
import backend.connectome.runtime as rt
torch.set_num_threads(4)
m=rt.SparseRecurrent('data/connectome/typing_circuit.npz',backend='csr'); p=m._plan()
def t(f,k=100):
    f();best=1e9
    for _ in range(5):
        s=time.perf_counter()
        for _ in range(k):f()
        best=min(best,(time.perf_counter()-s)/k*1e3)
    return best
for B in (1,2,4,8,16):
    G=torch.randn(m.n,B);R=torch.randn(m.n,B)
    print(B,"sampled %.3f naive %.3f"%(t(lambda:p.edge_dots(G,R)),t(lambda:(G[m.post]*R[m.pre]).sum(1))))
