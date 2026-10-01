import sys, time, torch
sys.path.insert(0,'.')
from backend.connectome.runtime import SparseRecurrent
torch.set_num_threads(int(sys.argv[1]))
def t(f,k=200):
    f();s=time.perf_counter()
    for _ in range(k):f()
    return (time.perf_counter()-s)/k*1e3
for b in ("csr","index_add","dense"):
    m=SparseRecurrent('data/connectome/typing_circuit.npz',backend=b)
    for B in (1,4,16,64):
        x=torch.randn(B,m.n_in)
        def f():
            with torch.no_grad(): m(x)
        def g():
            m.zero_grad(); r,_=m(x); (r**2).sum().backward()
        print(b,B,"fwd %.2f fb %.2f"%(t(f,50),t(g,20)))
