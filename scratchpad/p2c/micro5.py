import sys, time, torch
sys.path.insert(0,'.')
import backend.connectome.runtime as rt
torch.set_num_threads(4)
m=rt.SparseRecurrent('data/connectome/typing_circuit.npz',backend='csr')
def t(f,k=60):
    f();best=1e9
    for _ in range(5):
        s=time.perf_counter()
        for _ in range(k):f()
        best=min(best,(time.perf_counter()-s)/k*1e3)
    return best
for B in (1,2,4,8,16,32):
    x=torch.randn(B,m.n_in); row=[]
    for mv in (0,64):
        rt._MV_MAX_B=mv
        def f():
            with torch.no_grad(): m(x)
        def g():
            m.zero_grad(); r,_=m(x); (r**2).sum().backward()
        row.append("mv<=%d fwd %.2f fb %.2f"%(mv,t(f),t(g,20)))
    print(B,*row)
