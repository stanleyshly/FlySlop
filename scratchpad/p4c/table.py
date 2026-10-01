import time, numpy as np
from backend.fly_env import *
env=FlyTypingEnv(action_mode="claw")
env.reset(seed=0,options={"target":"a","start":(0.0,-2.4)})
r=env.sim.params.tip_radius
t0=time.time()
body=env.body.copy()
res={}
for leg in ("LF","RF"):
    i=LEGS.index(leg); a=env.anchor_world(leg)
    T=np.zeros((25,17))
    for ix,dx in enumerate(np.arange(-3,3.01,0.25)):
        for iy,dy in enumerate(np.arange(-2,2.01,0.25)):
            worst=0
            for z in (-0.1,HOVER):
                t=env._physical(np.array([[*a,0]]*6)); t[i]=[a[0]+dx,a[1]+dy,z+r]
                q=env.ik.solve(body,t,env.q,iterations=25)
                worst=max(worst,float(np.linalg.norm(env.ik.tips(body,q)[i]-t[i])))
            T[ix,iy]=worst
    res[leg]=T
print(time.time()-t0)
np.set_printoptions(linewidth=200,precision=2)
for leg in res:
    print(leg); print((res[leg][::2,::2]).T)
print("feasible frac",[(res[l]<0.1).mean() for l in res])
np.save("scratchpad/p4c/table.npy",np.stack([res["LF"],res["RF"]]))
