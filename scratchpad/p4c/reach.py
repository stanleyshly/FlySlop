import numpy as np
from backend.fly_env import *
env=FlyTypingEnv(action_mode="claw")
env.reset(seed=0,options={"target":"a","start":(0.0,-2.4)})
body=env.body.copy()
r=env.sim.params.tip_radius
for leg in ("LF","RF"):
    i=LEGS.index(leg); a=env.anchor_world(leg)
    for dx in np.arange(-6,6.1,1.0):
      row=[]
      for dy in (-2,0,2):
        for z in (-0.05,0.1):
            t=env._physical(np.array([[*a,0]]*6))
            t[i]=[a[0]+dx,a[1]+dy,z+r]
            q=env.ik.solve(body,t,env.q,iterations=40)
            e=np.linalg.norm(env.ik.tips(body,q)[i]-t[i])
            row.append(round(e,2))
      print(leg,dx,row)
