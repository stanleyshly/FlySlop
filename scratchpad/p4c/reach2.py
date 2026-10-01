import numpy as np
from backend.fly_env import *
env=FlyTypingEnv(action_mode="claw")
env.reset(seed=0,options={"target":"a","start":(0.0,-2.4)})
r=env.sim.params.tip_radius
for zf in (1.0,0.8,0.6,0.45):
  for roll in (0,):
    body=env.body.copy(); body[2]=zf*STAND_HEIGHT; body[5]=roll
    res=[]
    for dx in (3.0,3.5,4.0,4.5,5.0):
      errs=[]
      for leg,sgn in (("LF",-1),("RF",1)):
        i=LEGS.index(leg)
        env.body=body
        a=env.anchor_world(leg)
        t=env._physical(np.array([[*a,0]]*6)); t[i]=[a[0]+sgn*dx,a[1],-0.05+r]
        q=env.ik.solve(body,t,env.q,iterations=60)
        errs.append(round(float(np.linalg.norm(env.ik.tips(body,q)[i]-t[i])),2))
      res.append((dx,errs))
    print(zf,roll,res)
