import numpy as np, sys
from backend.fly_env import *
mode,k,seed=sys.argv[1],sys.argv[2],int(sys.argv[3])
env=FlyTypingEnv(action_mode=mode, terminate_on_error=False, physics={"shift_mode":"held"})
env.reset(seed=seed,options={"keys":[(k,True)],"max_steps":200})
for i in range(200):
    a=env.expert_action()
    obs,r,te,tr,info=env.step(a)
    if i%6==0 or info["events"]:
        c=env._current_chord()
        pl=env._chord_plan(c) if c else None
        print(env.tick,np.round(env.body[:2],2),pl and (pl["mod"],pl["mod_leg"],pl["key_leg"],np.round(pl["goal"],2)),dict(env.sim.keyboard.down),"LF",np.round(env.tips[0],2).tolist(),"RF",np.round(env.tips[3],2).tolist(),[(e["type"],e["key_id"]) for e in info["events"]])
    if te or tr: break
