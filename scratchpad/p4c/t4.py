import numpy as np
from backend.fly_env import *
env=FlyTypingEnv(action_mode="mn", terminate_on_error=False, physics={"shift_mode":"held"})
items=[("o",False),("ArrowLeft",True),("ArrowLeft",True),("x",True)]
env.reset(seed=0,options={"keys":items,"max_steps":400})
for i in range(400):
    a=env.expert_action()
    obs,r,te,tr,info=env.step(a)
    if env.qi==3 and i%10==0: 
        plan=env._chord_plan(env._current_item()) if env._current_chord() else None
        print(env.tick,env.qi,np.round(env.body[:2],2),plan and (plan["mod"],plan["mod_leg"],plan["key_leg"],np.round(plan["goal"],2)),dict(env.sim.keyboard.down),np.round(env.tips[[0,3]],2).tolist())
    if te or tr: break
print([ (e["tick"],e["type"],e["key_id"],e.get("foot")) for e in env.events if e["type"]!="modifier"])
