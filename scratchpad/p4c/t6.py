import numpy as np, sys
from backend.fly_env import *
mode=sys.argv[1]; seed=int(sys.argv[2])
env=FlyTypingEnv(action_mode=mode, terminate_on_error=False, physics={"shift_mode":"held"})
items=[("c",False),("9",True)]
env.reset(seed=seed,options={"keys":items,"max_steps":300})
for i in range(300):
    a=env.expert_action()
    obs,r,te,tr,info=env.step(a)
    if info["error"]: print("ERR at",env.tick)
    if te or tr: break
print([ (e["tick"],e["type"],e["key_id"],e.get("foot"),e.get("modifiers")) for e in env.events])
