import numpy as np
from backend.fly_env import *
env=FlyTypingEnv(action_mode="claw", terminate_on_error=False, physics={"fn_mode":"latch"})
env.reset(seed=0,options={"keys":[("Backspace",False,("Fn",))],"max_steps":120})
for i in range(45):
    obs,r,te,tr,info=env.step(env.expert_action())
    if i>=25:
        print(env.tick, np.round(env.body[:2],2), env.next_key(), env.typing_leg(), np.round(env.tips[[LEGS.index("LF"),LEGS.index("RF")]],2).tolist(), [ (e["type"],e["key_id"],e.get("foot")) for e in info["events"]])
print("---")
env.reset(seed=0,options={"keys":[("Backspace",False,("Fn",))],"max_steps":120})
i=LEGS.index("RF")
for t in range(40):
    a=env.expert_action()
    obs,r,te,tr,info=env.step(a)
    if t>=30: print(env.tick, np.round(a[[0,1,2,3,4]],2), np.round(env.rel["RF"],2), np.round(env.tips[i],2), np.round(env.anchor_world("RF"),2), np.round(env.velocity,2))
