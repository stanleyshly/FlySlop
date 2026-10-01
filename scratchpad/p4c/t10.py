import sys, numpy as np
from backend.fly_env import *
mode=sys.argv[1]
env=FlyTypingEnv(action_mode=mode, terminate_on_error=False, physics={"shift_mode":"held"})
for s in range(10):
    env.reset(seed=s,options={"keys":[("\\",True)],"max_steps":300})
    for i in range(300):
        obs,r,te,tr,info=env.step(env.expert_action())
        if te or tr: break
    print(s, env.tick, info["exact"], info["error"], np.round(env.body[:2],2), env.sim.keyboard.unattributed, [(e["type"][:4],e["key_id"],e.get("foot")) for e in env.events if e["type"]!="contact_offset"][-3:])
