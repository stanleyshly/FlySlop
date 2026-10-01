import sys, numpy as np, time
from backend.fly_env import *
mode=sys.argv[1]; items=eval(sys.argv[2]); phys=eval(sys.argv[3]) if len(sys.argv)>3 else {}
env=FlyTypingEnv(action_mode=mode, terminate_on_error=False, physics=phys)
try:
    obs,_=env.reset(seed=0,options={"keys":items,"max_steps":900})
except UnreachableChord as e:
    print("UNREACHABLE",e); sys.exit()
for i in range(900):
    obs,r,te,tr,info=env.step(env.expert_action())
    if te or tr: break
print(mode,"ticks",env.tick,"exact",info["exact"],"qi",env.qi,"text",repr(env.editor.text))
print([ (e["tick"],e["type"],e["key_id"],e.get("modifiers")) for e in env.events if e["type"]!="contact_offset"][:14])
