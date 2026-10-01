import sys
from backend.fly_env import *
from backend.keyboard import CHAR_TO_KEY
mode=sys.argv[1]; seeds=range(int(sys.argv[2]))
chars=[c for c,(k,sh) in sorted(CHAR_TO_KEY.items()) if sh]
env=FlyTypingEnv(action_mode=mode, terminate_on_error=False, physics={"shift_mode":"held"})
bad={}; unreach=[]; tot=0; tk=[]
for c in chars:
    k=CHAR_TO_KEY[c][0]
    try: env.check_chord((k,True,()))
    except UnreachableChord: unreach.append(c); continue
    fails=0
    for s in seeds:
        env.reset(seed=s,options={"keys":[(k,True)],"max_steps":300})
        for i in range(300):
            obs,r,te,tr,info=env.step(env.expert_action())
            if te or tr: break
        ok=info["exact"] and not info["error"]
        fails+=not ok; tot+=1; tk.append(env.tick)
    if fails: bad[c]=fails
print(mode,"unreachable:",unreach,"failing (of %d seeds each):"%len(seeds),bad,"runs",tot,"mean ticks",sum(tk)/len(tk))
