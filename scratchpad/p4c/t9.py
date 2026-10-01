import numpy as np, sys
from backend.fly_env import *
env=FlyTypingEnv(action_mode="mn", terminate_on_error=False, physics={"shift_mode":"held"})
items=[("d",False),("ArrowUp",True),("ArrowLeft",True)]
env.reset(seed=4,options={"keys":items,"max_steps":300,"start":(0.0,-2.0)})
for i in range(300):
    a=env.expert_action()
    obs,r,te,tr,info=env.step(a)
    if i>10 and (i%8==0 or info["events"]):
        c=env._current_chord(); pl=env._chord_plan(c) if c else None
        print(env.tick,env.qi,np.round(env.body[:2],2),pl and (pl["mod"],pl["mod_leg"],pl["key_leg"],np.round(pl["goal"],2)),dict(env.sim.keyboard.down),"LF",np.round(env.tips[0],2).tolist(),"RF",np.round(env.tips[3],2).tolist(),[(e["type"],e["key_id"]) for e in info["events"]])
    if te or tr: break
print('---')
env.reset(seed=4,options={"keys":items,"max_steps":300,"start":(0.0,-2.0)})
for i in range(90):
    a=env.expert_action()
    obs,r,te,tr,info=env.step(a)
    if i>=60 and i%6==0:
        tip=env.tips[3]; lx,ly=to_layout(float(tip[0]),float(tip[1]))
        key=BY_ID["ArrowUp"]
        print(env.tick,"a RF",np.round(a[5:8] if False else a[2:9],2)[:8], "cmdtip",np.round(env._cmd_tips[3],2), "tip",np.round(tip,2),"layout",round(lx,2),round(ly,2),"key y",key.y,key.y+key.height,"x",key.x,key.x+key.width, "depr",np.round(env.sim.depression()[env.sim.key_ids.index("ArrowUp")],3))
print('===')
env.reset(seed=4,options={"keys":items,"max_steps":300,"start":(0.0,-2.0)})
idx=env.sim.key_ids.index("ArrowUp")
prevd=0
for i in range(70):
    a=env.expert_action()
    obs,r,te,tr,info=env.step(a)
    d=env.sim.depression()[idx]
    if i>=38 and i<=62: print(env.tick,"depr",round(float(d),3),"armed",bool(env.sim.keyboard.armed[idx]),"unattr",env.sim.keyboard.unattributed,"touch",env.sim.touching().get(idx),"RFz",round(float(env.tips[3][2]),3),[(e["type"],e["key_id"],e.get("foot")) for e in info["events"]])
print('#####')
import mujoco
env.reset(seed=4,options={"keys":items,"max_steps":300,"start":(0.0,-2.0)})
m,d=env.sim.model,env.sim.data
for i in range(50):
    a=env.expert_action()
    for sub in range(1):
        obs,r,te,tr,info=env.step(a)
    if env.tick in (47,48):
        for c in d.contact[:d.ncon]:
            n1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom1); n2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,c.geom2)
            b1=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom1]); b2=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,m.geom_bodyid[c.geom2])
            if "keycap" in str(n1)+str(n2): print(env.tick,n1,b1,n2,b2, round(c.dist,4))
print('&&&&&')
env.reset(seed=4,options={"keys":items,"max_steps":300,"start":(0.0,-2.0)})
kb=env.sim.keyboard
orig=kb.update
def upd(tick,depression,touching):
    u=kb.unattributed
    ev=orig(tick,depression,touching)
    if kb.unattributed>u or tick in(47,48) and (depression>=0.03).any():
        print("update tick",tick,"dep>=.03:",[(env.sim.key_ids[j],round(float(depression[j]),3)) for j in np.flatnonzero(depression>=0.03)],"touching",{env.sim.key_ids[k]:v for k,v in touching.items()},"down",dict(kb.down),"unattr",kb.unattributed)
    return ev
kb.update=upd
for i in range(50):
    obs,r,te,tr,info=env.step(env.expert_action())
