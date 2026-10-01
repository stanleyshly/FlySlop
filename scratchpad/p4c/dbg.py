import numpy as np
from backend.fly_env import *
env=FlyTypingEnv(action_mode="claw",physics={"shift_mode":"held"})
env.reset(seed=0,options={"target":"a"})
best=[]
(x0,x1),(y0,y1)=BOUNDS
for ml,kl in (("LF","RF"),("RF","LF")):
    r=min((env._pair_ratio("a","ShiftLeft",ml,kl,(x,y)),x,y) for x in np.arange(x0,x1,.25) for y in np.arange(y0,y1,.25))
    print(ml,kl,r)
print(press_point("a",0.0),press_point("ShiftLeft",0.0))
