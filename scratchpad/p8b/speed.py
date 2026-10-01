import time,resource
from training.common import make_env, load_config
c=load_config("training/ppo_fly_mn.json")
t=time.time()
from training.stages import physical_common as p
c=p.build_config(p.phys_cfg({},False))
import numpy as np
env=make_env(c,list("asdf"))
print("build",time.time()-t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20)
obs,_=env.reset(seed=0,options={"target":"a"})
t=time.time();n=0;d=False
while not d:
    obs,r,te,tr,i=env.step(env.expert_action());n+=1;d=te or tr
print(n,"steps",time.time()-t, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20,"GB?")
