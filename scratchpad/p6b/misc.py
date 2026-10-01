import json, torch, time
from training import distill, edit_trainer as E, thinker_train_common as C
cfg = C.load_cfg(None, {"model.k_steps": 2, "train.batch": 8, "train.max_len": 96, "train.eval_pairs": 8,
    "train.gen_pairs": 4, "train.gen_max_new": 48, "train.eval_every": 0, "train.ckpt_every": 0, "train.log_every": 100})
# 1. connectome mix_inputs works (B/C path)
m,_ = C.build("real", cfg, 0)
x = torch.randint(20,100,(2,8)); mk = torch.zeros(2,8,dtype=torch.bool); mk[:,4:]=True
t=time.time(); mixed,f = C.mix_inputs(m,x,mk,1.0,torch.Generator().manual_seed(0)); print("connectome mix ok frac",f, round(time.time()-t,2),"s")
# 2. schedule excerpt + controls table (gru, shared cfg)
data = distill.load_pairs("dataset")
res = distill.run_controls(cfg, ["gru"], [0,1], "scratchpad/p6b/ctl", data, 40, log=lambda *_: None)
print(open("scratchpad/p6b/ctl/controls.md").read()); print([r["cfg_hash"] for r in res["rows"]])
rows=[json.loads(l) for l in open("scratchpad/p6b/ctl/gru_s0/schedule.jsonl")]
print("\n".join(json.dumps(r) for r in rows[::4]))
# 3. edit overfit: train and eval on same 48 examples
cfg2 = C.load_cfg(None, {"model.k_steps": 2, "train.batch": 16, "train.max_len": 160, "train.eval_pairs": 48,
    "train.eval_every": 0, "train.ckpt_every": 0, "train.log_every": 100, "edit.train_examples": 48, "edit.val_examples": 48})
tok = distill._tok(cfg2); data = distill.load_pairs("dataset")
exs = E.build_examples(tok, data["train"], 48, 0, 3, cfg2["edit"], 160)
items=[(e["ids"],e["mask"]) for e in exs]
m,_=C.build("gru",cfg2,0); tr=C.Trainer(m,cfg2["train"],{"a_frac":1,"b_frac":0,"b_levels":[.5]},"scratchpad/p6b/edit_of",tag="e")
def gb(step):
    idx=C.sample_indices(len(items),16,0,step); return C.pad_batch([items[i] for i in idx],160)
r=tr.run(gb,E.make_evaluator(tok,cfg2,exs),400,log=lambda *_:None); print("edit overfit",json.dumps(r["final"]), r["seconds"])
