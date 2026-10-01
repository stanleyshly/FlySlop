import sys, json, time
from pathlib import Path
from training import curriculum as cur
sid=int(sys.argv[1]); rd=Path(sys.argv[2])
cfg=cur.effective_config(json.loads(Path("training/curriculum.json").read_text()), True)
state=cur.new_state(cfg, True, {})
rd.mkdir(parents=True, exist_ok=True)
entry=state["stages"][sid-1]
t=time.time()
cur.run_stage(state, entry, cfg["stages"][sid-1], rd, True, 0.6, 0)
m=entry["metrics"]
print("STATUS", entry["status"], entry.get("reason"), "time", round(time.time()-t,1))
print(json.dumps({k:v for k,v in m.items() if k not in("phases",)}, default=str)[:3000])
print(json.dumps(m.get("phases"), default=str)[:2500])
print("artifacts", entry["artifacts"])
cur.atomic_write_json(rd/"state.json", state)
