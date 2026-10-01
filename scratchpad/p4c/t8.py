from training.physical_eval import type_tokens
r=type_tokens(["ab","<ENTER>","cd","<SEL_START>","<UP>","<LEFT>","<SEL_END>","<BS>"],"expert","mn",seed=4,recover=False)
print({k:r[k] for k in ("text","ticks","slips","unreachable","complete","success_rate")})
print([(e["tick"],e["type"],e["key_id"],e.get("foot"),e.get("modifiers")) for e in r["events"] if e["type"] in("key","contact_onset")])
