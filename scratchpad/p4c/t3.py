import sys, json
from training.physical_eval import type_tokens
mode=sys.argv[1]
cases={"del":["abc","<LEFT>","<DEL>"],"home":["abc","<HOME>","x"],"end":["abc","<HOME>","<END>","y"],
 "sel":["hello","<SEL_START>","<LEFT>","<LEFT>","<SEL_END>","X"],
 "Shiftchar":["Ab"],
 "selhome":["ab cd","<SEL_START>","<HOME>","<SEL_END>","Z"]}
for name,src in cases.items():
    kw={}
    if name=="Shiftchar": kw={"shift_mode":"held"}
    r=type_tokens(src,"expert",mode,**kw)
    print(name,mode,{k:r[k] for k in ("text","exact","contact_verified","success_rate","ticks","slips","replans","shift_mode","fn_mode","fn_latch_used","unreachable","complete")})
