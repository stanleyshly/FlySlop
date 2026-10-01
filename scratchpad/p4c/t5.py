from training.physical_eval import type_tokens
for src in (["Q!@#(){}:\"<>?|~_+"],["abcdefgh","<SEL_START>","<LEFT>","<LEFT>","<LEFT>","<SEL_END>","Q"]):
    r=type_tokens(src,"expert","mn",recover=False,shift_mode="held")
    print(r["unreachable"], r["unsupported"])
