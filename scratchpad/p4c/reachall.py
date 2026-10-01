from backend.fly_env import *
from backend.keyboard import CHAR_TO_KEY
env=FlyTypingEnv(action_mode="claw",physics={"shift_mode":"held"})
bad=[]
for ch,(k,sh) in sorted(CHAR_TO_KEY.items()):
    if sh:
        try: env.check_chord((k,True,()))
        except UnreachableChord as e: bad.append(ch)
print("unreachable shifted chars:",bad)
for k in ("Backspace","ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Tab","q","Space","Enter"):
    print("Fn+",k,"reachable" if env.chord_options(k,"Fn") else "UNREACHABLE span=%.1f"%env.chord_span(k,"Fn"))
for k in ("ArrowLeft","ArrowRight","ArrowUp","ArrowDown","Backspace","Enter","Tab","Space"):
    print("Shift+",k,"reachable" if env.chord_options(k,"Shift") else "UNREACHABLE")
