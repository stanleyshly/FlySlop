## Connectome Coding Curriculum

Our current codebase uses a fake MLP, we want to replace it all and use the real flyconecdome. We should benchmakr if MPS if makes sense over cpu here. We currently have it copying the physics here, we want the connectdome to use the original leg moving neurons here and have it controling the positon oft he legs instead of trying ot copy the staitc animation. Got it? 

We will need a circirulum to have it train to actually gernate code from a prompt, rather than just copying existing snippets. Here is my rough idea, our goal is to figure out a reaonsale ciricualt solution for this, we should have one script that automsticayll swithces from stage to stage once it is done. 

### 1. Key Control
**Input:** target key  
**Goal:** learn to reliably press the requested physical key.

```text
target_key → connectome/controller → motor policy → physical keypress
```

Start with a few keys, then expand to the full keyboard.

---

### 2. String Typing
**Input:** target text, we can extract segments fo RTl from the lab here as well as pull from online code/RTL datasets. Considering merging 2 and 3 if ti makes sense
**Goal:** learn ordered sequences of keypresses.

```text
"hello"
↓
h → e → l → l → o
```

Progress from short strings to longer sequences.

---

### 3. Code-String Typing
Introduce syntax-heavy strings and programming characters:

```text
()
[]
{}
:
=
+
-
*
quotes
indentation
Enter
Tab
Shift
```

The system is still copying known target text rather than generating code.

---

### 4. Editing
Teach actions beyond forward typing:

```text
Backspace
Delete
arrow keys
cursor movement
selection
insertion
replacement
```

The system should be able to modify an existing program rather than only type from an empty editor.

---

### 5. Error Recovery
Deliberately introduce mistakes:

```text
wrong character
missing character
duplicate character
incorrect indentation
wrong cursor position
```

The controller must observe the editor and repair the program.

This prevents it from depending on perfect trajectories.

---

# 6. LLM Oracle / Teacher

Introduce a frozen coding LLM. Gemma 4 E2B shoudl be good for this

Example:

```text
PROMPT:
"Write a Python function that doubles x."

          ↓

LLM TEACHER

          ↓

def double(x):
    return x * 2
```

Initially, the connectome does **not** generate the solution.

The LLM determines what code should be written, while the already-trained connectome/typing system physically types the teacher's output. We should start mixing in heavy RTL prompts from the lab here by this point. we can geanrets synthiec ones by extaction functions from the code, feeding it into the LLM and creating a prompt. Then we can fed the promtp into and use that to generate spromtps. 

```text
prompt
↓
LLM oracle
↓
target code
↓
connectome typing system
↓
physical keyboard
```

At the same time, save large numbers of:

```text
(prompt, teacher-code)
```

pairs.

These become the dataset for the next stage.

---

# 7. Distillation

This is the main handoff.

Instead of permanently relying on the LLM, train the connectome to imitate the LLM's coding behavior.

The connectome learns:

```text
prompt + code generated so far
              ↓
          connectome
              ↓
        next code token
```

For example:

```text
Prompt:
"Write a function that doubles x."

Teacher output:
def double(x):
    return x * 2
```

Turn this into training examples such as:

```text
prompt + []                    → "def"
prompt + ["def"]               → "double"
prompt + ["def","double"]      → "("
prompt + ["def","double","("]  → "x"
...
```

Use **code tokens**, rather than physical characters, for this level of learning.

The predicted token is then passed into the typing stack:

```text
PROMPT
   ↓
CONNECTOME
   ↓
next code token
   ↓
token → characters
   ↓
trained typing controller
   ↓
motor controller
   ↓
physical keyboard
```

This separates two problems:

```text
Connectome:
"What code should come next?"

Typing system:
"How do I physically type it?"
```

### Distillation Phase A — Teacher Forcing

Give the connectome the real teacher-generated history:

```text
prompt + correct previous tokens → predict next teacher token
```

Train with ordinary next-token loss.

At this point mistakes do not accumulate because the previous context is always correct.

### Distillation Phase B — Mixed Rollouts

Gradually expose the connectome to its own outputs.

For example:

```text
100% teacher history
↓
75% teacher / 25% student
↓
50 / 50
↓
25 / 75
↓
mostly student history
```

Now it learns what to do after making its own imperfect predictions.

### Distillation Phase C — Student-Only Generation

Remove teacher tokens entirely.

```text
prompt
↓
connectome
↓
token
↓
connectome
↓
token
↓
...
↓
complete program
```

The LLM is no longer involved during inference.

It remains useful for producing additional training examples.

---

# 8. Autonomous Connectome Coding

Now test the actual objective:

```text
natural-language prompt
        ↓
    connectome
        ↓
   generated code
        ↓
typing controller
        ↓
 physical keyboard
```

Start with extremely small programming problems:

```text
"return 5"

"double x"

"add x and y"

"return the larger number"

"check whether x is even"
```

Then progressively increase semantic difficulty.

The main measurement here is how much coding behavior the connectome can absorb from the stronger LLM teacher.

---

### 9. Longer Programs

Increase sequence length only after autonomous short programs work:

```text
single expression
↓
one-line function
↓
3–5 line function
↓
multiple branches
↓
loops
↓
10–20 line programs
↓
multiple functions
↓
larger programs
```

This stage tests the connectome's memory and context limits.

---

### 10. Self-Correction

Finally, close the loop.

```text
prompt
↓
connectome generates code
↓
physically types code
↓
run tests
↓
observe editor/test output
↓
connectome decides next edit
↓
physical correction
↓
run again
```

Eventually success can be defined by behavior rather than exact imitation:

```text
program parses
tests execute
tests pass
```

rather than:

```text
student code == teacher code
```

---

## Overall Curriculum

```text
KEY CONTROL
↓
STRING TYPING
↓
CODE-STRING TYPING
↓
EDITING
↓
ERROR RECOVERY
↓
LLM ORACLE
↓
collect prompt → teacher-code dataset
↓
DISTILLATION
    ├─ teacher forcing
    ├─ mixed teacher/student rollouts
    └─ student-only generation
↓
AUTONOMOUS CONNECTOME
prompt → code directly
↓
LONGER PROGRAMS
↓
SELF-CORRECTION
prompt → write → run → observe → edit → run
```

The key conceptual transition is:

```text
EARLY:
LLM thinks
connectome types

          ↓ distillation ↓

FINAL:
connectome thinks
connectome types
```

The LLM therefore acts as a **teacher used to bootstrap coding ability**, rather than a permanent component of the final system.

All fo this rianing should be one command. 