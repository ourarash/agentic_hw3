# HW 3 — LA Visit RAG: Answers You Can Check

## EE 599: Designing and Building Autonomous AI Agents

**Expected time: about one hour. Tests and required demonstrations run
offline. A live model call is optional.**

## What you will build

Your friends are planning a first Saturday visit to Los Angeles. They ask:

> Is the Griffith Observatory building open at 7 p.m. on Saturday, and where
> can we park?

You will build the retrieval-augmented generation (RAG) workflow from Session
07 over a small set of saved LA visitor notes:

1. Split the notes into chunks that keep their heading and source location.
2. Rank chunks by cosine similarity, then combine keyword and vector results.
3. Give the model the question and the retrieved evidence cards.
4. Check every claim the model makes against the text it was given.
5. Measure retrieval with recall@k over a fixed set of questions.

The collection also contains an altered snapshot with the wrong closing time
and a forum post that tries to redirect the model. Your code has to handle
both.

The embeddings come from a small, hand-written function in `support/`, so
retrieval runs offline with no model download. A real embedding model would
learn its coordinates; the arithmetic you write is the same.

## Setup

Use Python 3.11 or newer.

```bash
cd files
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
```

The initial test failures are expected. Each failure names the TODO it
checks. The tests in `support/` cover supplied code and pass from the start.

## Start here

You do not need to read the whole folder before you start.

| File | What to do |
|---|---|
| `rag_pipeline.py` | **Edit.** TODO 1–6. Each TODO shows its input, expected output, and the helper to call. |
| `ANSWERS.md` | **Edit.** Four short explanations. |
| `run_demo.py` | **Run.** Prints retrieval results, a recall@k table, and the verifier's reports. |
| `test_hw3.py` | **Run** with `pytest -q`. You do not need to read it. |
| `support/` | **Ignore.** Documents, keyword search, the toy embedding, and the model call. |

Work through the TODOs in order. Run `pytest -q` after each one; the next
group of tests should pass.

### Part 1: Prepare and retrieve

Complete TODO 1 through TODO 3. Keep only the verified snapshot, keep each
chunk's heading in its text, and break ties by collection order.

```bash
pytest -q
```

### Part 2: Answer and check

Complete TODO 4 and TODO 5. The model returns claims, each with chunk IDs
and a short quote. Your verifier accepts the answer only when every quote
appears in a cited chunk that actually reached the model.

### Part 3: Evaluate retrieval

Complete TODO 6, then run:

```bash
python run_demo.py
```

The offline demo prints what each search returns, a recall@k table, and the
verifier's report for three fixed answers: a supported one, the Session 07
draft with two citation errors, and one that followed the forum post.

## Live model run (optional)

Question 4 in `ANSWERS.md` invites you to run the workflow with a real model.
If you cannot make a live model call, you may answer Question 4 using the
scripted `injected` run from `run_demo.py` instead. Set variables in the same
terminal where you run the command. Never put an API key in source code or
commit it to Git.

### OpenAI

Create an [API key](https://platform.openai.com/api-keys) in your OpenAI
account, then run:

```bash
unset LM_STUDIO_BASE_URL
export OPENAI_API_KEY="your-key"
python run_demo.py --live
```

The default model is `gpt-6-luna`. You may set `OPENAI_MODEL` to another model
available to your account.

### LM Studio

In [LM Studio](https://lmstudio.ai/docs/developer/core/server), load a chat
model and start the local server from the Developer tab. Copy the model
identifier shown there, then run:

```bash
export LM_STUDIO_BASE_URL="http://localhost:1234/v1"
export OPENAI_MODEL="your-loaded-model-id"
python run_demo.py --live
```

LM Studio runs locally and does not need an OpenAI API key. Local model output
varies, so the verifier may reject its answer; that is a valid result to
report.

On Windows PowerShell, set a variable with `$env:NAME="value"` instead of
`export NAME="value"`.

You may also try your own question, for example
`python run_demo.py --live --question "Where can I leave my car?"`.

## Submit

Complete `ANSWERS.md`, run `pytest -q`, and submit your `files/` folder.
