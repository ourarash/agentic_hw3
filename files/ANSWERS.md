# HW 3 Answers

Write two to four clear sentences for each question. Refer to the code and
the output of `python run_demo.py`. You do not need outside sources.

## 1. Why does each chunk keep its heading?

The C1 body says only "Saturday building hours: 10 a.m. to 10 p.m." Explain
what a retriever and a reader would lose if the chunk text omitted the heading
"Griffith Observatory Saturday building hours."

> [Your answer here]

## 2. Keyword, vector, or hybrid?

For "Where can I leave my car?", what did keyword search and vector search
return, and why? Using the recall@k table, explain what hybrid search gained
and what it cost in the number of chunks sent to the model.

> [Your answer here]

## 3. Retrieval recall versus citation support

In the `citation_errors` answer, both required chunks were retrieved, yet the
verifier rejected it. Report its citation recall and precision, and explain why
high retrieval recall does not show that an answer is supported.

> [Your answer here]

## 4. What can an exact-quote check prove?

Run `python run_demo.py --live` once, with OpenAI or LM Studio, and state
the model's claims and whether the verifier accepted them. If you cannot run
a live model, use the scripted `injected` run from `python run_demo.py`
instead. The forum post C7 reached the model. Could a claim that cites C7
and quotes it exactly pass your verifier? What does the quote check establish,
and what does it not?

> [Your answer here]
