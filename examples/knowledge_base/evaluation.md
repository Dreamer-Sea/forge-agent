# Evaluation

## Concept

Evaluation is the process of checking whether an agent produces correct, grounded, and useful results for a given task. In forge-agent, eval work starts with a small dataset of queries, expected sources, and expected behaviors.

## Components

A minimal evaluation loop includes:

- Eval Dataset: defines user queries and expected outcomes.
- Expected Source: identifies which document or chunk should be retrieved.
- Actual Retrieval Result: records the top-k chunks returned by the retriever.
- Hit or Miss: checks whether the expected source appears in the retrieved results.
- Failure Reason: explains why retrieval or answering failed.
- Regression Test: turns important eval cases into automated tests.

## How Eval Works

Eval work follows a repeatable loop:

1. Prepare representative user queries.
2. Define expected sources or expected answers.
3. Run the retriever or agent runtime.
4. Compare actual results with expected results.
5. Record misses and improve loader, chunker, retriever, context builder, or prompts.
6. Convert important cases into tests.

## Design Notes

Evaluation should not only check final answers. It should also check retrieval quality, citation coverage, trace events, and whether the answer is grounded in retrieved context.
