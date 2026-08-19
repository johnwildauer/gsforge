# Plan and Design an Initiative

## Purpose

This process turns a user request or product goal into one bounded initiative and an ordered set of executable work orders. It is the planning/design phase; implementation begins only after [`03-design-review.md`](03-design-review.md) passes and the required human design approval is recorded.

## Create the initiative

1. Read [`00-overview.md`](00-overview.md), the relevant requirements and blueprints, and the initiative template at [`../initiatives/YYYY-MM-DD-initiative-template/`](../initiatives/YYYY-MM-DD-initiative-template/).
2. Create a dated, short-slug directory under [`../initiatives/`](../initiatives/). Copy the template contents and rename placeholders.
3. Create or select an initiative branch according to the project policy. Record the branch in the initiative metadata.
4. Start the execution log before any agent is dispatched.

## Planning loop

The orchestrator may dispatch one or more architect agents to collaborate with the user. The planner must:

- define the outcome, scope, non-goals, constraints, risks, and success measures;
- link the relevant requirements and blueprints;
- identify required durable-document updates;
- state assumptions and unresolved decisions;
- describe the intended technical approach without prescribing every line of code; and
- update the initiative plan version when material design changes occur.

## Decompose into work orders

Create work orders from the approved design. Each work order must have one clear owner and outcome, explicit in/out-of-scope boundaries, verbatim applicable requirements, blueprint references, implementation guidance, files/systems likely to change, validation steps, and documentation obligations.

Work orders should be independently reviewable and ordered by dependency. Sequential execution is the default. Parallel work requires non-overlapping ownership or a documented merge strategy in the initiative plan.

## Design readiness

Before requesting review, confirm that the plan has no unresolved blocker, every requirement has a technical path, every work order has validation, and project-specific gates are identified. Set the initiative to `in-review` and dispatch [`03-design-review.md`](03-design-review.md).

## Rework and approval

If design review fails, record the findings, return the initiative to `draft` or `in-progress`, and revise the plan/work orders. Do not implement around a failed design gate. When the review passes, obtain the required human approval and set the initiative to `approved` before dispatching implementation.
