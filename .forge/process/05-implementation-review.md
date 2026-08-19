# Implementation Review

## Role

Act as an independent, fresh-context reviewer after a work order or configured batch is implemented. Read the work order, linked requirements/blueprints, execution log, and relevant diff. Report findings; do not silently modify the implementation.

## Review checklist

- The diff is limited to the authorized scope and does not contain accidental changes.
- The implementation satisfies every applicable acceptance criterion.
- Tests, lint, build, runtime, and manual gates required by policy have evidence.
- Error, security, compatibility, and persistence behavior are appropriate for the change.
- Authorized requirements and blueprint updates accurately describe the final behavior.
- The work order's out-of-scope boundary was respected.

## Outcome

Create a review from [`../initiatives/YYYY-MM-DD-initiative-template/reviews/review-template.md`](../initiatives/YYYY-MM-DD-initiative-template/reviews/review-template.md). A blocker or failed required gate returns the work order to `in-progress` or `blocked`. A pass allows the orchestrator to continue to the next ordered work order. Human approval is not required after every implementation review by default; follow the project policy and record any bypass.
