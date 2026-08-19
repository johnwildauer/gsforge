# Design Review

## Role

Act as a fresh-context reviewer. Read the initiative, all work orders, relevant requirements, blueprints, and [`00-overview.md`](00-overview.md). Do not implement or silently rewrite the design under review.

## Review questions

1. Does the initiative solve a real product or repository need documented in requirements?
2. Is the scope coherent, bounded, and free of unrelated work?
3. Does every major requirement have a plausible blueprint and work-order path?
4. Are work orders independently executable, ordered, and free of hidden dependencies?
5. Are durable documentation updates explicit and appropriately owned?
6. Are automated, manual, external, and human approval gates identified?
7. Are risks, rollback/recovery, security, compatibility, and failure behavior addressed where relevant?
8. Does the proposed parallelism comply with project policy?

## Output

Create a review using [`../initiatives/YYYY-MM-DD-initiative-template/reviews/review-template.md`](../initiatives/YYYY-MM-DD-initiative-template/reviews/review-template.md). Classify findings as blockers, warnings, or notes and include evidence and required action.

The outcome is `pass` only when no blocker remains. Warnings may pass only when the initiative owner and required human approver explicitly accept them. A failed review returns the initiative to planning and must be logged.
