# Closeout Review

## Role

Act as a fresh-context final reviewer for the complete initiative branch. Read the authoritative initiative plan, all work orders and reviews, execution log, relevant requirements and blueprints, and the complete branch diff.

## Procedure

1. Confirm every work order is complete or explicitly tabled.
2. Confirm all required implementation reviews, automated checks, manual gates, and approvals have evidence.
3. Compare the final implementation with durable requirements and blueprints; request or make only authorized final documentation fixes.
4. Check for accidental files, unresolved conflicts, untracked changes, and incomplete rollback or migration work.
5. Record deferred work as a clearly described follow-up item with rationale and discoverable ownership.
6. Create the closeout review, append the final execution-log entry, and set the initiative to `closed` only when all mandatory gates pass.

## Closeout outcomes

- **Closed:** all scoped work and required evidence are complete.
- **Blocked:** a required fix, decision, or external gate remains.
- **Closed with follow-up:** scoped work is complete and deferred items are explicitly recorded under project policy.

Do not delete completed initiatives by default. Keep them in place with status metadata so future agents can trace decisions and prior work.
