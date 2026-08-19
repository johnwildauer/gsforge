# Execute a Work Order

## Role and scope

Act as the implementer for one assigned work order. Read [`00-overview.md`](00-overview.md), the initiative plan, the work order, linked requirements, blueprints, and prior execution-log entries before changing code. Do not expand scope without updating the work order and obtaining the required approval.

## Procedure

1. Confirm the work order is approved/ready, its dependencies are complete, and the working branch is correct.
2. Inspect the current code and verify the work-order assumptions. Record material discrepancies before implementation.
3. Implement the smallest coherent change within scope.
4. Update requirements or blueprints only when explicitly authorized by the work order. Keep documentation aligned with the resulting behavior.
5. Run the commands and manual checks required by the work order and project policy. Preserve useful evidence, including failures.
6. Update the work order status and append an execution-log entry describing files changed, checks run, results, and follow-up.
7. Hand the work order to [`05-implementation-review.md`](05-implementation-review.md). Do not mark it complete until the configured review and gates pass.

## Blocking conditions

Stop and mark the work order `blocked` when requirements are contradictory, a dependency is missing, a required command cannot run, a manual gate is unavailable, or safe implementation would exceed scope. Record the exact issue and proposed next action.
