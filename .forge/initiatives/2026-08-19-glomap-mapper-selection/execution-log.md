# Execution Log

> Append entries chronologically. Never rewrite prior entries to conceal failures, retries, or superseded decisions.

## Initiative

- **Initiative:** [`initiative-v1.md`](initiative-v1.md)
- **Current status:** `draft`
- **Orchestrator:** `gsforge maintainers`

## Log entries

### 2026-08-19 — Initiative opened and issue evidenced

- **Actor/role:** architect/planner
- **State transition:** `uninitialized initiative` → `draft`
- **Work order/review:** planning / WO-001 drafted
- **Action:** Read the Forge index, governing process, initialization baseline, reconstruction requirement/blueprints, implementation, CLI, and tests. Inspected the current mapper dispatch and official COLMAP CLI guidance through Playwright.
- **Evidence:** [`src/gsforge/sfm.py:350`](../../../src/gsforge/sfm.py:350) computes `GLOBAL`; [`src/gsforge/sfm.py:368`](../../../src/gsforge/sfm.py:368) shows the only `--Mapper.mapper_type GLOBAL` proposal is commented out; [`src/gsforge/sfm.py:374`](../../../src/gsforge/sfm.py:374) still invokes `mapper`; [`src/gsforge/cli.py:334`](../../../src/gsforge/cli.py:334) defaults the CLI to `glomap`; [`REQ-RECONSTRUCTION.md:25`](../../../requirements/features/REQ-RECONSTRUCTION.md:25) records the alignment gap; official [COLMAP CLI documentation](https://colmap.github.io/cli.html) describes `global_mapper` as the global-SfM command.
- **Result:** Exact cleanup target identified as method dispatch/command construction, with a binary-help verification gate added because installed COLMAP versions may differ in command and option support. No source or test files were modified.
- **Next action/owner:** human design approver to review the initiative and approve the target binary/dataset A/B gate; then design review.

## Unresolved and deferred work

- Target installed COLMAP version and its exact `global_mapper` option set are unresolved until WO-001 executes the binary-help checks; owner: implementer/human approver.
- No durable requirement or blueprint update is planned; any future clarification requires explicit authorization in a later review.
