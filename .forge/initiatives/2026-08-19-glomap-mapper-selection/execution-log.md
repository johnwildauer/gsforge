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

### 2026-08-19 — Local COLMAP CLI evidence recorded

- **Actor/role:** human-provided local CLI evidence
- **State transition:** `draft` → `draft`
- **Work order/review:** planning evidence update; WO-001 remains unexecuted
- **Action:** Checked the installed CUDA-enabled COLMAP binary's CLI help. The binary reports COLMAP `4.0.2`, commit `d927f7e`, and a CUDA build. Top-level help lists both `mapper` and `global_mapper`; `mapper -h` exposes `--Mapper.*`; `global_mapper -h` exposes `--GlobalMapper.*`.
- **Evidence:** The user-provided COLMAP `4.0.2` help included these relevant excerpts:

  ```text
  $ colmap -h
  mapper
  global_mapper

  $ colmap mapper -h
  --Mapper.ba_refine_focal_length
  --Mapper.ba_refine_principal_point
  --Mapper.ba_refine_extra_params

  $ colmap global_mapper -h
  --GlobalMapper.ba_refine_focal_length
  --GlobalMapper.ba_refine_principal_point
  --GlobalMapper.ba_refine_extra_params
  ```

  The command surface confirms that global mapping is a distinct subcommand with its own option namespace, rather than a `mapper` option selected by `--Mapper.mapper_type GLOBAL`.

- **Result:** The correct cleanup target is subcommand dispatch to `global_mapper`; enabling a `--Mapper.mapper_type GLOBAL` flag is not the correct approach. No source or test files were modified, and no A/B reconstruction test has run yet.
- **Next action/owner:** human dataset gate owner to run the deferred dataset A/B reconstruction gate.

### 2026-08-19 — WO-001 mapper dispatch cleanup implemented

- **Actor/role:** gsforge implementer
- **State transition:** `draft` → `draft`
- **Work order/review:** WO-001 implementation and automated validation
- **Action:** Updated [`src/gsforge/sfm.py`](../../../src/gsforge/sfm.py:327) so the existing `glomap` method dispatches to COLMAP's `global_mapper` subcommand with the `GlobalMapper` option namespace, while the explicit `colmap` method continues to dispatch to the ordinary incremental `mapper` subcommand with the `Mapper` namespace. The existing full-pipeline default remains `glomap` and the CLI method option was not changed.
- **Tests:** Added focused command-capture coverage in [`tests/test_sfm.py`](../../../tests/test_sfm.py:70) for both subcommands and option namespaces, plus an assertion that the full-pipeline default remains `glomap`.
- **Results:** `pixi run -e dev test tests/test_sfm.py` — 24 passed. `pixi run -e dev test` — 79 passed. `pixi run -e dev ruff check tests/test_sfm.py` — passed. `pixi run -e dev ruff format --check src/gsforge/sfm.py tests/test_sfm.py` — passed. `git diff --check` — passed. The repository-wide lint task remains non-zero on pre-existing diagnostics outside this cleanup and existing diagnostics elsewhere in `sfm.py`; no lint autofixes were applied.
- **A/B gate:** No real-data COLMAP reconstruction or comparative `--method colmap`/`--method glomap` A/B run was performed. Version/help evidence for COLMAP 4.0.2 and the separate subcommands is retained in the prior log entry; registered-camera, exit-status, sparse-model, and unknown-option/subcommand results remain deferred to the human dataset gate.

### 2026-08-19 — GLOMAP/COLMAP A/B validation completed

- **Actor/role:** human-provided local reconstruction logs
- **State transition:** `draft` → `draft`
- **Work order/review:** WO-001 dataset A/B validation
- **Action/evidence:** Both runs used the same 185 prepared frames and the same project-local CUDA-enabled COLMAP `4.0.2` binary. The `glomap` run dispatched to `global_mapper` and completed with 185 cameras, producing sparse model `sfm/sparse/0`. The explicit `colmap` run dispatched to `mapper` and completed with 185 cameras, producing sparse model `sfm/sparse/0`.
- **Caveats:** GLOMAP emitted a non-fatal focal-length-prior warning and a non-fatal CPU Ceres/cuDSS fallback warning. The initial Windows Rich Unicode failure was bypassed only with session UTF-8 environment variables and is out of scope. The `colmap --version` probe is unsupported by this binary; the binary was independently verified as COLMAP `4.0.2`. No training was run.
- **Result:** The bounded A/B execution gate passed for dispatch, completion, registered-camera count, and sparse-model output. These logs do not establish qualitative reconstruction superiority between methods.

## Unresolved and deferred work

- Installed COLMAP version and command/option evidence are recorded as COLMAP `4.0.2`, commit `d927f7e`, CUDA build, with separate `mapper` and `global_mapper` subcommands and namespaces; WO-001 dispatch cleanup is complete.
- The completed A/B validation confirms both bounded reconstruction executions completed with 185 cameras and sparse model `sfm/sparse/0`; no qualitative superiority claim is made.
- No durable requirement or blueprint update is planned; any future clarification requires explicit authorization in a later review.

### 2026-08-19 — Implementation review and initiative closeout

- **Actor/role:** gsforge implementation/closeout reviewer
- **State transition:** `draft` → `closed-with-follow-up`
- **Work order/review:** WO-001 `complete`; implementation review `pass-with-warnings`; initiative closeout `closed with follow-up`
- **Action:** Reviewed the implementation-review and closeout processes, initiative/work-order artifacts, requirements and blueprints, validated source/test state, automated test results, and human-provided COLMAP command/help and A/B evidence.
- **Result:** All bounded acceptance and review gates passed: global dispatch uses `global_mapper`, incremental dispatch uses `mapper`, focused and repository tests passed, scoped lint/format checks passed, and both same-project runs completed with 185 registered cameras and `sfm/sparse/0`. No qualitative reconstruction superiority is claimed.
- **Warnings/follow-ups:** Windows Unicode output workaround, unsupported `colmap --version`, optional focal calibration, and the non-fatal CPU Ceres/cuDSS fallback remain explicitly out of scope. Repository-wide lint findings remain pre-existing/outside this cleanup.
- **Records:** [`WO-001-implementation-review.md`](reviews/WO-001-implementation-review.md) and [`initiative-closeout-review.md`](reviews/initiative-closeout-review.md).
