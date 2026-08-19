# Execution Log — Harden COLMAP Runtime Compatibility

## 2026-08-19 — planning initialized

- Created initiative [`initiative-v1.md`](initiative-v1.md) from the repository template and set the current gate to `design-review`.
- Confirmed the predecessor's four follow-ups from [`initiative-closeout-review.md`](../2026-08-19-glomap-mapper-selection/reviews/initiative-closeout-review.md:24).
- Removed Windows Unicode console handling from this initiative after the user clarified that the symptom was caused by separate agent/user terminals.
- Chose three sequential work orders: capability/version probing, optional focal calibration, and GLOMAP CPU fallback.
- Chosen branch: `initiative/2026-08-19-colmap-runtime-robustness`; branch creation evidence remains a pre-implementation gate.
- Recorded the executable-terminal/PowerShell limitation and reviewer artifact-location policy in [`process/00-overview.md`](../../process/00-overview.md:67).
- No production source or tests were changed.

## 2026-08-19 — human scope decision

- **Actor/role:** human stakeholder
- **Decision:** Defer focal-length calibration entirely to a later initiative. WO-002 is retained as a deferred planning record and is removed from the active execution sequence.
- **Capture requirement:** The README must state that source footage used by the reconstruction workflow must maintain consistent focal length. This documents an input prerequisite and does not add automatic calibration behavior.
- **Remaining design gate:** Resolve the target GLOMAP CPU/Ceres/cuDSS fallback contract and the normalized COLMAP capability-probe integration contract, then request a fresh independent design review. No production implementation is authorized yet.

## 2026-08-19 — WO-001 evidence and contract refinement

- **Static evidence:** [`check_colmap_version()`](../../../src/gsforge/sfm.py:154) currently invokes only `--version`, returns a plain string, and collapses all exceptions to `unknown`. [`run_sfm()`](../../../src/gsforge/sfm.py:688) calls it before the failure-state `try` block. Existing [`test_sfm.py`](../../../tests/test_sfm.py:1) has no capability-probe coverage.
- **Recorded runtime evidence:** The predecessor GLOMAP log shows the target Windows binary rejecting `--version`, recommending `colmap help`, and subsequently completing GLOMAP and COLMAP reconstruction for the same 185-frame dataset.
- **WO-001 contract:** The work order now defines structured availability/version/capability states, bounded raw evidence, diagnostics, and explicit fatal/non-fatal integration rules for `run_sfm()`.
- **Remaining external evidence:** Before implementation review, capture the target binary's exit code and stdout/stderr for `--version`, `help`, `-h`, `feature_extractor -h`, `mapper -h`, and `global_mapper -h`. No new binary is required for WO-001 if the existing target binary remains the support target.

## 2026-08-19 — WO-003 GPU requirement clarified

- **Actor/role:** human stakeholder
- **Target:** Existing project-local binary [`colmap.exe`](../../../bin/colmap-x64-windows-cuda/bin/colmap.exe) and the runtime captured in [`garden-glomap-sfm.txt`](../../../garden-glomap-sfm.txt).
- **Decision:** CPU fallback is not an acceptable successful GLOMAP path because it defeats the purpose of using GLOMAP. WO-003 is therefore reframed from fallback enablement to GPU solver compatibility/troubleshooting.
- **Evidence:** The captured run confirms GPU SIFT feature extraction and GPU feature matching, but GLOMAP global positioning and iterative bundle adjustment report that Ceres was compiled without CUDA and without cuDSS, then fall back to CPU dense/sparse solvers. The run converged and produced a reconstruction, but it does not satisfy the desired GPU-backed GLOMAP contract.
- **Required user evidence:** Provide the target binary's `help`, `global_mapper -h`, and build/version output if available, plus the GPU/CUDA/driver identity. No official binary version is assumed yet; the existing project-local binary is the initial support target.

## 2026-08-19 — WO-001/WO-003 target evidence received

- **Actor/role:** human stakeholder
- **WO-001 evidence:** The target reports `COLMAP 4.0.2`, commit `d927f7e`, dated `2026-03-18`, with CUDA. Top-level `help` lists the relevant commands, including `global_mapper`; `global_mapper -h` exposes `GlobalMapper.gp_use_gpu=1` and `GlobalMapper.ba_ceres_use_gpu=1`.
- **WO-003 environment:** NVIDIA RTX 4090, NVIDIA Studio Driver 610.88, CUDA Toolkit/runtime 12.4. The binary is the release installed under the project-local `bin/` path documented by the README.
- **Interpretation:** The command contract requests GPU solver use, but the runtime log shows the bundled Ceres lacks CUDA and cuDSS support. GPU feature extraction/matching therefore succeeds while GLOMAP solver stages run on CPU. This is currently a COLMAP release/build compatibility issue, not an apparent gsforge mapper-option omission.
- **Next evidence needed for WO-003:** establish whether the official Windows COLMAP 4.0.2 package is intended to ship CUDA/cuDSS-enabled Ceres, and if so, inspect the package/build dependencies or reproduce with a clean install. If not, identify an approved GPU-capable COLMAP build or formally document that the requested GPU-backed GLOMAP contract cannot be met by this release. No automatic CPU fallback will be implemented.

## 2026-08-19 — official changelog review

- **Source reviewed:** [COLMAP changelog](https://colmap.github.io/changelog.html), inspected through the official documentation site.
- **Relevant findings:** COLMAP 4.0.0 documents automatic CPU BA fallback when the GPU solver fails; 4.0.3 fixes bundle-adjustment option piping into the global mapper; 4.1.0 adds the Caspar GPU-accelerated bundle-adjustment backend as an alternative to Ceres with GPU device selection; 4.1.1 is the latest stable release listed and includes further Windows/CUDA fixes.
- **Decision:** The next binary experiment should use an official COLMAP 4.1.1 Windows CUDA release rather than continuing to treat 4.0.2 as the only candidate. This is a binary validation step, not yet a gsforge dependency decision.
- **Acceptance evidence:** Run the same 185-frame project with the 4.1.1 binary, capture `global_mapper -h`, record the available BA backend/device options, and verify from runtime logs whether global positioning and iterative BA use a GPU backend. A successful reconstruction with CPU fallback warnings does not satisfy the desired GPU-backed GLOMAP outcome.

## 2026-08-19 — issue and FAQ resolve GPU-BA question

- **Sources reviewed:** [COLMAP issue #3474](https://github.com/colmap/colmap/issues/3474) and the official bundle-adjustment FAQ supplied by the stakeholder.
- **Conclusion:** The repeated 4.1.1 warning is expected for official Windows CUDA binaries: CUDA SIFT/matching support does not imply that bundled Ceres was compiled with CUDA/cuDSS. The FAQ states that GPU Ceres/cuDSS requires compiling Ceres with those capabilities and linking that build to COLMAP.
- **Flag assessment:** gsforge already requests `GlobalMapper.ba_ceres_use_gpu=1`. The FAQ's `Mapper.ba_use_gpu` and standalone `BundleAdjustmentCeres.use_gpu` options do not apply to `global_mapper`; Caspar is not exposed as a global-mapper backend. No additional flag can solve the current official-binary limitation.
- **Performance evidence:** The 185-frame run spent `204.584 s` in GLOMAP; global positioning took `40.704 s`, iterative BA `86.512 s`, and retriangulation/refinement `74.701 s`. BA was about 42% of mapper time and about 29% of the full SfM wall time using the recorded extraction/matching phases. It is a significant, not exclusive, optimization target.
- **Planning consequence:** WO-003 must decide between an explicitly documented CPU-BA GLOMAP mode with truthful performance diagnostics or a hard GPU-BA requirement that requires a custom COLMAP/Ceres build or future global-mapper GPU backend. A gsforge option-only change is not a valid solution.

## 2026-08-19 — option 1 accepted

- **Actor/role:** human stakeholder
- **Decision:** Adopt option 1 for standard gsforge support. Official COLMAP binaries may complete GLOMAP with CPU bundle adjustment when CUDA/cuDSS Ceres support is absent, provided the solver mode is reported truthfully. GLOMAP remains the selected global mapper and is not treated as a mapper fallback.
- **Documentation:** [`README.md`](../../../README.md) now explains the Ceres 2.3 status, the official-binary CPU BA behavior, and the optional custom COLMAP/Ceres/Caspar path as untested and unsupported.
- **Custom-build boundary:** Users may compile COLMAP from source and experiment with Caspar, but `global_mapper` backend support must be verified independently; no gsforge compatibility claim is made for that path.

## 2026-08-19 — WO-003 scope simplified by stakeholder

- **Decision:** Keep WO-003 at a high level. Official COLMAP's own CPU bundle-adjustment fallback is sufficient for the standard easy-to-use CLI workflow; gsforge does not need to build a deeply structured solver telemetry system.
- **Accepted evidence:** A normal Ceres solver report and/or the known COLMAP CPU-fallback warning is sufficient to establish that GLOMAP completed. gsforge must identify GLOMAP and avoid claiming GPU BA, but does not need to guarantee exhaustive solver classification across custom builds.
- **Unsupported path:** Users may compile Ceres/Caspar/COLMAP themselves if they require GPU BA. That path remains explicitly untested and unsupported and may require matching gsforge command configuration to the custom binary.

## 2026-08-19 — WO-001/WO-003 implementation validation

- **Implementation:** Added structured COLMAP capability probing with bounded raw evidence, supported-help fallback, per-command capability states, and failed-stage handling for unusable binaries. The probe is integrated into [`run_sfm()`](../../../src/gsforge/sfm.py:749) without changing mapper selection.
- **Implementation:** GLOMAP now explicitly requests its GPU settings and captures mapper output to report the official Ceres CUDA/cuDSS CPU fallback truthfully. It preserves GLOMAP success and does not switch to incremental COLMAP.
- **Tests:** Focused SfM tests passed: 26. Full test suite passed: 81.
- **Scoped quality:** Ruff check passed for [`sfm.py`](../../../src/gsforge/sfm.py) and [`test_sfm.py`](../../../tests/test_sfm.py); scoped Ruff formatting passed. Repository-wide Ruff still reports pre-existing issues outside this change set.
- **External probe:** The installed binary reports COLMAP 4.1.1, commit `a0d785f` dated `2026-07-17`, with CUDA. `help` exposes a `version` command and `global_mapper -h` exposes `GlobalMapper.gp_use_gpu=1` and `GlobalMapper.ba_ceres_use_gpu=1`; no Caspar selector is exposed by `global_mapper`.
- **Gate status:** Production implementation evidence is available, but independent WO-001/WO-003 implementation reviews and the initiative implementation-review gate remain required before completion/closeout.

## 2026-08-19 — independent design review blocked

- **Actor/role:** independent Forge design reviewer
- **State transition:** `in-review` → `blocked`
- **Work order/review:** initiative design review
- **Action:** Reviewed the governing process, initiative plan, execution log, all three work orders, predecessor initiative records, requirements, blueprints, current SfM/CLI/project/logging source, and current tests. No production code or tests were changed.
- **Findings:** Human decisions for the focal-calibration default/control and acceptable CPU fallback semantics remain unresolved; WO-003 does not yet define a selected fallback contract; and WO-001 does not specify the normalized probe-to-`run_sfm()` integration contract. The review also recorded a non-blocking unnecessary WO-003 dependency on WO-002 and incomplete persistence details for calibration policy.
- **Unicode-console assessment:** Excluding Windows Unicode console portability is justified narrowly because the predecessor evidence attributes the observed failure to separate terminals and records a successful UTF-8-session workaround; that evidence does not establish a general portability conclusion. The concern remains a discoverable follow-up, not an initiative capability.
- **Result/next action:** Design review outcome is `blocked`. Return the initiative to planning, resolve the blockers, append the human decisions and binary evidence, obtain a fresh independent design review, and only then seek the required human design approval. See [`initiative-design-review.md`](reviews/initiative-design-review.md).

## 2026-08-19 — fresh independent design review blocked

- **Actor/role:** fresh independent Forge design reviewer
- **State transition:** `planning-revision` → `blocked`
- **Work order/review:** initiative design review
- **Action:** Re-read the governing Forge process, revised initiative plan, execution log, active WO-001 and WO-003, deferred WO-002, prior review, requirements, blueprints, README, predecessor records, and relevant SfM source/tests. No production code or tests were changed.
- **Confirmed decisions:** The human focal-calibration deferral is correctly represented: WO-002 remains deferred, it is absent from the active sequence, and no calibration implementation or persistence contract is authorized. The README capture prerequisite accurately states that source footage should maintain consistent focal length and is appropriately scoped as an input prerequisite, not calibration behavior.
- **Findings:** WO-001 still lacks a testable normalized probe result, authoritative `mapper`/`global_mapper` capability contract, raw-evidence retention rule, and explicit `run_sfm()` continuation/failure mapping. WO-003 still lacks the target binary's supported fallback control, selected human-approved CPU/Ceres/cuDSS behavior, exact solver diagnostic/state contract, and explicit no-safe-fallback block condition. Prior mapper A/B evidence does not resolve either blocker.
- **Result/next action:** Fresh review outcome is `blocked`. Return the initiative to planning, resolve both active work-order contracts and their evidence, obtain another independent design review, and only then seek human design approval. The reviewer did not self-approve unresolved human or binary-contract decisions. See [`initiative-design-review.md`](reviews/initiative-design-review.md).

## 2026-08-19 — independent implementation review blocked

- **Actor/role:** independent Forge implementation reviewer
- **Scope:** Reviewed active WO-001 and WO-003, current [`sfm.py`](../../src/gsforge/sfm.py), [`test_sfm.py`](../../tests/test_sfm.py), README, requirements, blueprints, process policy, validation claims, and existing review artifacts. No production code or tests were changed; no commands or external binaries were run.
- **WO-001 result:** `blocked`. Findings include a contradiction between the required unsupported-version state and the implementation/test expectation, acceptance of empty help as usable, discarded successful probe evidence/diagnostics, and missing `run_sfm()` continuation/failure-state coverage.
- **WO-003 result:** `blocked`. Findings include an untested and non-structured CPU solver diagnostic, incomplete/unreproducible external evidence, missing mapper-failure persistence tests, and no proof of the required manual gate.
- **Shared process result:** Production implementation was recorded while the initiative remained in `planning-revision` after a blocked design review; no human design approval or explicit waiver was found. This remains a completion blocker.
- **Review artifacts:** [`WO-001-implementation-review.md`](reviews/WO-001-implementation-review.md) and [`WO-003-implementation-review.md`](reviews/WO-003-implementation-review.md).
- **Next action:** Return both work orders and the initiative to planning/in-progress, resolve every blocker and warning in the review artifacts, attach reproducible validation evidence, then obtain the required design authorization and fresh implementation reviews. No self-approval was issued.

## 2026-08-19 — remediation pass

- **WO-003 scope decision applied:** Simplified the contract to high-level truthful CPU-fallback reporting. Standard official-binary GLOMAP completion remains valid; custom Ceres/Caspar builds remain user-managed, untested, and unsupported.
- **WO-001 remediation:** A rejected `--version` now reports `version_status=unsupported` even when version text is recovered from help; the documented `version` command is probed; empty or semantically unusable help is rejected; successful probe evidence is summarized at the [`run_sfm()`](../../../src/gsforge/sfm.py:807) boundary.
- **Regression coverage:** Added tests for rejected-version semantics, empty help, CPU fallback warning classification, and command capability evidence.
- **Validation:** Full test suite passed: 83. Scoped Ruff check and formatting passed for [`sfm.py`](../../../src/gsforge/sfm.py) and [`test_sfm.py`](../../../tests/test_sfm.py).
- **Remaining gate:** The previous blocked reviews are not self-marked passed. Fresh independent reviews and the required design approval or explicit waiver remain outstanding.

## 2026-08-19 — fresh evidence and integration coverage

- **WO-001:** Added top-level `-h` probing, preserved unsupported subcommand states, added semantic command-help validation, and added integration tests for rejected-version continuation and mapper-failure persistence.
- **WO-003:** Kept the stakeholder-approved high-level fallback contract and added representative CPU-warning coverage without introducing structured solver telemetry.
- **External evidence artifact:** [`WO-003-external-evidence-4.1.1.md`](reviews/WO-003-external-evidence-4.1.1.md) records the official binary, environment, help commands, exit statuses, options, and runtime-log linkage.
- **Validation:** Full test suite passed: 85. Scoped Ruff check and formatting passed for [`sfm.py`](../../../src/gsforge/sfm.py) and [`test_sfm.py`](../../../tests/test_sfm.py).
- **Remaining process gate:** Fresh implementation reviews must be rerun after this pass; dedicated-branch evidence and human design approval or explicit waiver remain required and are not self-approved.

## 2026-08-19 — WO-003 real-data validation after approval

- **Command:** `pixi run -e dev run sfm --project garden-glomap.gsproject --method glomap`
- **Environment workaround:** `PYTHONIOENCODING=utf-8` was required for the Windows Rich console; without it, the pre-existing Unicode console issue stopped the CLI before SfM started.
- **COLMAP:** 4.1.1, commit `a0d785f`, with CUDA; capability probe reported `help`, `-h`, `version`, both matchers, `mapper`, and `global_mapper` supported.
- **Result:** GLOMAP completed successfully, registered 185 cameras, selected `sfm/sparse/0`, and persisted `sfm_status=completed` with `sfm_method=glomap`.
- **Solver evidence:** COLMAP reported Ceres compiled without CUDA/cuDSS and fell back to CPU dense/sparse solvers. gsforge emitted the high-level CPU bundle-adjustment warning while preserving GLOMAP identity and completion.
- **Disposition:** WO-003 standard official-binary acceptance behavior is demonstrated on the real 185-frame project. The Unicode console issue remains the separately documented portability follow-up.

## 2026-08-19 — human closeout authorization

- **Actor/role:** human stakeholder
- **Decision:** Authorized closeout with warnings/follow-ups, commit of the initiative branch, and merge back to `master`. The stakeholder will perform independent working-directory cleanup afterward.
- **Closeout disposition:** `closed-with-follow-up`.
- **Retained follow-ups:** Windows Unicode console portability, unsupported/alternate COLMAP version probing compatibility, optional focal-length calibration, official-binary CPU Ceres/cuDSS fallback, and unsupported custom Ceres/Caspar build experimentation.

## 2026-08-19 — branch and exact 4.1.1 help evidence

- **Branch:** Switched to `initiative/2026-08-19-colmap-runtime-robustness` before continuing remediation; uncommitted initiative changes were preserved.
- **Technical test evidence:** Added an exact COLMAP 4.1.1 `global_mapper -h` fixture to [`test_sfm.py`](../../../tests/test_sfm.py), asserting the reported `GlobalMapper.gp_use_gpu=1`, `GlobalMapper.ba_ceres_use_gpu=1`, GPU indices, and absence of a Caspar selector.
- **User-provided transcript:** The supplied 4.1.1 output reports commit `a0d785f` from 2026-07-17 with CUDA and confirms the same options; the transcript is linked from [`WO-003-external-evidence-4.1.1.md`](reviews/WO-003-external-evidence-4.1.1.md).

## 2026-08-19 — human design approval recorded

- **Actor/role:** human stakeholder
- **Approval:** Approved the revised initiative design and authorized sequential execution of WO-001 followed by WO-003, with independent implementation reviews for both work orders.
- **Approved boundaries:** Official COLMAP binaries may complete GLOMAP with high-level CPU bundle-adjustment fallback reporting; custom Ceres/Caspar builds remain untested and unsupported; focal-length calibration remains deferred.
- **Branch evidence:** Initiative work is on `initiative/2026-08-19-colmap-runtime-robustness`, created before the continuing implementation/evidence pass.
- **Gate transition:** `design-review` passed with warnings; initiative transitioned to `approved` / `work-order-execution`.

## 2026-08-19 — exact transcript fixture validation

- Added the supplied COLMAP 4.1.1 `global_mapper -h` GPU-option excerpt as a focused test fixture in [`test_sfm.py`](../../../tests/test_sfm.py), including the absence of a Caspar selector.
- Full test suite passed: 86.
- Scoped Ruff check remains passing for [`sfm.py`](../../../src/gsforge/sfm.py) and [`test_sfm.py`](../../../tests/test_sfm.py).
- The prior independent review identified contract/evidence gaps rather than a failure of the 4.1.1 CLI itself. Those gaps are now represented by explicit test/evidence additions; a fresh review is still required before completion.

## 2026-08-19 — fresh independent implementation reviews after remediation

- **Actor/role:** independent Forge architect/reviewer
- **Scope:** Re-read active WO-001 and WO-003, current [`sfm.py`](../../src/gsforge/sfm.py), [`test_sfm.py`](../../tests/test_sfm.py), README, requirements, blueprints, latest execution log, prior blocked reviews, and available runtime evidence. No production code or tests were changed; automated tests and external binaries were not rerun, per the agreed documentary-evidence review policy.
- **WO-001 result:** `blocked`. Rejected-version semantics, semantic help rejection, and probing of the documented `version` command are now present. Remaining blockers are incomplete normalized per-command state mapping, non-retention of bounded raw probe evidence at the [`run_sfm()`](../../src/gsforge/sfm.py:772) boundary, and absent continuation/failure-state integration tests. See [`WO-001-fresh-implementation-review.md`](reviews/WO-001-fresh-implementation-review.md).
- **WO-003 result:** `blocked`. The stakeholder-approved high-level CPU fallback contract is implemented without overengineering; GLOMAP dispatch, GPU-request flags, and no-auto-switch failure boundary remain intact. Remaining blockers are insufficient warning/no-warning classification coverage, absent mapper-failure persistence tests, and missing reproducibly retained 4.1.1 external probe/run evidence. See [`WO-003-fresh-implementation-review.md`](reviews/WO-003-fresh-implementation-review.md).
- **Process-gate result:** `blocked`. The initiative remains `planning-revision`; no human design approval or explicit waiver is recorded, and dedicated-branch creation evidence remains absent. The fresh reviews do not self-approve these gates.

## 2026-08-19 — final fresh independent implementation review after latest remediation

- **Actor/role:** independent Forge architect/reviewer.
- **Scope:** Re-read the current [`sfm.py`](../../src/gsforge/sfm.py:179), [`test_sfm.py`](../../tests/test_sfm.py:137), active work orders, README, requirements, blueprints, Forge process, latest execution log, prior reviews, [`WO-003-external-evidence-4.1.1.md`](reviews/WO-003-external-evidence-4.1.1.md:1), and linked runtime evidence. No production code or tests were changed.
- **Validation disposition:** No command-execution tool was available, so recorded test/lint/external results were independently assessed as documentary evidence and not rerun. The repository policy identifies command execution as unavailable in this environment at [`00-overview.md`](../../process/00-overview.md:70).
- **WO-001 result:** `blocked`. Rejected-version semantics, semantic `help` rejection, version-command probing, rejected-version continuation, mapper-failure persistence, and dispatch preservation are confirmed. Remaining technical blockers are that top-level `-h` is invoked but not semantically validated, launch failures are mapped to `unknown` instead of normalized `unavailable`, and raw probe streams/exit evidence are not retained at the [`run_sfm()`](../../src/gsforge/sfm.py:834) boundary. Fatal-probe integration coverage is also absent. See [`WO-001-final-independent-review.md`](reviews/WO-001-final-independent-review.md:1).
- **WO-003 result:** `blocked`. The high-level official-binary CPU fallback contract, warning path, mapper-failure persistence, GLOMAP/incremental dispatch separation, and no-auto-switch behavior are confirmed. The remaining technical blocker is that the linked 4.1.1 evidence is not retained with reproducible raw probe/run streams and complete command metadata; warning tests remain narrow but no longer justify the prior mapper-failure or dispatch findings. See [`WO-003-final-independent-review.md`](reviews/WO-003-final-independent-review.md:1).
- **Process-gate result:** `blocked` independently. The initiative remains `planning-revision`; active work orders remain `draft`; dedicated-branch creation evidence and human design approval or an explicit waiver remain absent. These gates are not self-approved.

## 2026-08-19 — final independent review after branch switch and exact 4.1.1 fixture

- **Actor/role:** independent Forge architect/reviewer.
- **Scope:** Re-read the current branch/workspace as represented by the recorded branch switch, active WO-001/WO-003, all prior and final review artifacts, current [`sfm.py`](../../src/gsforge/sfm.py:179), [`test_sfm.py`](../../tests/test_sfm.py:147), README, [`WO-003-external-evidence-4.1.1.md`](reviews/WO-003-external-evidence-4.1.1.md:1), runtime log, requirements/blueprints, and Forge process. No production code or tests were changed.
- **Validation:** No command-execution tool was available. Recorded full-suite/Ruff/external results were independently assessed as documentary evidence, not rerun. The readable 4.1.1 runtime log contains the CPU fallback warnings and 185-camera completion at [`garden-glomap-sfm.txt`](../../garden-glomap-sfm.txt:1210).
- **WO-001 disposition:** `blocked`. Rejected-version continuation, semantic help rejection, version probing, mapper-failure persistence, and dispatch preservation are confirmed. Remaining blockers are top-level `-h` semantic assertion, complete `supported`/`unsupported`/`unknown`/`unavailable` per-command mapping, raw evidence retention at the [`run_sfm()`](../../src/gsforge/sfm.py:834) boundary, and fatal-probe integration coverage. See [`WO-001-final-independent-review-after-branch-switch.md`](reviews/WO-001-final-independent-review-after-branch-switch.md:1).
- **WO-003 disposition:** `blocked`. The exact 4.1.1 fixture resolves the global-mapper option/absence-of-Caspar contract; GLOMAP dispatch, high-level CPU fallback reporting, mapper-failure persistence, and no-auto-switch behavior are confirmed. Remaining findings are missing high-level CPU fallback positive/no-warning boundary coverage and incomplete reproducible raw external evidence linkage. See [`WO-003-final-independent-review-after-branch-switch.md`](reviews/WO-003-final-independent-review-after-branch-switch.md:1).
- **Process-gate disposition:** `blocked` independently. The initiative remains `planning-revision`, active work orders remain `draft`, and no independently retained dedicated-branch creation evidence or human design approval/explicit waiver is recorded. The reviewer did not self-approve those gates.

## 2026-08-19 — independent initiative design review after contract revision

- **Actor/role:** independent Forge design reviewer.
- **State transition:** `planning-revision` → `pass-with-warnings` at the design-review gate; the initiative is not transitioned to `approved`.
- **Work order/review:** initiative design review; revised active [`WO-001-colmap-capability-probe.md`](work-orders/WO-001-colmap-capability-probe.md:1) and [`WO-003-glomap-cpu-fallback.md`](work-orders/WO-003-glomap-cpu-fallback.md:1); deferred [`WO-002-optional-focal-calibration.md`](work-orders/WO-002-optional-focal-calibration.md:1).
- **Action:** Read the governing process, initiative plan, active and deferred work orders, requirements/blueprints, predecessor records, [`README.md`](../../../README.md:1), and this log. Reviewed design coherence only; no production implementation, tests, or implementation-review artifacts were modified.
- **Technical result:** No design blockers remain. WO-001 now provides a normalized availability/version/capability contract with bounded raw probe evidence, diagnostics, and explicit non-fatal metadata versus fatal binary integration rules. WO-003 now intentionally defines high-level truthful official-binary CPU BA fallback behavior, preserves GLOMAP identity, avoids automatic mapper substitution, and places custom GPU builds outside the supported contract. Sequencing, scope, acceptance, validation, risk/recovery, and focal-calibration deferral are coherent.
- **Process findings:** Human design approval remains outstanding; dedicated-branch creation evidence is not independently retained; and the log records production changes before the design gate was approved. These are process findings only, and this review does not assess the correctness of those changes. WO-003's title should also be reconciled with its CPU-fallback reporting scope.
- **Result/next action:** `pass-with-warnings` for technical design coherence. Obtain explicit human design approval and record it before setting the initiative to `approved`; retain branch evidence or record an authorized policy waiver if bypass is intended. Do not self-approve the design, waiver, or pre-gate production changes. Review artifact: [`reviews/initiative-design-review.md`](reviews/initiative-design-review.md:1).

## 2026-08-19 — independent WO-001 implementation review after human approval

- **Actor/role:** independent Forge architect/reviewer.
- **Scope:** Reviewed only WO-001: its current contract, [`sfm.py`](../../src/gsforge/sfm.py:180), [`test_sfm.py`](../../tests/test_sfm.py:147), prior WO-001 reviews, requirements/blueprints, process, initiative approval record, and execution evidence. WO-003 implementation was not reviewed. No production code or tests were changed.
- **Validation basis:** The user-provided focused validation result of 33 tests and recorded quality/external evidence were assessed as documentary evidence and not rerun because command execution is unavailable under [`00-overview.md`](../../process/00-overview.md:70).
- **Confirmed:** Rejected `--version` yields `version_status=unsupported`; the documented `version` command and top-level `-h` are invoked; semantic top-level `help` rejects empty/unusable output; [`run_sfm()`](../../src/gsforge/sfm.py:831) writes bounded raw probe evidence to `logs/colmap-capability-probe.json`; continuation, mapper-failure persistence, fatal probe persistence, and scope tests are present.
- **Remaining WO-001 blockers:** Top-level `-h` invocation is not semantically validated or included in the normalized command map; launch failures are mapped to `unknown` instead of retaining `unavailable`, and fatal results omit complete per-command entries; the fatal integration test does not assert that extraction, matching, and mapping are not called. Raw-evidence serialization is implemented, with only a lower-priority test-strengthening gap for artifact-content assertions.
- **Disposition:** WO-001 implementation review result is `blocked`. Review artifact: [`WO-001-independent-implementation-review-2026-08-19.md`](reviews/WO-001-independent-implementation-review-2026-08-19.md:1). Human design approval and branch evidence are recognized as recorded; this disposition is technical and limited to WO-001.

## 2026-08-19 — independent WO-001 re-review after latest fixes

- **Actor/role:** independent Forge architect/reviewer.
- **Scope:** Reviewed only WO-001: the current [`sfm.py`](../../src/gsforge/sfm.py:180), [`test_sfm.py`](../../tests/test_sfm.py:147), WO-001 contract, prior WO-001 reviews, approval/branch record, and latest validation evidence. No production code or tests were changed.
- **Validation basis:** The recorded 86-test full-suite result and scoped Ruff result were assessed as documentary evidence and were not rerun.
- **Confirmed:** Top-level `-h` is semantically evaluated and retained in the normalized result; fatal probe results contain complete unavailable command entries; raw probe evidence is persisted in `logs/colmap-capability-probe.json`; fatal state persistence and downstream-step suppression are tested; rejected-version continuation, version/help behavior, and mapper dispatch preservation remain intact.
- **Remaining technical blocker:** Normal usable-help command mapping converts launch-failure `unavailable` results to `unknown`, so the required four-state per-command contract is still incomplete. A lower-priority test-strengthening gap remains for positive `-h` semantics and JSON payload-content assertions.
- **Technical disposition:** `blocked` solely for the normalized launch-failure state mapping.
- **Process/documentation disposition:** Human design approval and branch activation are recognized. WO-001 remains marked `draft` with an unchecked completion checklist; this is a separate reconciliation issue, not a new technical finding.
- **Review artifact:** [`WO-001-re-review-after-latest-fixes.md`](reviews/WO-001-re-review-after-latest-fixes.md:1).

## 2026-08-19 — independent WO-001 re-review after launch-state and test fixes

- **Actor/role:** independent Forge architect/reviewer.
- **Scope:** Reviewed only WO-001: the current [`sfm.py`](../../src/gsforge/sfm.py:180), [`test_sfm.py`](../../tests/test_sfm.py:148), WO-001 contract, prior WO-001 reviews, approval/branch record, and latest focused validation. No production code or tests were changed.
- **Validation basis:** The reported 34 focused tests and scoped Ruff results were assessed as documentary evidence and were not rerun.
- **Confirmed:** Top-level `-h` semantic handling and positive normalized assertion, complete four-state command mapping including launch-failure `unavailable`, raw evidence JSON retention, fatal probe persistence with downstream suppression, version/help behavior, and mapper dispatch preservation.
- **Warnings:** JSON-content assertions could be stronger for each raw stream/argument/return-code field. This is test-strengthening only; the implementation serializes the required bounded evidence.
- **Technical disposition:** `pass-with-warnings`; the prior launch-state mapping blocker is resolved.
- **Separate metadata issue:** WO-001 still has stale `draft` status and an unchecked completion checklist in its work-order metadata. Human design approval and branch activation are recognized and are not re-raised as technical blockers.
- **Review artifact:** [`WO-001-re-review-after-launch-state-fixes.md`](reviews/WO-001-re-review-after-launch-state-fixes.md:1).

## 2026-08-19 — independent WO-003 implementation review after real-data validation

- **Actor/role:** independent Forge architect/reviewer.
- **Scope:** Reviewed only WO-003: the revised [`WO-003-glomap-cpu-fallback.md`](work-orders/WO-003-glomap-cpu-fallback.md:1), current [`sfm.py`](../../src/gsforge/sfm.py:321), [`test_sfm.py`](../../tests/test_sfm.py:116), prior WO-003 reviews, README, official 4.1.1 evidence, the latest execution log, and the recorded 185-frame real-data run. No production code or tests were modified.
- **Confirmed:** The intentionally high-level CPU fallback reporting contract is implemented; both representative warning and normal no-warning tests are present; mapper failures persist failed state without changing method; GLOMAP/global and incremental/mapper dispatch remain isolated; official 4.1.1 help/runtime evidence and real-data completion are linked; and the Unicode-console workaround is a separate portability limitation rather than a WO-003 defect.
- **Technical disposition:** `pass-with-warnings`. The concise external artifact is documentary rather than a complete raw-stream archive, and solver detection remains intentionally bounded to the high-level contract; neither is a blocker for the approved scope.
- **Separate metadata/closeout disposition:** Any stale WO-003 `draft` metadata or unchecked completion checklist is a closeout reconciliation issue, not a technical finding. No metadata was changed by this review.
- **Review artifact:** [`WO-003-independent-implementation-review-after-real-data-validation.md`](reviews/WO-003-independent-implementation-review-after-real-data-validation.md:1).

## 2026-08-19 — initiative implementation review

- **Actor/role:** independent Forge initiative implementation reviewer.
- **Scope:** Reviewed the authoritative initiative plan, design review, human approval and branch record, execution log, active WO-001/WO-003 and deferred WO-002, all latest independent implementation reviews, current [`sfm.py`](../../src/gsforge/sfm.py:180), [`test_sfm.py`](../../tests/test_sfm.py:148), README, external 4.1.1 evidence, requirements, blueprints, and Forge implementation/closeout process. No production code or tests were modified.
- **End-to-end result:** WO-001 and WO-003 are sequentially scoped and technically coherent. Capability probing, failed-state handling, GLOMAP CPU-fallback reporting, mapper dispatch preservation, focal-calibration deferral, README documentation, focused tests, recorded full-suite/scoped-Ruff results, and the 185-frame external run align with the approved initiative contract.
- **Blockers:** None identified within the approved implementation scope. The latest independent WO-001 and WO-003 reviews both passed with warnings.
- **Warnings:** Validation results were not rerun in this review environment; WO-001 raw-evidence assertions could be stronger; WO-003 external evidence is concise rather than a complete raw-stream archive; solver reporting is intentionally bounded and transient; initiative/work-order metadata still requires reconciliation; and the separate closeout/manual gate plus Unicode-console follow-up remain outstanding boundaries.
- **Disposition:** Initiative implementation review is `pass-with-warnings` and `ready-for-closeout-review`, but the initiative is not closed. A separate closeout reviewer must perform final branch/diff, evidence, metadata, manual-gate, and human-approval reconciliation before any `closed` transition.
- **Review artifact:** [`initiative-implementation-review.md`](reviews/initiative-implementation-review.md:1).

## 2026-08-19 — separate initiative closeout review

- **Actor/role:** independent Forge closeout reviewer.
- **State transition:** `approved` / `implementation-review` → `blocked` / `closeout-review`.
- **Work order/review:** WO-001 and WO-003 `complete-with-warnings`; initiative closeout review.
- **Action:** Read the closeout process, initiative plan, all work orders and reviews, latest independent reviews, execution log, predecessor closeout, README, requirements, blueprints, recorded runtime evidence, and current branch reference. No production code or tests were changed, and no commands were rerun.
- **Reconciliation:** Scoped implementation and recorded validation align with the approved contract. WO-002 remains deferred. The later passing independent reviews supersede earlier technical blockers for current disposition, while historical blocked reviews remain preserved. The current ref is the required initiative branch, but complete branch status/diff and branch-creation evidence are not retained.
- **Outcome:** `blocked`. The mandatory final human closeout/merge approval required by [`process/00-overview.md`](../../process/00-overview.md:80) is not recorded. Clean-tree, accidental-file, and conflict checks also cannot be verified because no complete `git status`/diff evidence is available. These are explicit manual/process gates, not inferred failures or passes.
- **Warnings/follow-ups:** Validation remains documentary and was not rerun here; WO-001 raw-evidence assertions could be stronger; WO-003 evidence is concise rather than a complete raw-stream archive; solver reporting is intentionally bounded; focal calibration remains deferred; and the Windows Unicode-console workaround remains outside this initiative. See [`reviews/initiative-closeout-review.md`](reviews/initiative-closeout-review.md:1).
- **Next action/owner:** `gsforge maintainers` must obtain and record final human closeout/merge approval or an authorized waiver, retain complete branch status/diff evidence, and repeat closeout before any `closed` transition.
