# K2VAE Python 3.12 Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the official K2VAE repository and complete a real ETTh1 training-and-prediction run in the existing `py312` Conda environment.

**Architecture:** Preserve upstream history and model code, add a narrow Python 3.12 compatibility layer only where runtime evidence requires it, and drive a shortened version of the official ETTh1 Lightning pipeline. Use the official dataset first and the existing local ETTh1 CSV as the deterministic fallback, then retain the checkpoint, test metrics, and exported CSVs as run evidence.

**Tech Stack:** Python 3.12, Conda `py312`, PyTorch, Lightning, GluonTS, pandas, ETTh1, PowerShell, pytest

**Spec:** `docs/superpowers/specs/2026-09-27-k2vae-py312-integration-design.md`

## Global Constraints

- Use `D:\Application\ApplicationData\anaconda3\envs\py312` and Python 3.12.
- Keep K2VAE model and forecasting logic unchanged unless a demonstrated compatibility defect prevents execution.
- Try the official README dataset source first; fall back to `D:\DingJunjie\CodeFiles\PythonCode\Project2\dataset\ETT-small\ETTh1.csv` if it is unavailable.
- Use one epoch and small positive train, validation, and test batch limits; this is not a paper-score reproduction.
- Prefer CUDA when PyTorch reports it usable; otherwise run the same workflow on CPU.
- Preserve the existing untracked `.vscode/` and `dataset/` directories.
- Never expose credentials or overwrite unrelated packages/files without a failing compatibility check that justifies the change.

## Review Focus

- Python 3.12 lacks a wheel for an upstream exact pin: installation must select and record a compatible release instead of attempting an indefinite source build (Task 2).
- Conda's CUDA virtual-package probe fails under the sandbox or Windows permissions: direct environment executables must still characterize and run the environment (Task 2).
- Official dataset download requires authentication or returns a web page instead of an archive: reject the invalid payload and use the local fallback (Task 3).
- ETTh1 exists under an unexpected folder or has invalid columns/row count: fail before training with a path/schema diagnostic (Task 3).
- CUDA is installed but unusable at runtime, or Windows data-loader workers fail to spawn: retry with CPU and `num_workers=0` without changing model semantics (Task 4).

---

### Task 1: Integrate the Official Upstream Source

**Files:**
- Modify: `.git/config`
- Create from upstream: `README.md`, `pyproject.toml`, `run.py`, `probts/**`, `config/**`, `scripts/**`, and other tracked files from `decisionintelligence/K2VAE`
- Preserve: `.vscode/**`, `dataset/**`, `docs/superpowers/**`

**Interfaces:**
- Consumes: local design and plan commits; upstream `master` branch at `https://github.com/decisionintelligence/K2VAE`
- Produces: a working tree containing the official source plus the local documentation, with upstream history reachable through Git

- [ ] **Step 1: Record the pre-integration state**

Run: `git status --short --branch && git log --oneline --decorate -5`

Expected: the design/plan history is present; `.vscode/` and `dataset/` are untracked and unchanged.

- [ ] **Step 2: Fetch the official repository without replacing local files**

Run: `git remote add upstream https://github.com/decisionintelligence/K2VAE.git` followed by `git fetch upstream master`.

Expected: `upstream/master` resolves to an upstream commit and the local working tree remains unchanged.

- [ ] **Step 3: Merge the unrelated histories and inspect before committing**

Run: `git merge upstream/master --allow-unrelated-histories --no-commit --no-ff` and then `git status --short`.

Expected: upstream source is staged, local documentation remains present, and neither `.vscode/` nor `dataset/` is staged. Resolve only genuine path conflicts; do not discard local files.

- [ ] **Step 4: Verify the imported source shape**

Run: `git ls-files README.md pyproject.toml run.py probts config scripts` and `git diff --cached --check`.

Expected: the upstream entry point, package, and ETTh1 K2VAE config are tracked; the cached diff has no whitespace errors.

- [ ] **Step 5: Commit**

```powershell
git commit -m "chore: import official K2VAE source"
```

### Task 2: Establish a Reproducible Python 3.12 Environment

**Files:**
- Create: `requirements-py312.txt`
- Create: `tests/compat/test_py312_environment.py`
- Modify if required by a demonstrated failure: `pyproject.toml`

**Interfaces:**
- Consumes: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe`, upstream dependency declarations
- Produces: a recorded install set and a passing import/device probe for Python 3.12

- [ ] **Step 1: Capture the environment without relying on `conda activate`**

Run the environment's `python.exe` directly to print `sys.version`, `sys.executable`, `pip --version`, Torch version, `torch.version.cuda`, and `torch.cuda.is_available()`. Also run `pip freeze` into a temporary diagnostic file outside the tracked source.

Expected: Python reports 3.12.x and the executable path belongs to `envs\py312`; CUDA capability is recorded rather than assumed.

- [ ] **Step 2: Write the failing compatibility probe**

Create `test_runtime_is_python_312_and_core_dependencies_import` asserting Python 3.12 and successful imports of `torch`, `lightning`, `gluonts`, `pandas`, `pyarrow`, `jsonargparse`, and `probts`. Create `test_lightning_cli_constructs_help` that invokes `run.py --help` in a subprocess and asserts exit code 0 and the presence of `--config`.

- [ ] **Step 3: Run the probe to identify concrete dependency failures**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/compat/test_py312_environment.py -v`

Expected: FAIL on the first missing/incompatible dependency before installation; record the exact error.

- [ ] **Step 4: Create the Python 3.12 dependency manifest**

Populate `requirements-py312.txt` from upstream requirements. Retain exact upstream versions when wheels/imports work on Python 3.12; replace only incompatible exact pins with the lowest practical Python-3.12-compatible release. Use a released Lightning version rather than the moving `master.zip` URL. Include `pytest` for verification.

- [ ] **Step 5: Install through the environment interpreter**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pip install -r requirements-py312.txt`

Expected: exit code 0 without source-build hangs. If package resolution or import fails, use `superpowers:systematic-debugging`, update only the implicated constraint, and preserve the resolver error as evidence.

- [ ] **Step 6: Run the probe to verify it passes**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/compat/test_py312_environment.py -v`

Expected: PASS for both the core import/device probe and CLI construction.

- [ ] **Step 7: Commit**

```powershell
git add requirements-py312.txt tests/compat/test_py312_environment.py pyproject.toml
git commit -m "build: support K2VAE on Python 3.12"
```

### Task 3: Validate and Stage ETTh1 Data

**Files:**
- Create: `scripts/prepare_etth1.py`
- Create: `tests/compat/test_prepare_etth1.py`
- Create or reuse runtime data: `datasets/ETT-small/ETTh1.csv` (ignored, not committed)
- Modify if required by the upstream loader's actual convention: `.gitignore`

**Interfaces:**
- Consumes: `prepare_etth1.py --destination <path> --fallback <csv>` and the official README download URL
- Produces: exit code 0 and a validated ETTh1 CSV in the exact layout consumed by `DataManager`

- [ ] **Step 1: Inspect the upstream ETTh1 loader contract**

Use `rg` in `probts/data` to identify the exact path, filename, timestamp column, target columns, minimum split sizes, and dataset name for `etth1`. Record these values in test fixtures and script constants; do not invent a second layout.

- [ ] **Step 2: Write failing data-preparation tests**

Add tests named `test_rejects_html_download_payload`, `test_rejects_missing_required_columns`, `test_rejects_too_few_rows`, and `test_uses_valid_local_fallback`. The success test must assert the destination path matches the loader contract and that content/hash matches the fallback CSV.

- [ ] **Step 3: Run tests to verify they fail**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/compat/test_prepare_etth1.py -v`

Expected: FAIL because `scripts.prepare_etth1` does not yet exist.

- [ ] **Step 4: Implement the preparation interface**

Implement `validate_etth1(path: Path) -> None`, `try_official_download(destination: Path) -> bool`, and `prepare_etth1(destination: Path, fallback: Path) -> Path`. Validate HTTP status, content type/archive signature, CSV schema, parseable timestamp values, and minimum row count before accepting input. Use `shutil.copy2` for the fallback and never alter the source CSV.

- [ ] **Step 5: Run tests to verify they pass**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/compat/test_prepare_etth1.py -v`

Expected: PASS for invalid download, invalid schema, insufficient rows, and local fallback.

- [ ] **Step 6: Attempt official data, then stage the deterministic fallback if needed**

Run `scripts/prepare_etth1.py` with the loader's destination and `D:\DingJunjie\CodeFiles\PythonCode\Project2\dataset\ETT-small\ETTh1.csv` as fallback.

Expected: the official artifact is accepted only if valid; otherwise the script reports the reason and copies the validated local ETTh1 file. The destination is excluded from Git.

- [ ] **Step 7: Exercise the actual K2VAE data manager**

Add and run `test_real_etth1_loader_produces_train_val_test_batches`, using context length 96 and prediction length 96. Assert each loader yields a batch and expected tensors contain finite values.

Expected: PASS against the staged real ETTh1 data.

- [ ] **Step 8: Commit**

```powershell
git add scripts/prepare_etth1.py tests/compat/test_prepare_etth1.py .gitignore
git commit -m "test: validate ETTh1 data pipeline"
```

### Task 4: Run End-to-End K2VAE Training and Prediction

**Files:**
- Create: `config/ltsf/etth1/k2vae_smoke.yaml`
- Create: `scripts/run_etth1_smoke.ps1`
- Create: `tests/compat/test_smoke_config.py`
- Modify only if a runtime defect is reproduced: the smallest implicated file under `probts/**` or `run.py`
- Generate: `results/smoke/**` (ignored runtime evidence)

**Interfaces:**
- Consumes: `scripts/run_etth1_smoke.ps1 -PythonExe <path> -DataRoot <path> -ResultRoot <path> [-Accelerator cpu|gpu]`
- Produces: exit code 0 plus at least one checkpoint, Lightning test metrics, experiment summary, and exported prediction/metric CSV

- [ ] **Step 1: Write failing smoke-configuration tests**

Add `test_smoke_config_uses_k2vae_and_bounded_work`, asserting the forecaster class is `k2VAEModel`, `max_epochs` is 1, every train/validation/test limit is positive and bounded, and Windows worker count is 0. Add `test_runner_rejects_missing_python_or_data`, asserting clear nonzero exits before Lightning starts.

- [ ] **Step 2: Run tests to verify they fail**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/compat/test_smoke_config.py -v`

Expected: FAIL because the smoke config and runner do not exist.

- [ ] **Step 3: Create the smoke config and runner**

Derive `k2vae_smoke.yaml` from the official ETTh1 K2VAE config. Set one epoch, deterministic small positive batch limits, `num_workers: 0`, and an explicit test-batch limit while retaining the model architecture. The PowerShell runner must use the passed Python executable directly, validate paths, choose GPU only when the Torch probe succeeds, and call `run.py` with explicit data/result/context/prediction arguments.

- [ ] **Step 4: Run configuration tests to verify they pass**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/compat/test_smoke_config.py -v`

Expected: PASS for bounded configuration and path preflight.

- [ ] **Step 5: Execute the real ETTh1 workflow**

Run the PowerShell wrapper with the `py312` interpreter, staged dataset root, and `results/smoke`. Keep the command's complete output in `results/smoke/run.log` without printing secrets.

Expected: at least one optimizer step and one epoch complete; a checkpoint is saved and reloaded; Lightning test finishes; summary and CSV export messages appear.

- [ ] **Step 6: Diagnose any runtime failure before patching**

If Step 5 fails, invoke `superpowers:systematic-debugging`, reproduce the smallest failing boundary, add a regression test beside `tests/compat`, make the minimum compatibility fix, rerun that regression test, and then rerun Step 5. For CUDA initialization failure, rerun with CPU; for worker-spawn failure, keep `num_workers=0`.

- [ ] **Step 7: Verify runtime artifacts**

Run a PowerShell check that asserts `results/smoke` contains a nonempty `.ckpt`, test metrics, experiment summary, and at least one exported `.csv`. Search `run.log` for `Traceback` and failed-state markers.

Expected: all required files exist and are nonempty; no unhandled traceback is present.

- [ ] **Step 8: Commit source/configuration changes only**

```powershell
git add config/ltsf/etth1/k2vae_smoke.yaml scripts/run_etth1_smoke.ps1 tests/compat probts run.py .gitignore
git commit -m "feat: add ETTh1 K2VAE smoke workflow"
```

Do not commit checkpoints, datasets, logs, or generated result CSVs.

### Task 5: Document and Reverify the Working Workflow

**Files:**
- Create: `RUN_PY312.md`
- Modify: `README.md` (add a short link to `RUN_PY312.md` only)

**Interfaces:**
- Consumes: the exact passing commands and versions from Tasks 2-4
- Produces: a copy-paste Windows/Conda run guide and final evidence-backed verification report

- [ ] **Step 1: Write the run guide**

Document the requested activation form, the direct-interpreter form used for reliability, dependency installation, data preparation, smoke execution, full-run parameter changes, expected output paths, CPU fallback, and the exact tested package/device versions. Clearly distinguish the short validation run from paper benchmark reproduction.

- [ ] **Step 2: Run the complete automated verification**

Run: `D:\Application\ApplicationData\anaconda3\envs\py312\python.exe -m pytest tests/compat -v`

Expected: all compatibility, data, and smoke-configuration tests pass.

- [ ] **Step 3: Re-run the documented command from a clean result subdirectory**

Use a new path under `results/smoke` and execute the command exactly as written in `RUN_PY312.md`.

Expected: training, checkpoint reload, prediction/test, and artifact verification all pass a second time.

- [ ] **Step 4: Inspect the final diff and repository state**

Run: `git diff --check`, `git status --short`, and `git log --oneline --decorate -10`.

Expected: no whitespace errors; only deliberately ignored runtime outputs remain outside Git; `.vscode/` and the user's original `dataset/` content are unchanged.

- [ ] **Step 5: Commit**

```powershell
git add RUN_PY312.md README.md
git commit -m "docs: add K2VAE Python 3.12 run guide"
```

- [ ] **Step 6: Perform completion verification**

Invoke `superpowers:verification-before-completion`, rerun its required checks, and report the exact command, device, dataset path, checkpoint path, metric/result files, test counts, and any compatibility deviations from upstream.
