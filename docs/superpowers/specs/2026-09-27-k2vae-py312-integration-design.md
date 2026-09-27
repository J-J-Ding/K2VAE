# K2VAE Python 3.12 Integration Design

## Objective

Bring the official `decisionintelligence/K2VAE` source into this project,
run it with the existing Conda environment `py312`, and verify a real ETTh1
training-and-prediction workflow. The verification must cover data loading,
training, checkpoint creation and reload, test inference, and result export.

## Constraints

- Use `D:\Application\ApplicationData\anaconda3\envs\py312` and Python 3.12.
- Keep the K2VAE model and forecasting logic unchanged unless a compatibility
  defect prevents execution.
- Prefer the dataset source linked by the official README. If it is unavailable,
  use `D:\DingJunjie\CodeFiles\PythonCode\Project2\dataset\ETT-small\ETTh1.csv`.
- Keep runtime practical by reducing epochs and batch limits. This is an
  execution check, not a reproduction of the paper's final benchmark scores.
- Run on CUDA when the installed PyTorch build and hardware support it;
  otherwise use CPU.

## Source Integration

The official upstream repository is
`https://github.com/decisionintelligence/K2VAE`. Its history will be fetched and
merged into the repository rooted at this directory. Existing local files such
as `.vscode` and this design document will be preserved.

Upstream files remain recognizable. Local compatibility edits will be small and
documented so they can be reviewed or rebased against future upstream changes.

## Python 3.12 Compatibility

The upstream project declares Python 3.10+ but pins several packages released
before Python 3.12. Installation will begin from the upstream dependency list,
then replace only packages that lack Python 3.12 wheels or fail at import time
with compatible releases. Likely candidates include pandas, pyarrow, Ray,
Pydantic, orjson, Lightning, and their transitive dependencies.

The existing `py312` environment may already contain packages used by other
projects. Before installation, its Python, pip, PyTorch, CUDA, and key package
versions will be recorded. Dependency changes will be limited to what K2VAE
needs, and the final working versions will be captured in a reproducible
requirements file or setup note.

## Dataset Flow

The official Google Drive or Baidu source will be attempted first. A downloaded
archive will be inspected before extraction and placed under the layout expected
by K2VAE's data loader. If download, authentication, or archive retrieval fails,
the local ETTh1 CSV will be used instead.

After source integration, the loader implementation will determine the exact
folder and filename convention. The input CSV will be validated for existence,
nonzero size, required timestamp and target columns, and enough rows for the
configured context and prediction lengths.

## Training and Prediction Verification

The experiment will start from `config/ltsf/etth1/k2vae.yaml` with explicit
ETTh1 dataset path, context length, prediction length, and result directory.
For a practical smoke run, the trainer will use one epoch and a small positive
number of train, validation, and test batches. Worker count will be reduced on
Windows if multiprocessing causes startup issues.

Success requires all of the following:

1. The ETTh1 loader produces training, validation, and test batches.
2. K2VAE completes at least one optimizer step and one epoch.
3. A Lightning checkpoint is written.
4. The checkpoint is loaded for the test stage.
5. Test prediction completes without an unhandled exception.
6. The project writes its summary and prediction/metric CSV artifacts.

Metric quality is not an acceptance criterion for this shortened run.

## Error Handling

Failures will be diagnosed at the first failing boundary: environment,
dependency import, dataset parsing, model construction, device setup, training,
checkpoint reload, or prediction export. Compatibility fixes will target that
boundary instead of changing unrelated code. If CUDA fails, the same experiment
will be retried on CPU; this fallback does not weaken the workflow acceptance
criteria.

## Deliverables

- Official K2VAE source in the current project directory.
- Minimal Python 3.12 and Windows compatibility changes, if required.
- A repeatable ETTh1 smoke-run configuration or command.
- Checkpoint, logs, test metrics, and exported CSV evidence from a completed run.
- A concise run guide using the user's `py312` Conda environment.
