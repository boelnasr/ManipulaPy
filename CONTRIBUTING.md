# Contributing to ManipulaPy

Thank you for your interest in contributing to ManipulaPy! We welcome contributions that help improve this project and make it more robust, reliable, and user-friendly.

---

## Contribution Guidelines

### How to Contribute

- Open an issue if you notice a bug, have a feature request, or a question.
- Fork the repository and create a new branch for your feature or fix.
- Ensure your code adheres to PEP8 and includes tests where applicable.
- Submit a Pull Request (PR) with a clear description of what it does.

### Testing

- We use `pytest` to run our test suite.
- Tests should cover new features and significant changes to existing logic.
- Include clear docstrings and comments in your test cases.

---

## Development

### Build & Development Commands

```bash
# Install in dev mode
pip install -e ".[dev]"

# Run all tests
python -m pytest tests/ -v

# Run a single test file
python -m pytest tests/test_kinematics.py -v

# Run a single test
python -m pytest tests/test_control.py::TestManipulatorController::test_pid_control -v

# Run tests by marker
python -m pytest tests/ -v -m "not (cuda or vision or simulation)"

# Run with coverage
python -m pytest tests/ --cov=ManipulaPy --cov-report=term-missing

# Lint (check only)
python -m black --check ManipulaPy/ tests/
python -m isort --check-only ManipulaPy/ tests/
python -m flake8 ManipulaPy/ tests/

# Lint (auto-fix)
python -m black ManipulaPy/ tests/
python -m isort ManipulaPy/ tests/

# Build docs
python -m sphinx -b html docs/source docs/build/html
```

### CI/CD

- **test.yml** (*CI Tests*): hosted NumPy, Torch-CPU, and JAX-CPU contracts plus
  CuPy, Torch-CUDA, and JAX-CUDA contracts on the self-hosted GPU runner. Every
  axis runs the public-API freeze; Torch/JAX axes also run gradient contracts.
- **tpu-release.yml** (*TPU Release Contract*): release-only, reviewer-approved
  Google Cloud TPU validation for the exact commit. It requires a successful
  `gpu-axes-passed` check for the same SHA, provisions one `v5litepod-1`, uploads
  JUnit evidence, and always tears the TPU down.
- **lint.yml** (*Lint with flake8 and black*): `flake8 ManipulaPy tests --max-line-length=88` + `black --check`; auto-commits formatting fixes (`contents: write`).
- **publish.yml** (*Publish to PyPI*): triggered when a GitHub **Release** is published; builds the sdist/wheel, waits for the GPU/TPU release workflow, and uploads via PyPI **Trusted Publishing** (OIDC `id-token: write`, `pypi` environment) — no API token.
- **draft-pdf.yml**: builds the JOSS `paper.pdf` on `v*` tags and attaches it to the release.
- **codeql.yml** (*CodeQL Advanced*) and **scorecard.yml** (*Scorecard supply-chain security*): security analysis on push/PR plus weekly schedules.

### Code Review & Branch Protection

- All changes to `main` land through a pull request. Direct pushes are disabled by
  branch protection.
- A PR is mergeable only once its required checks are green: `numpy-cpu`,
  `torch-cpu`, `jax-cpu`, `non-tpu-regression (3.9)`,
  `non-tpu-regression (3.10)`, `non-tpu-regression (3.11)`,
  `non-tpu-regression (3.12)`, `build-check`, and
  `Code Linting and Autoformat`. The
  self-hosted GPU checks are day-to-day opt-in checks and become mandatory
  exact-SHA evidence through the release gate described below.
- This is a solo-maintainer project. External PRs receive a maintainer review
  before merge; maintainer-authored PRs rely on the CI gates above plus
  self-review. Community review on any PR is welcome and encouraged — leave a
  comment on the PR.

For architecture details, class hierarchy, GPU/CPU strategy, and code conventions,
see [ARCHITECTURE.md](ARCHITECTURE.md).

### Self-hosted GPU runner

The CUDA jobs target exactly `runs-on: [self-hosted, gpu]`. Register a dedicated
Linux x86-64 GitHub Actions runner for this repository and add the custom `gpu`
label during `config.sh` setup. Do not attach the label to a machine without a
working NVIDIA device: once a job is assigned, every device assertion and the
JUnit no-skips check is fail-closed.

The runner host must provide:

- a supported NVIDIA driver, CUDA 12 runtime/toolkit, `nvidia-smi`, and visible
  `/dev/nvidia*` devices;
- Python 3.12 and the normal GitHub runner build prerequisites; and
- enough isolated disk space for the jobs to install `.[dev,cuda]`,
  `.[dev,pytorch]`, and `.[dev,jax-cuda]` without reusing a stale environment.

Before enabling jobs, verify `nvidia-smi`, a CuPy allocation, both
`torch.cuda.is_available()` and a CUDA tensor, and `jax.devices()` with
`JAX_PLATFORMS=cuda`. The GPU jobs set `NUMBA_DISABLE_CUDA=0` and
`MANIPULAPY_FORCE_CPU=0`; do not export either variable as true in the runner
service, and do not hide the device with `CUDA_VISIBLE_DEVICES=-1`. Hosted CPU
jobs and the TPU job deliberately use `NUMBA_DISABLE_CUDA=1`.

GitHub cannot evaluate an in-job device probe until an offline self-hosted
runner has already been assigned, so a probe cannot neutralize an unavailable
runner. Repository variable `GPU_RUNNER_ENABLED` is the assignment precondition:

1. Leave it unset or set it to `false` for ordinary PR work while the runner is
   offline. GPU jobs and `gpu-axes-passed` are skipped at the job boundary.
2. Set it to `true` only while the labeled runner is online. The three live
   CUDA jobs must pass before `gpu-axes-passed` succeeds.
3. For a release commit or `v1.4.*` tag, keep it `true` through completion. The
   TPU release workflow looks up `gpu-axes-passed` for the exact release SHA and
   fails closed if the check is absent, skipped, pending, cancelled, or failed.

Fork pull requests never receive a self-hosted runner, even when
`GPU_RUNNER_ENABLED=true`: every GPU job also requires the pull request head
repository to equal `boelnasr/ManipulaPy`. Run fork contributions only on the
hosted CPU and non-TPU regression jobs until their changes have landed on a
trusted branch.

Branch protection may treat the skipped day-to-day marker as neutral, but a
skipped marker is never release evidence. Record the live Actions URL in the
release checklist before starting the TPU workflow.

### Google Cloud TPU release gate

The maintainer performs this one-time setup in Cloud Shell. Replace the first
two values, then run the commands with an account allowed to configure IAM and
billing. Do not create or upload a JSON service-account key.

```bash
PROJECT_ID="your-google-cloud-project"
BILLING_ACCOUNT_ID="000000-000000-000000"
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
SERVICE_ACCOUNT="manipulapy-tpu-ci@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud services enable \
  compute.googleapis.com \
  tpu.googleapis.com \
  iamcredentials.googleapis.com \
  sts.googleapis.com \
  --project="$PROJECT_ID"

gcloud iam service-accounts create manipulapy-tpu-ci \
  --project="$PROJECT_ID" \
  --display-name="ManipulaPy TPU release CI"

for role in roles/tpu.admin roles/compute.viewer roles/iam.serviceAccountUser; do
  gcloud projects add-iam-policy-binding "$PROJECT_ID" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="$role"
done

gcloud iam workload-identity-pools create github \
  --project="$PROJECT_ID" --location=global \
  --display-name="GitHub Actions"

gcloud iam workload-identity-pools providers create-oidc github-manipulapy \
  --project="$PROJECT_ID" --location=global \
  --workload-identity-pool=github \
  --display-name="boelnasr/ManipulaPy releases" \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref" \
  --attribute-condition="assertion.repository == 'boelnasr/ManipulaPy' && (assertion.ref == 'refs/heads/release/v1.4' || assertion.ref.startsWith('refs/tags/v1.4'))"

gcloud iam service-accounts add-iam-policy-binding "$SERVICE_ACCOUNT" \
  --project="$PROJECT_ID" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github/attribute.repository/boelnasr/ManipulaPy"

gcloud billing budgets create \
  --billing-account="$BILLING_ACCOUNT_ID" \
  --display-name="ManipulaPy TPU release" \
  --budget-amount=15USD \
  --threshold-rule=percent=0.5 \
  --threshold-rule=percent=0.9 \
  --threshold-rule=percent=1.0
```

Confirm the service account has only `roles/tpu.admin`,
`roles/compute.viewer`, and `roles/iam.serviceAccountUser` at the project level,
plus `roles/iam.workloadIdentityUser` on the service account. The provider
condition must remain exactly repository `boelnasr/ManipulaPy` and either branch
`refs/heads/release/v1.4` or tag prefix `refs/tags/v1.4`.

Create these GitHub repository variables (Settings → Secrets and variables →
Actions → Variables):

- `GCP_PROJECT_ID`: the project ID;
- `GCP_WIF_PROVIDER`:
  `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github/providers/github-manipulapy`;
- `GCP_TPU_SERVICE_ACCOUNT`: the full service-account email; and
- `GCP_TPU_ZONE`: exactly `europe-west4-b`.

Create the protected GitHub environment `tpu-release`. Require maintainer
reviewer approval and restrict deployments to branch `release/v1.4` and tags
matching `v1.4.*`. The workflow has concurrency group
`manipulapy-tpu-release`, does not cancel an in-progress run, and times out after
60 minutes.

For each release, first obtain live `gpu-axes-passed` evidence on the exact SHA.
Then dispatch **TPU Release Contract**, approve `tpu-release`, and retain its
Actions URL plus the `tpu-contract-SHA` JUnit artifact. Confirm the real-TPU
platform assertion, linalg/FK/dynamics/gradient contracts, exact printed commit,
and zero-skips check are green. Finally verify teardown independently:

```bash
gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" --zone=europe-west4-b \
  --filter="name~'^manipulapy-'"
```

The filtered list must be empty. TPU capacity errors, reviewer timeouts, CPU
fallback, skipped tests, contract failures, timeouts, missing live GPU evidence,
or teardown failures block publishing.

---

## Regression Test Discipline

Every bug fix follows a RED→GREEN workflow:

1. Write the failing test first. Run it and confirm it fails with the bug's actual
   symptom (not an unrelated error). Commit the failing test on its own or together
   with the fix in a single commit whose message names the symptom.
2. Apply the fix. The test must turn green. Do not merge a fix whose test is still
   red or that only passes because the assertion was weakened.
3. Each fix gets its own focused commit. Bundling unrelated changes into the same
   commit makes bisect and revert harder.

The canonical regression file is `tests/test_v132_regressions.py`. Add new test
classes there for new modules or subsystems.

**Optional-dependency behavior** (sim and control importable without cupy/pybullet)
is tested with a subprocess-style probe so the check is side-effect-free:

```python
def test_sim_module_imports_without_pybullet(self):
    """ManipulaPy.sim must import even when pybullet is missing."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.modules['pybullet'] = None; import ManipulaPy.sim"],
        capture_output=True, text=True,
    )
    self.assertEqual(result.returncode, 0,
                     f"sim import failed: {result.stderr}")
```

Use this pattern (not in-process `sys.modules` mutation) for any test that needs
to simulate a missing optional dependency — in-process reloads leak across test
modules and corrupt shared state.

---

## AI Usage Policy

We value transparency and responsibility in software development. While generative AI tools (e.g., GitHub Copilot, ChatGPT, CodeWhisperer) can be helpful in drafting and brainstorming, **all contributions must be human-verified and reviewed**.

### Acceptable AI Use

You may use AI tools to:
- Generate boilerplate code or documentation drafts.
- Explore alternative implementations (after reviewing and testing).
- Assist with initial responses to issues (after editing for clarity and correctness).

### Unacceptable AI Use

You must **not**:
- Submit AI-generated code or text **without understanding and testing it**.
- Use AI to auto-respond to issues or PRs without human oversight.
- Include unexplained or unverifiable code from AI tools.

---

##  Maintainer Commitment

As the project maintainer:
- I may use AI tools to accelerate documentation or draft code—but **every line is reviewed and validated before merging**.
- I take full responsibility for all merged content, whether AI-assisted or not.
- Users and contributors will receive **thoughtful, human-reviewed support**.

This policy aligns with [JOSS guidance](https://joss.theoj.org/about#ai-policy), which states:
> "Authors are responsible for understanding and explaining submitted code and its provenance, and should respond in good faith to reviewer questions about LLM use as they would with any other topic."

---

##  Contributor AI Use Policy

If you use AI assistance for your contribution:
- Mention it in your pull request or commit message.
- Only submit code you fully understand and have tested.
- Be prepared to explain your changes.

---

## 🔄 Policy Evolution

This AI policy may evolve as community practices and journal guidelines (including JOSS) develop. We welcome suggestions and questions.

---

Last updated: 2025-07-26
