# Accelerator CI Setup — GPU and TPU

Operational runbook for bringing the GPU and TPU release gates online.

The workflows are already written and committed. Nothing in this document asks
you to author CI. What is missing is infrastructure: a labelled self-hosted
runner with a real NVIDIA device, and a Google Cloud project wired to GitHub
through Workload Identity Federation.

`CONTRIBUTING.md` is the canonical policy reference for both gates — the exact
IAM roles, the provider attribute condition, and the fail-closed rules live
there. This document is the sequencing and verification layer on top of it. If
the two ever disagree, `CONTRIBUTING.md` wins and this file is the stale one.

---

## Why this blocks the release

`publish.yml` declares `needs: [build, gpu-release-gate]`, and that gate queries
the check-runs API for a successful `gpu-axes-passed` on the exact release SHA.
That marker is produced only by `gpu-axes-passed` in `test.yml`, which requires
all three self-hosted CUDA jobs to pass.

With `GPU_RUNNER_ENABLED` unset, those jobs are skipped at the job boundary.
A skipped marker is not evidence, so the gate fails closed and `publish` never
runs.

Phase 1 is now complete: the runner is registered, `GPU_RUNNER_ENABLED=true`,
and every job including the three CUDA axes is green. Phase 2 is not — the
`v5litepod-1` quota is not granted, so `tpu-release.yml` is **not** in
`publish.yml`'s `needs` and runs on `workflow_dispatch` only. Restoring the full
gate means swapping that one job back; `tpu-release.yml` calls the same
`gpu-release-gate.yml`, so the GPU requirement is unaffected either way.

There is a second reason worth naming. CuPy has never executed anywhere in this
project — not in CI, not locally. Phase 1's stated acceptance is "existing suite
green on NumPy **and CuPy**", and only the NumPy half has ever run. Task 4 also
recorded a known CuPy/NumPy divergence (`np.errstate` is a no-op for CuPy) and
deferred validation to GPU CI. Standing up the runner is what closes that.

---

## Phase 0 — TPU quota (do this first, today)

Everything else in this document takes minutes. This one takes days, and
nothing downstream can be tested until it lands.

Cloud console → **IAM & Admin → Quotas** → filter service `TPU API`, region
`europe-west4`. Request TPU v5e (`v5litepod`) chip quota in **`europe-west4-b`**.
One chip is enough for `v5litepod-1`.

`tpu-release.yml` hard-asserts the zone:

```yaml
test "$TPU_ZONE" = europe-west4-b
```

Quota granted in any other zone is unusable without editing the workflow. New
projects commonly show zero TPU v5e quota and the grant is manual.

While you wait, do Phase 1. It is independent and delivers value on its own.

---

## Phase 1 — Self-hosted GPU runner

### 1.1 Confirm the host qualifies

The three CUDA jobs are fail-closed: each asserts a live device before running
tests, and each pipes results through `scripts/assert_junit_no_skips.py`, so a
skipped test is treated as a failure. Do not attach the `gpu` label to a machine
that cannot satisfy these.

Verify on the intended host:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
ls /dev/nvidia*
python -c "import torch; print(torch.cuda.is_available())"
python -c "from numba import cuda; print(cuda.is_available())"
```

Requirements: NVIDIA driver supporting CUDA 12.x, visible `/dev/nvidia*`,
compute capability ≥ 6.0, Python 3.12 available, and roughly 20 GB free disk —
the three jobs install `.[dev,cuda]`, `.[dev,pytorch]` and `.[dev,jax-cuda]`
into separate environments.

You do **not** need the CUDA toolkit or `nvcc`. CuPy, PyTorch and JAX each ship
their own CUDA runtime in their wheels; the driver is what matters.

> **Verified on the primary dev machine (2026-08-08):** RTX 3060 Laptop, driver
> 580.173.02, compute capability 8.6, `/dev/nvidia*` present,
> `torch.cuda.is_available()` → `True` with a real matmul, `numba.cuda` →
> `True`. CuPy is not installed but is a single `pip install cupy-cuda12x` away.
> This host qualifies as the runner.

### 1.2 Register the runner

Settings → Actions → Runners → **New self-hosted runner**, and follow the
download and `config.sh` commands shown on that page. Use the commands GitHub
displays rather than a copied version string — the runner version moves, and a
stale tarball URL fails with a confusing 404.

At the **labels** prompt, add `gpu`.

That label is the entire contract. The jobs select on:

```yaml
runs-on: [self-hosted, gpu]
```

Install as a service so it survives logout:

```bash
sudo ./svc.sh install
sudo ./svc.sh start
sudo ./svc.sh status
```

### 1.3 Do not poison the runner environment

The GPU jobs set `NUMBA_DISABLE_CUDA=0` and `MANIPULAPY_FORCE_CPU=0`. A
service-level export wins over the job-level value, and every device assertion
then fails closed.

Check the service environment does **not** export any of:

- `NUMBA_DISABLE_CUDA=1`
- `MANIPULAPY_FORCE_CPU=1`
- `CUDA_VISIBLE_DEVICES=-1` or `CUDA_VISIBLE_DEVICES=""`

This is easy to get wrong on a machine also used for local development, because
the repo's own test commands set `NUMBA_DISABLE_CUDA=1` deliberately. Hosted CPU
jobs and the TPU job keep that variable at `1` on purpose.

### 1.4 Enable the jobs

Settings → Secrets and variables → Actions → **Variables**:

| variable | value |
| --- | --- |
| `GPU_RUNNER_ENABLED` | `true` |

Set it to `true` only while the labelled runner is online. Leave it unset or
`false` for ordinary work — GitHub cannot evaluate an in-job device probe until
an offline runner has already been assigned, so this variable is the assignment
precondition, not a fallback.

Fork pull requests never receive the self-hosted runner regardless: every GPU
job additionally requires the PR head repository to equal `boelnasr/ManipulaPy`.

### 1.5 Verify

Push to `release/v1.4` and confirm all four previously-skipped jobs now run:

```bash
gh api repos/boelnasr/ManipulaPy/actions/runs/<RUN_ID>/jobs \
  -q '.jobs[] | "\(.conclusion)\t\(.name)"'
```

Expected:

```
success  cupy-gpu
success  torch-cuda
success  jax-cuda
success  gpu-axes-passed
```

`gpu-axes-passed` asserts all three are `success`; it does not pass on partial
results.

---

## Phase 2 — Google Cloud TPU gate

### 2.1 Project and billing

```bash
gcloud projects create manipulapy-ci --name="ManipulaPy CI"
gcloud billing projects link manipulapy-ci --billing-account=<BILLING_ACCOUNT_ID>
```

Billing must be attached before the TPU API will enable.

### 2.2 One-time IAM and federation setup

Run the block in `CONTRIBUTING.md` under **Google Cloud TPU release gate**,
in Cloud Shell, with an account allowed to configure IAM and billing. It:

- enables `compute`, `tpu`, `iamcredentials` and `sts` APIs;
- creates the `manipulapy-tpu-ci` service account with exactly
  `roles/tpu.admin`, `roles/compute.viewer`, `roles/iam.serviceAccountUser`;
- creates the `github` Workload Identity pool and the `github-manipulapy` OIDC
  provider, conditioned on repository and ref;
- binds the repository principal to `roles/iam.workloadIdentityUser`;
- creates a 15 USD budget alert.

**Never create or upload a JSON service-account key.** Authentication is OIDC:
`google-github-actions/auth` exchanges the GitHub token for a short-lived
credential at job time. `.gitignore` already excludes `gha-creds-*.json`, and
the artifact upload step excludes it again, specifically so a key file cannot
leak if one is ever produced.

### 2.3 Repository variables

Settings → Secrets and variables → Actions → **Variables**. These are
`vars.*`, not secrets:

| variable | value |
| --- | --- |
| `GCP_PROJECT_ID` | the project id |
| `GCP_WIF_PROVIDER` | `projects/<PROJECT_NUMBER>/locations/global/workloadIdentityPools/github/providers/github-manipulapy` |
| `GCP_TPU_SERVICE_ACCOUNT` | `manipulapy-tpu-ci@<PROJECT_ID>.iam.gserviceaccount.com` |
| `GCP_TPU_ZONE` | `europe-west4-b` |

`GCP_WIF_PROVIDER` uses the project **number**, not the id:

```bash
gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)'
```

Getting this wrong produces an OIDC failure that does not name the cause.

### 2.4 Protected environment

Settings → Environments → **New environment** → `tpu-release`.

- add yourself as a **required reviewer**;
- restrict deployment branches to `release/v1.4` and tag pattern `v1.4.*`.

The `tpu-contract` job declares `environment: tpu-release`. Without the
environment the job fails immediately.

### 2.5 Verify

`tpu-release.yml` supports `workflow_dispatch`, so it can be exercised before a
real release — but `gpu-release-gate` runs first and requires a successful
`gpu-axes-passed` on that exact SHA. **Phase 1 must be working before Phase 2
can be tested at all.** That ordering is deliberate.

On the run, confirm:

- `Testing exact commit <sha>` and `Tested commit: <sha>` both match the target;
- `test_running_on_real_tpu` passed — it asserts every device reports
  `platform == "tpu"`, so a CPU fallback fails rather than quietly passing;
- all four `assert_junit_no_skips.py` calls passed;
- the `tpu-contract-<sha>` artifact downloaded and retained.

Then verify teardown independently. The delete step is `if: always()` and
re-checks with `describe` then `list`, but confirm yourself:

```bash
gcloud compute tpus tpu-vm list \
  --project="$PROJECT_ID" --zone=europe-west4-b \
  --filter="name~'^manipulapy-'"
```

The filtered list must be empty. A leaked TPU VM bills continuously.

---

## Phase 3 — Release sequencing

Order matters and is the most common way this goes wrong:

1. Bring the runner online, set `GPU_RUNNER_ENABLED=true`.
2. Push the tag. Let CI run to completion.
3. **Wait for `gpu-axes-passed` to actually report success.**
4. Only then publish the GitHub Release, which triggers `publish.yml`.

Publishing immediately after tagging races `gpu-release-gate` against the GPU
jobs. The gate looks up a marker that does not exist yet and fails closed.

Keep `GPU_RUNNER_ENABLED=true` through completion. If it flips to `false`
mid-flight the marker becomes `skipped`, which is not evidence.

Anything that blocks publication: TPU capacity errors, reviewer approval
timeout, CPU fallback, a skipped test, a contract failure, a job timeout,
missing live GPU evidence, or a teardown failure.

---

## Cost

`v5litepod-1` runs roughly 1.20–1.40 USD/hour on demand in `europe-west4`. The
`tpu-contract` job caps at `timeout-minutes: 60`, with a 24-minute inner timeout
on the test step and a 7-minute cap on creation, so a worst-case run costs a
little over a dollar.

The 15 USD budget alert from the setup block is a notification threshold, not a
spending cap. It will not stop a leaked VM from billing — the teardown
verification in §2.5 is what protects you.

The self-hosted runner costs nothing beyond the machine's own power.

---

## Troubleshooting

**GPU jobs stay `skipped`.** `GPU_RUNNER_ENABLED` is not exactly `true`, or the
run is a fork PR. The comparison is a string equality against `'true'`.

**Jobs queue forever.** No online runner carries the `gpu` label. Check
Settings → Actions → Runners shows it Idle, not Offline.

**Device assertion fails on a machine with a working GPU.** Almost always a
service-level environment variable — see §1.3.

**`gpu-axes-passed` is green but `gpu-release-gate` fails.** The gate looks up
the marker for `GITHUB_SHA`, the exact release commit. A marker on a different
commit does not count. Confirm the tag points at the SHA that CI ran.

**OIDC authentication fails.** Three usual causes: `GCP_WIF_PROVIDER` uses the
project id instead of the number; the provider's attribute condition does not
match the branch or tag being run; or the principal binding was never applied.

**The attribute condition rejects a v1.5 tag.** Known limitation. The condition
hardcodes `refs/heads/release/v1.4` and tag prefix `refs/tags/v1.4`, so it must
be widened when the next release branch is cut. The failure presents as a
permission error, not as a stale-condition message.

---

## Checklist

Phase 0

- [ ] TPU v5e quota requested in `europe-west4-b`

Phase 1 — GPU

- [ ] Host verified: driver, `/dev/nvidia*`, torch CUDA, numba CUDA
- [ ] Runner registered with the `gpu` label, installed as a service
- [ ] Service environment free of CPU-forcing variables
- [ ] `GPU_RUNNER_ENABLED=true`
- [ ] `cupy-gpu`, `torch-cuda`, `jax-cuda`, `gpu-axes-passed` all green

Phase 2 — TPU

- [ ] Project created and billing linked
- [ ] `CONTRIBUTING.md` gcloud block run; no JSON key created
- [ ] Four `GCP_*` repository variables set
- [ ] `tpu-release` environment created with a required reviewer
- [ ] `workflow_dispatch` run green, artifact retained, teardown verified empty

Phase 3 — Release

- [ ] Tag pushed, CI green, `gpu-axes-passed` confirmed on the tag SHA
- [ ] Release published from the existing tag
- [ ] TPU teardown verified again after the release run
