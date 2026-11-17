# Local AI Validation Framework (Hermes-supervised)

Runs product tests on local/VM shell through a **Hermes supervisor**. Hermes is the normal execution entrypoint. Legacy Python orchestration remains in the repo only for debugging and is not part of the user-facing flow.

## What this includes
- Hermes-supervised execution mode
- Hermes compatibility runtime for non-native features:
  - workflow API
  - agent/session control API
  - spawn-workers API
  - task graph API
  - retry policy API
  - structured tool-calling contract
- Gate decision: `PASS`, `PASS_WITH_RISK`, `FAIL`
- Hermes-powered post-run analysis from logs
- `job0_test` as the **first gate test** (`run_first=true`)
- `fastpath` as a dedicated long-running Jenkins gate (8-hour budget) before perf suites

## Files
- `configs/suites.json` test suites and commands
- `configs/suites.fastpath.json` Jenkins-triggered fastpath gate only
- `configs/suites.install.json` Hermes-controlled CPD install gate
- `scripts/run_hermes_supervisor.py` Hermes supervisor runner; `--control-plane hermes` enables Hermes task planning and retry decisions
- `scripts/run_local_orchestrator.py` legacy debug-only runner; `--control-plane hermes` enables the experimental prompt/retry planner
- `backups/20260602/` frozen copies of the current runner scripts before the planner refactor
- `out/` generated logs and reports

## Prerequisites
1. Python 3.9+
2. Hermes Agent installed locally:
   - `curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash`
3. Hermes model/provider configured:
   - example one-shot mode from docs: `hermes chat -q "hello"`

## Configure your tests
Edit `configs/suites.json` and replace `cmd` values with your real commands.
For install runs, use `configs/suites.install.json` and pass the installer path via `HERMES_INSTALL_SCRIPT` or remote-runner flags.

### First test gate behavior
- Any suite with `"run_first": true` runs sequentially before parallel suites.
- If a `run_first` suite is `critical=true` and fails, remaining suites are skipped.
- Hermes still summarizes available results/logs.

## Run (local only, no Jenkins)
```bash
cd ai_installer/hermes_local_framework
python3 scripts/run_hermes_supervisor.py \
  --config configs/suites.json \
  --outdir out/run-$(date +%Y%m%d-%H%M%S) \
  --max-workers 4 \
  --max-retries 1
```

This sends the suite plan and retry policy to Hermes CLI and expects Hermes to control execution and write artifacts under the requested `out/` directory.
The default backend is `compat-runtime`: Hermes is the supervisor, and the compatibility runtime executes the task graph deterministically from `out/<run>/hermes_session.json`.

## Run install under Hermes control
```bash
cd ai_installer/hermes_local_framework
HERMES_INSTALL_SCRIPT=/root/cp4d-install-automation/cpd-install.sh \
HERMES_INSTALL_BRANCH=5.3-release \
HERMES_SCHEDULER_NS=ibm-cpd-scheduler \
python3 scripts/run_hermes_supervisor.py \
  --config configs/suites.install.json \
  --outdir out/install-$(date +%Y%m%d-%H%M%S) \
  --max-workers 1 \
  --max-retries 1
```

## Run from Mac, execute on s390x VM (scp + remote execute)
```bash
cd ai_installer/hermes_local_framework
chmod +x scripts/run_remote_vm_from_mac.sh
scripts/run_remote_vm_from_mac.sh \
  --host <s390x-vm-host> \
  --user root \
  --cluster <cluster-name> \
  --install-script /root/cp4d-install-automation/cpd-install.sh \
  --install-branch 5.3-release \
  --scheduler-ns ibm-cpd-scheduler \
  --config configs/suites.install.json \
  --max-retries 1 \
  --max-workers 4
```

## Jenkins flow for long-running fastpath then perf
Run the fastpath gate first:
```bash
python3 scripts/run_hermes_supervisor.py \
  --config configs/suites.fastpath.json \
  --outdir out/fastpath-$(date +%Y%m%d-%H%M%S) \
  --max-workers 1 \
  --max-retries 1
```

After fastpath succeeds, run the perf suite separately:
```bash
python3 scripts/run_hermes_supervisor.py \
  --config configs/suites.performance.json \
  --outdir out/perf-$(date +%Y%m%d-%H%M%S) \
  --max-workers 4 \
  --max-retries 1
```

This command will:
1. Pack `hermes_local_framework` + `job0`
2. SCP to VM
3. Run orchestrator on VM
4. Download output artifacts to `~/ds-ai/hermes-runs`

Optional provider/model override:
```bash
HERMES_MODEL="gemma4:12b" \
python3 scripts/run_hermes_supervisor.py --config configs/suites.json --outdir out/latest --max-workers 4
```

## Output
- `out/<run>/summary.json` machine-readable results and gate
- `out/<run>/summary.txt` concise status
- `out/<run>/*.log` per-suite raw logs
- `out/<run>/hermes_analysis.txt` RCA from Hermes or Hermes-written report
- `out/<run>/hermes_session.json` compatibility runtime session manifest
- `out/<run>/hermes_task_graph.json` compatibility runtime task graph
- `out/<run>/hermes_supervisor_prompt.txt` Hermes-first execution prompt
- `out/<run>/hermes_supervisor_output.txt` Hermes supervisor transcript
- `out/<run>/events.jsonl` stage start/end events and heartbeats
- `out/<run>/status-board.json` live suite/worker status board
- `out/<run>/remote_command_trace.jsonl` remote SSH/SCP command trace

## Schemas And Contracts
- `schemas/execution-event.schema.json`
- `schemas/status-board.schema.json`
- `schemas/suite-result.schema.json`
- `schemas/adapter-result.schema.json`
- `runtime/adapters.py` adapter interface contract
- `docs/framework_refactor_plan.md` file-level refactor plan and implementation order

## Legacy fallback
- `scripts/run_local_orchestrator.py` and `scripts/perf_parallel_agent.py` remain only for debugging or migration fallback.
- The legacy orchestrator is preserved as a backup, and the experimental Hermes control-plane lives behind `--control-plane hermes`.
- Normal runs should go through `run_hermes_supervisor.py` or `run_remote_vm_from_mac.sh`.

## Gate logic
- Any failed `critical=true` suite => `FAIL`
- Only non-critical failures => `PASS_WITH_RISK`
- No failures => `PASS`

## Notes
- This is local-first orchestration (you asked to ignore Jenkins for now).
- Later you can reuse the same script in Jenkins with one stage call.


By default it also auto-runs cpdctl setup on VM using:
`job0/setup_cpdctl_from_cluster.sh --cluster <cluster-name>`.
Disable with `--auto-setup-cpdctl false`.
