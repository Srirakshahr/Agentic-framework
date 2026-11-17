# DS Hermes Agent

Single agentic entrypoint for all DataStage automated scenarios.
**One file. One command. The agent does everything.**

## What the agent does

```
agent.py
 ├─ Reads suite config
 ├─ Calls Hermes (via OpenAI-compatible API or CLI fallback)
 └─ Agent loop:
      list_suites → run_suite (run_first first) → read_log (on failure)
      → retry decision → run_suite again → write_analysis → set_gate
```

The Hermes agent controls the entire execution: planning, running, retrying,
diagnosing, and reporting. No separate orchestration or analysis scripts.

---

## Prerequisites

```bash
# Start Ollama with Qwen (default model)
ollama run qwen2.5-coder:14b-64k
```

That's it. `agent.py` auto-installs the Hermes CLI on Mac at startup if it's not already present.
Hermes is **never installed on the VM** — s390x is not supported by the Hermes framework.

---

## Execution — your cluster

Exact drop-in for the old `run_remote_vm_from_mac.sh` command:

```bash
cd ~/Desktop/DS_Hermes_Framework/hermes_local_framework

python3 agent.py \
  --remote-host  api.mycluster.example.com \
  --remote-user  root \
  --ssh-password 'pass2ibm' \
  --auto-setup-cpdctl \
  --cluster      mycluster \
  --cpd-url      'https://cpd-ds.apps.mycluster.example.com' \
  --config       configs/suites.json \
  --max-workers  4 \
  --max-retries  1 \
  --extra-path   /home/user/Desktop/cpd-perf-automation
```

## Other common runs

```bash
# Install gate on VM
python3 agent.py \
  --config configs/suites.install.json \
  --remote-host <vm> --remote-user root --ssh-password '<pass>' \
  --cluster <cluster> --auto-setup-cpdctl

# Performance suites only
python3 agent.py \
  --config configs/suites.performance.json \
  --remote-host <vm> --remote-user root --ssh-password '<pass>' \
  --cluster <cluster>

# Local-only (no VM)
python3 agent.py --config configs/suites.json

# Mac browser tests
python3 agent.py --config configs/suites.mac_browser.json
```

---

## Configuration

### Model / provider (pick one)

| Method | Example |
|--------|---------|
| Hermes config (`~/.hermes/config.yaml`) | `hermes setup` |
| Environment variables | `HERMES_MODEL=gemma4:12b` |
| CLI flags | `--model gemma4:12b` |

### Supported providers

| `--provider` | API base used |
|---|---|
| *(empty)* | `http://127.0.0.1:11434/v1` (local Ollama / Hermes) |
| `openai` | `https://api.openai.com/v1` |
| `openrouter` | `https://openrouter.ai/api/v1` |
| `custom` | value from `--api-base` or hermes config `model.base_url` |

### Flag mapping — old script → agent.py

| Old `run_remote_vm_from_mac.sh` | New `agent.py` |
|---------------------------------|----------------|
| `--host` | `--remote-host` |
| `--user` | `--remote-user` |
| `--ssh-password` | `--ssh-password` |
| `--cluster` | `--cluster` |
| `--cpd-url` | `--cpd-url` |
| `--auto-setup-cpdctl true` | `--auto-setup-cpdctl` (flag) |
| `--config` | `--config` |
| `--max-workers` | `--max-workers` |
| `--max-retries` | `--max-retries` |
| `--extra-path` | `--extra-path` |

### All flags

```
# Suite
--config              Suite config JSON            [configs/suites.json]
--outdir              Output dir                   [out/run-YYYYMMDD-HHMMSS]
--max-workers         Parallel workers             [4]
--max-retries         Retries per suite            [1]
--max-iterations      Agent loop cap               [80]

# LLM / Hermes
--provider            openai | openrouter | custom | (empty=local)
--model               Model name                   [hermes3]
--api-base            Override API base URL
--api-key             API key (env vars / ~/.hermes/config.yaml)

# Remote VM
--remote-host         VM hostname or IP
--remote-user         SSH user                     [root]
--remote-base         Base dir on VM               [/root/DS_Hermes_Framework]
--ssh-password        SSH password (needs sshpass)

# Cluster / CPD credentials
--cluster             Cluster name, e.g. mycluster
--cpd-url             CPD base URL
--cpd-user            CPD username                 [cpadmin]
--cpd-password        CPD password
--cpd-namespace       OCP namespace                [ds]
--auto-setup-cpdctl   Run setup_cpdctl_from_cluster.sh on VM

# Extra inputs
--extra-path          Local file/dir to bundle onto VM (repeatable)

# Artifacts download
--local-artifacts     Local dir for downloaded artifacts
                      [~/ds-ai/hermes-runs]
```

---

## Tools the agent can call

| Tool | What it does |
|------|-------------|
| `list_suites` | Confirm suite plan from config |
| `run_suite` | Execute one suite (local or remote) |
| `read_log` | Tail a suite log for failure diagnosis |
| `get_status` | Current pass/fail counts and circuit breaker state |
| `set_gate` | Declare final gate: PASS / PASS_WITH_RISK / FAIL |
| `write_analysis` | Save root-cause analysis to `hermes_analysis.txt` |
| `ssh_command` | Run arbitrary shell command on remote host |
| `scp_fetch` | Pull a file from remote host |
| `scp_push` | Push a local file to remote host |

---

## Outputs

```
out/<run>/
├── summary.json              ← machine-readable gate + per-suite results
├── summary.txt               ← human-readable status report
├── hermes_analysis.txt       ← root-cause analysis written by the agent
├── hermes_session.json       ← session manifest (compat-runtime)
├── hermes_supervisor_prompt.txt  ← full system prompt sent to Hermes
├── hermes_supervisor_output.txt  ← full agent transcript (tool calls + results)
├── events.jsonl              ← stage start/end/heartbeat event stream
├── status-board.json         ← live suite/worker status board
├── remote_command_trace.jsonl    ← SSH/SCP command audit log
└── <suite_name>.log          ← raw output for each suite
```

### Gate values

| Gate | Meaning |
|------|---------|
| `PASS` | All suites passed |
| `PASS_WITH_RISK` | Non-critical failure(s) only |
| `FAIL` | At least one critical suite failed |

Exit code: `0` for PASS/PASS_WITH_RISK, `1` for FAIL.

---

## Fallback mode

If the LLM API is not reachable, `agent.py` automatically falls back to:
1. Running all suites via the compat-runtime engine (same retry/circuit-breaker logic)
2. Calling `hermes chat -q` for post-run root-cause analysis

---

## Architecture

```
agent.py  (single file)
│
├─ Tool functions  (Python)
│   list_suites / run_suite / read_log / get_status
│   set_gate / write_analysis / ssh_command / scp_fetch / scp_push
│
├─ Tool schemas  (OpenAI function-calling JSON)
│
├─ System prompt  (built from session + suite metadata)
│
├─ _run_agent_loop()     ← primary path: Hermes API (OpenAI-compatible)
│   messages[] ──► _chat() ──► tool_calls ──► _dispatch() ──► result
│   └─ repeat until set_gate called or max_iterations reached
│
├─ _run_cli_fallback()   ← when API unreachable
│   run_session() (compat-runtime) + hermes chat -q (RCA)
│
└─ runtime/              (reused as-is)
    contracts.py / engine.py / adapters.py
```
