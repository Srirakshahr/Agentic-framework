#!/usr/bin/env python3
"""
Hermes Multi-Agent Test Orchestration Framework — HTML Dashboard Report Generator

Reads from a Hermes run output directory and generates a professional
hermes_dashboard.html report with charts, tables, and analysis.

Usage:
    python3 generate_html_report.py <run_dir>
    python3 generate_html_report.py              # auto-finds latest run in out/
"""

import json
import os
import re
import sys
import glob
from datetime import datetime
from pathlib import Path
from hermes_security import get_sanitizer


# ── Colour palette (IBM Carbon dark) ────────────────────────────────────────
BG       = "#161616"
BG_CARD  = "#262626"
BG_TABLE = "#333333"
TEXT     = "#f4f4f4"
TEXT_DIM = "#c6c6c6"
BLUE     = "#0f62fe"
RED      = "#da1e28"
GREEN    = "#24a148"
YELLOW   = "#f1c21b"
CYAN     = "#1192e8"
PURPLE   = "#8a3ffc"
TEAL     = "#009d9a"
MAGENTA  = "#ee5396"


def find_latest_run(base_dir: str) -> str:
    """Find the most recent run-* directory under out/."""
    out_dir = os.path.join(base_dir, "out")
    if not os.path.isdir(out_dir):
        sys.exit(f"ERROR: out/ directory not found at {out_dir}")
    runs = sorted(glob.glob(os.path.join(out_dir, "run-*")))
    if not runs:
        sys.exit("ERROR: No run-* directories found in out/")
    return runs[-1]


def read_json(path: str):
    """Read a JSON file, return None if missing."""
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def read_text(path: str) -> str:
    """Read a text file, return empty string if missing."""
    if not os.path.isfile(path):
        return ""
    with open(path, "r") as f:
        return f.read()


def extract_skills_learnt(analysis_text: str) -> str:
    """Extract the SKILLS LEARNT section from analysis text."""
    patterns = [
        r"(?:SKILLS?\s+LEARN[TE]D?|LESSONS?\s+LEARN[TE]D?).*?(?=\n(?:={3,}|-{3,}|\#{2,}|\n\S+:|\Z))",
        r"(?<=\n)(#+\s*Skills?\s+Learn[te]d?.*?)(?=\n#+\s|\Z)",
        r"(?:SKILLS?\s+LEARN[TE]D?[:\s]*\n)(.*?)(?=\n\n[A-Z]{2,}|\Z)",
        r"(?:OBSERVATIONS?\s*(?:&|AND)?\s*NEXT\s+STEPS?)[:\s]*\n(.*?)(?=\n\n[A-Z]{2,}|\Z)",
        r"(?<=\n)(#+\s*Observations?\s*(?:&|and)?\s*Next\s+Steps?.*?)(?=\n#+\s|\Z)",
        r"(?<=\n)(#+\s*(?:Key\s+)?Insights?.*?)(?=\n#+\s|\Z)",
    ]
    for pat in patterns:
        m = re.search(pat, analysis_text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(0).strip()
    # Fallback: look for bullet lines after any skills/observations header
    lines = analysis_text.split("\n")
    capture = False
    captured = []
    for line in lines:
        if re.search(r"skills?\s+learn|observations?\s*&?\s*next\s+steps|key\s+insights", line, re.IGNORECASE):
            capture = True
            captured.append(line)
            continue
        if capture:
            if line.strip() == "" and captured and not captured[-1].strip().startswith(("-", "*", "1", "2", "3")):
                break
            if line.strip().startswith(("=", "#")) and len(captured) > 1:
                break
            captured.append(line)
    return "\n".join(captured).strip() if captured else ""


def extract_log_errors(log_text: str) -> dict:
    """Extract error info from a suite log file."""
    lines = log_text.strip().split("\n")
    header = {}
    for line in lines[:5]:
        if line.startswith("suite="):
            header["suite"] = line.split("=", 1)[1]
        elif line.startswith("exit_code="):
            header["exit_code"] = line.split("=", 1)[1]
        elif line.startswith("duration="):
            header["duration"] = line.split("=", 1)[1]

    errors = []
    tracebacks = []
    current_tb = []
    in_traceback = False

    for line in lines:
        low = line.lower()
        if "traceback" in low and "most recent" in low:
            in_traceback = True
            current_tb = [line]
        elif in_traceback:
            current_tb.append(line)
            if line and not line.startswith(" ") and not line.startswith("\t") and "File" not in line:
                tracebacks.append("\n".join(current_tb))
                in_traceback = False
                current_tb = []
        elif any(kw in low for kw in ["[error]", "error:", "failed", "fatal", "permission denied", "timed out"]):
            stripped = line.strip()
            if stripped and len(stripped) > 10 and stripped not in errors:
                errors.append(stripped)

    last_errors = errors[-5:] if errors else []
    last_tb = tracebacks[-2:] if tracebacks else []

    return {
        "header": header,
        "errors": last_errors,
        "tracebacks": last_tb,
    }


def build_deterministic_analysis(run_dir: str, unique_results: list) -> tuple:
    """Build structured analysis and skills from logs deterministically."""
    passed_suites = []
    failed_suites = []

    for r in unique_results:
        name = r.get("name", "unknown")
        status = str(r.get("status", "")).upper()
        rc = r.get("return_code", "N/A")
        dur = r.get("duration_sec", 0)

        log_path = os.path.join(run_dir, f"{name}.log")
        log_text = read_text(log_path)
        info = extract_log_errors(log_text) if log_text else {"header": {}, "errors": [], "tracebacks": []}

        retry_log = os.path.join(run_dir, f"{name}.retry1.log")
        retry_text = read_text(retry_log)
        retry_info = extract_log_errors(retry_text) if retry_text else None

        entry = {
            "name": name, "status": status, "rc": rc, "dur": dur,
            "info": info, "retry_info": retry_info,
        }

        if status in ("PASS", "PASSED", "SUCCESS"):
            passed_suites.append(entry)
        else:
            failed_suites.append(entry)

    # Build analysis text
    lines = []
    lines.append("HERMES FAILURE ANALYSIS REPORT")
    lines.append("=" * 55)
    lines.append("")
    lines.append("EXECUTIVE SUMMARY")
    lines.append(f"  {len(unique_results)} suites executed: "
                 f"{len(passed_suites)} passed, {len(failed_suites)} failed.")
    if failed_suites:
        lines.append(f"  Failed: {', '.join(s['name'] for s in failed_suites)}")
    lines.append("")

    if failed_suites:
        lines.append("-" * 55)
        lines.append("FAILED SUITE DEEP-DIVES")
        lines.append("-" * 55)
        lines.append("")

        for s in failed_suites:
            lines.append(f"## {s['name']}")
            lines.append(f"  Status: FAILED  |  Exit code: {s['rc']}  |  Duration: {s['dur']:.0f}s")
            retried = "Yes" if s["retry_info"] else "No"
            lines.append(f"  Retried: {retried}")
            lines.append("")

            lines.append("  ROOT CAUSE")
            if s["info"]["tracebacks"]:
                for tb in s["info"]["tracebacks"]:
                    for tb_line in tb.split("\n")[-3:]:
                        lines.append(f"  > {tb_line}")
            elif s["info"]["errors"]:
                for err in s["info"]["errors"][:3]:
                    lines.append(f"  > {err}")
            else:
                if s["rc"] == 124:
                    lines.append("  > Timed out (exit code 124)")
                else:
                    lines.append("  > No specific error captured in log")

            if s["retry_info"]:
                lines.append("")
                lines.append("  RETRY RESULT")
                if s["retry_info"]["tracebacks"]:
                    for tb in s["retry_info"]["tracebacks"][-1:]:
                        for tb_line in tb.split("\n")[-2:]:
                            lines.append(f"  > {tb_line}")
                elif s["retry_info"]["errors"]:
                    for err in s["retry_info"]["errors"][:2]:
                        lines.append(f"  > {err}")
                else:
                    rc_retry = s["retry_info"]["header"].get("exit_code", "?")
                    lines.append(f"  > Retry exit code: {rc_retry}")

            lines.append("")

    if passed_suites:
        lines.append("-" * 55)
        lines.append("PASSED SUITES")
        lines.append("-" * 55)
        for s in passed_suites:
            lines.append(f"  {s['name']}: PASSED (exit {s['rc']}, {s['dur']:.0f}s)")
        lines.append("")

    analysis = "\n".join(lines)

    # Build actionable skills from log patterns
    skill_lines = []
    all_errors = []
    for s in failed_suites + passed_suites:
        for err in s["info"]["errors"]:
            all_errors.append((s["name"], s["status"], err))
        for tb in s["info"]["tracebacks"]:
            last_line = tb.strip().split("\n")[-1]
            all_errors.append((s["name"], s["status"], last_line))

    # Detect patterns and generate structured skill cards
    skills_seen = set()
    skill_num = 0

    def add_skill(key, title, learned, action):
        nonlocal skill_num
        if key not in skills_seen:
            skills_seen.add(key)
            skill_num += 1
            skill_lines.append(
                f"SKILL #{skill_num}: {title}\n"
                f"  Learned  : {learned}\n"
                f"  Action   : {action}"
            )

    for suite, status, err in all_errors:
        low = err.lower()

        if "timeoutexception" in low and ("element_to_be_clickable" in low or "selenium" in low):
            add_skill("selenium",
                "Use direct URL navigation instead of UI clicks in CPD",
                "CPD SPA elements exist in DOM but aren't interactive during page transitions — Selenium waits time out.",
                "Replace element clicks with driver.get(base_url + '/zen/#/projectList') for navigation.")

        if "hamburger" in low or "sidemenu" in low:
            add_skill("nav_inside_project",
                "Navigate out of project context via URL, not hamburger menu",
                "After project creation, CPD auto-navigates into the project where the side menu is unclickable.",
                "Use urlparse to extract scheme://host and navigate to /zen/#/projectList directly.")

        if "bad request" in low and "job create" in low:
            add_skill("api_400",
                "RowGen flows need different job creation parameters",
                "cpdctl job create returns 400 for RowGen-type flows — they have different API schema requirements.",
                "Validate flow type before job creation; skip or use alternate params for RowGen flows.")

        if "internal server error" in low:
            add_skill("api_500",
                "Check engine pods before running performance suites",
                "500 errors on job run start mean the DataStage engine backend is unhealthy.",
                "Run 'oc get pods -n ds' to verify engine pods are Running before launching perf suites.")

        if "connection timeout" in low or "db2" in low:
            add_skill("db_conn",
                "Validate database connectivity before import/run tests",
                "DB connection timeouts during jobs mean network/firewall blocks between cluster and DB host.",
                "Add a pre-flight DB ping step in the suite setup phase.")

        if ("arrange" in low or "compile" in low) and "button" in low:
            add_skill("ui_buttons",
                "Wait for canvas readiness before toolbar actions",
                "Arrange/Compile buttons aren't clickable until the DataStage canvas fully renders.",
                "Add WebDriverWait for canvas SVG element before clicking toolbar buttons.")

        if "float" in low and "attribute" in low:
            add_skill("type_bug",
                "Destructure run_command() as (bool, str, float)",
                "run_command() returns execution_time as float in position 3 — not stderr.",
                "Use: ok, output, _ = run_command(...) — never assign position 3 to a string var.")

        if "timed out" in low or "exit code 124" in low.replace("=", " "):
            add_skill("suite_timeout",
                "Increase timeout or reduce batch size for long-running suites",
                "Exit code 124 means the suite was killed by timeout before completing.",
                "Bump timeout_sec in suites.json or split the suite into smaller batches.")

        if "permission denied" in low and ("ssh" in low or "publickey" in low):
            add_skill("ssh_auth",
                "Pre-validate SSH credentials before VM-lane execution",
                "SSH auth fails silently with exit 255 — wastes a suite slot without useful output.",
                "Add 'ssh -o BatchMode=yes host exit' as a pre-flight check in the supervisor.")

        if "read timed out" in low and "localhost" in low:
            add_skill("local_proxy",
                "Add pre-flight port check before Selenium login",
                "Localhost proxy ports (60xxx) hang when the browser process crashes mid-auth.",
                "Run 'lsof -i :PORT' or 'nc -z localhost PORT' before attempting login.")

        if "login error" in low or "login failed" in low:
            add_skill("login_cascade",
                "Verify CPD route and credentials before launching UI suites",
                "Login failure cascades — every subsequent step fails, wasting the full suite duration.",
                "Add a quick curl to the CPD /zen/ endpoint as a pre-flight in suite setup.")

        if "bad request" in low and "job create" not in low:
            add_skill("api_400_general",
                "Validate cpdctl payload schema before API calls",
                "400 errors from cpdctl mean the request body doesn't match the API schema.",
                "Log the full request payload on 400 and validate project_id + flow name exist.")

        if "jsondecodeerror" in low:
            add_skill("json_parse",
                "Handle text/plain error responses from cpdctl",
                "cpdctl returns plain text (not JSON) on errors — json.loads() crashes.",
                "Check response content-type or wrap json.loads() in try/except with raw text logging.")

    # Retry pattern insight
    retried = [s for s in failed_suites if s["retry_info"]]
    if retried:
        fixed_on_retry = [s for s in retried
                          if s["retry_info"]["header"].get("exit_code", "1") == "0"]
        still_failing  = [s for s in retried
                          if s["retry_info"]["header"].get("exit_code", "1") != "0"]
        if fixed_on_retry:
            names = ", ".join(s["name"] for s in fixed_on_retry)
            add_skill("retry_transient",
                "Retry is effective for transient failures",
                f"{len(fixed_on_retry)} suite(s) recovered on retry ({names}) — these are timing/resource issues.",
                "Keep retry enabled for these suites; focus debugging effort on deterministic failures instead.")
        if still_failing:
            names = ", ".join(s["name"] for s in still_failing)
            add_skill("retry_persistent",
                "Persistent failures need code/config fixes, not retries",
                f"{len(still_failing)} suite(s) failed again after retry ({names}).",
                "Investigate root cause in the logs above — retrying won't help these.")

    if not skill_lines:
        skill_lines.append("All suites passed — no new skills to extract from this run.")

    skills = "\n\n".join(skill_lines)
    return analysis, skills


def gate_color(gate: str) -> str:
    g = gate.upper().strip()
    if g == "PASS":
        return GREEN
    elif "RISK" in g:
        return YELLOW
    else:
        return RED


def gate_emoji(gate: str) -> str:
    g = gate.upper().strip()
    if g == "PASS":
        return "&#x2705;"
    elif "RISK" in g:
        return "&#x26A0;&#xFE0F;"
    else:
        return "&#x274C;"


def escape_html(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_html(run_dir: str) -> str:
    """Build the full dashboard HTML string."""

    run_id = os.path.basename(run_dir)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    m = re.search(r"run-(\d{8})-(\d{6})", run_id)
    if m:
        try:
            timestamp = datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    # ── Load data ────────────────────────────────────────────────────────
    metrics = read_json(os.path.join(run_dir, "hermes_metrics.json")) or {}
    results = read_json(os.path.join(run_dir, "hermes_results.json")) or []
    analysis_text = read_text(os.path.join(run_dir, "hermes_analysis.txt"))
    if not analysis_text.strip():
        analysis_text = read_text(os.path.join(run_dir, "hermes_analyst_transcript.txt"))
    gate_text = read_text(os.path.join(run_dir, "hermes_gate.txt")).strip() or metrics.get("gate", "UNKNOWN")

    # ── Deduplicate results by suite name (keep last entry) ──────────────
    seen = {}
    for r in results:
        seen[r.get("name", "unknown")] = r
    unique_results = list(seen.values())

    # ── Detect retries from .retry1.log files ──────────────────────────
    retried_suites = set()
    for f in os.listdir(run_dir):
        m_retry = re.match(r"(.+)\.retry\d+\.log$", f)
        if m_retry:
            retried_suites.add(m_retry.group(1))
    for r in unique_results:
        if r.get("name", "") in retried_suites:
            r["retried"] = True

    total_suites = len(unique_results)
    passed = sum(1 for r in unique_results if str(r.get("status", "")).upper() in ("PASS", "PASSED", "SUCCESS"))
    failed = total_suites - passed
    pass_rate = (passed / total_suites * 100) if total_suites > 0 else 0

    total_duration = metrics.get("timing", {}).get("total_supervisor_sec", sum(r.get("duration_sec", 0) for r in unique_results))

    agents = metrics.get("agents", [])
    total_tokens = metrics.get("total_tokens_estimate", sum(a.get("tokens_estimate", 0) for a in agents))
    model_used = agents[0].get("model", "N/A") if agents else "N/A"

    # Phase timings
    timing = metrics.get("timing", {})

    # ── Log snippets for expandable rows ─────────────────────────────────
    suite_logs = {}
    for r in unique_results:
        log_file = r.get("log_file", "")
        if log_file:
            log_path = log_file if os.path.isabs(log_file) else os.path.join(run_dir, log_file)
            if os.path.isfile(log_path):
                with open(log_path, "r") as f:
                    content = f.read()
                content = get_sanitizer().sanitize(content)
                lines = content.strip().split("\n")
                snippet = "\n".join(lines[-80:]) if len(lines) > 80 else content
                suite_logs[r.get("name", "")] = snippet

    # Build deterministic analysis from logs as primary source
    det_analysis, det_skills = build_deterministic_analysis(run_dir, unique_results)

    # Use agent analysis only if it follows the structured format; otherwise deterministic
    has_structure = any(kw in analysis_text.upper() for kw in ["EXECUTIVE SUMMARY", "DEEP-DIVE", "ROOT CAUSE", "FIX STEPS"])
    if has_structure and len(analysis_text.strip()) > 300:
        # Strip out SKILLS LEARNT and GATE RECOMMENDATION sections — we render those separately
        clean_analysis = analysis_text
        for section_name in ["SKILLS LEARNT", "GATE RECOMMENDATION", "AGENT TELEMETRY"]:
            idx = clean_analysis.upper().find(section_name)
            if idx != -1:
                # Walk back to include the separator line above the heading
                start = idx
                while start > 0 and clean_analysis[start - 1] in "-─ \n":
                    start -= 1
                clean_analysis = clean_analysis[:start].rstrip()
        display_analysis = clean_analysis
    else:
        display_analysis = det_analysis

    skills_text = det_skills

    # ── Chart.js data ────────────────────────────────────────────────────
    suite_names_js = json.dumps([r.get("name", "?") for r in unique_results])
    suite_durations_js = json.dumps([round(r.get("duration_sec", 0), 2) for r in unique_results])
    suite_colors_js = json.dumps([GREEN if str(r.get("status", "")).upper() in ("PASS", "PASSED", "SUCCESS") else RED for r in unique_results])

    # ── Build suite table rows ───────────────────────────────────────────
    suite_rows = ""
    for idx, r in enumerate(unique_results):
        name = r.get("name", "unknown")
        status = str(r.get("status", "UNKNOWN")).upper()
        rc = r.get("return_code", "N/A")
        dur = r.get("duration_sec", 0)
        critical = r.get("critical", False)
        retry = r.get("retry", r.get("retried", False))

        s_color = GREEN if status in ("PASS", "PASSED", "SUCCESS") else RED
        s_badge = f'<span class="badge" style="background:{s_color}">{status}</span>'
        crit_badge = f'<span class="badge" style="background:#f1c21b;color:#161616">CRITICAL</span>' if critical else '<span style="color:#6f6f6f">No</span>'
        retry_badge = '<span class="badge" style="background:#1192e8">RETRIED</span>' if retry else '<span style="color:#6f6f6f">No</span>'

        log_snippet = escape_html(suite_logs.get(name, "No log available."))
        suite_rows += f'''
        <tr class="suite-row" onclick="toggleLog('log-{idx}')">
            <td>{escape_html(name)}</td>
            <td>{s_badge}</td>
            <td><code>{rc}</code></td>
            <td>{dur:.1f}s</td>
            <td>{crit_badge}</td>
            <td>{retry_badge}</td>
        </tr>
        <tr id="log-{idx}" class="log-row" style="display:none">
            <td colspan="6"><pre class="log-block">{log_snippet}</pre></td>
        </tr>'''

    # ── Agent telemetry rows ─────────────────────────────────────────────
    agent_rows = ""
    totals = {"duration": 0, "tool_calls": 0, "prompt": 0, "completion": 0, "total": 0}
    for a in agents:
        aid = a.get("agent_id", "?")
        amodel = a.get("model", "?")
        via = a.get("via", "?")
        adur = a.get("duration_sec", 0)
        tc = a.get("tool_calls", 0)
        pt = a.get("prompt_tokens", 0)
        ct = a.get("completion_tokens", 0)
        tt = a.get("tokens_estimate", 0) or (pt + ct)

        totals["duration"] += adur
        totals["tool_calls"] += tc
        totals["prompt"] += pt
        totals["completion"] += ct
        totals["total"] += tt

        agent_rows += f'''
        <tr>
            <td>{escape_html(str(aid))}</td>
            <td>{escape_html(str(amodel))}</td>
            <td><code>{escape_html(str(via))}</code></td>
            <td>{adur:.1f}s</td>
            <td>{tc}</td>
            <td>{pt:,}</td>
            <td>{ct:,}</td>
            <td><strong>{tt:,}</strong></td>
        </tr>'''

    agent_rows += f'''
    <tr style="background:{BG};font-weight:bold;border-top:2px solid {BLUE}">
        <td colspan="3">TOTAL</td>
        <td>{totals["duration"]:.1f}s</td>
        <td>{totals["tool_calls"]}</td>
        <td>{totals["prompt"]:,}</td>
        <td>{totals["completion"]:,}</td>
        <td>{totals["total"]:,}</td>
    </tr>'''

    # ── Pipeline diagram ─────────────────────────────────────────────────
    pipeline_stages = [
        {"id": "discovery", "label": "Discovery", "match": ["discovery", "discover"]},
        {"id": "run_first", "label": "Run First", "match": ["run_first", "runfirst", "first", "job0"]},
        {"id": "executor_mac", "label": "Executor-MAC", "match": ["executor-mac", "mac"]},
        {"id": "executor_vm", "label": "Executor-VM", "match": ["executor-vm", "vm"]},
        {"id": "retry", "label": "Retry", "match": ["retry"]},
        {"id": "summary", "label": "Summary", "match": ["summary", "analyst", "analysis"]},
    ]

    def agent_for_stage(stage):
        for a in agents:
            aid = a.get("agent_id", "").lower()
            for kw in stage["match"]:
                if kw in aid:
                    return a
        return None

    pipeline_html = ""
    for i, stage in enumerate(pipeline_stages):
        ag = agent_for_stage(stage)
        col = GREEN if ag else BG_TABLE
        is_parallel = stage["id"] in ("executor_mac", "executor_vm")
        pipeline_html += f'''
        <div class="pipeline-node" style="border-color:{col};{'margin-left:40px;' if is_parallel else ''}">
            <div class="pipeline-dot" style="background:{col}"></div>
            <span>{stage["label"]}</span>
        </div>'''
        if i < len(pipeline_stages) - 1 and stage["id"] != "executor_mac":
            if pipeline_stages[i + 1]["id"] == "executor_mac":
                pipeline_html += '<div class="pipeline-arrow">&#x2193; (parallel)</div>'
            else:
                pipeline_html += '<div class="pipeline-arrow">&#x2193;</div>'

    # ── Timing breakdown rows ────────────────────────────────────────────
    timing_rows = ""
    phase_labels = {
        "remote_prep_sec": "Remote Prep",
        "discovery_agent_sec": "Discovery Agent",
        "run_first_sec": "Run First",
        "execution_parallel_sec": "Parallel Execution",
        "retry_sec": "Retry",
        "analyst_agent_sec": "Analyst Agent",
        "total_supervisor_sec": "Total Supervisor",
    }
    phase_colors = [BLUE, CYAN, PURPLE, TEAL, MAGENTA, GREEN, YELLOW]
    max_phase_dur = max((v for k, v in timing.items() if k != "total_supervisor_sec" and isinstance(v, (int, float))), default=1) or 1
    pi = 0
    for pkey, pdur in timing.items():
        if not isinstance(pdur, (int, float)):
            continue
        pname = phase_labels.get(pkey, pkey)
        pct = (pdur / max_phase_dur * 100) if max_phase_dur > 0 and pkey != "total_supervisor_sec" else 100
        pct = min(pct, 100)
        pc = phase_colors[pi % len(phase_colors)]
        pi += 1
        timing_rows += f'''
        <div class="timing-row">
            <span class="timing-label">{escape_html(str(pname))}</span>
            <div class="timing-bar-bg">
                <div class="timing-bar" style="width:{pct}%;background:{pc}"></div>
            </div>
            <span class="timing-value">{pdur:.1f}s</span>
        </div>'''

    # ── Assemble HTML ────────────────────────────────────────────────────
    gate_col = gate_color(gate_text)
    gate_icon = gate_emoji(gate_text)

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
<meta http-equiv="Pragma" content="no-cache">
<meta http-equiv="Expires" content="0">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hermes CI Dashboard — {escape_html(run_id)}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 0;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: {BG}; color: {TEXT};
    line-height: 1.6;
  }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}

  .header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 20px 28px; margin-bottom: 24px;
    background: {BG_CARD}; border-radius: 8px;
    border-left: 4px solid {BLUE};
  }}
  .header h1 {{ margin: 0; font-size: 28px; font-weight: 600; }}
  .header h1 span {{ color: {BLUE}; }}
  .header-meta {{ text-align: right; color: {TEXT_DIM}; font-size: 14px; }}
  .gate-badge {{
    display: inline-block; padding: 6px 18px; border-radius: 20px;
    font-weight: 700; font-size: 15px; letter-spacing: 0.5px;
    color: {BG}; margin-top: 6px;
  }}

  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 28px; }}
  .card {{
    background: {BG_CARD}; border-radius: 8px; padding: 20px;
    text-align: center; border-top: 3px solid {BLUE};
  }}
  .card-value {{ font-size: 32px; font-weight: 700; margin: 4px 0; }}
  .card-label {{ font-size: 13px; color: {TEXT_DIM}; text-transform: uppercase; letter-spacing: 1px; }}

  .section {{
    background: {BG_CARD}; border-radius: 8px; padding: 24px;
    margin-bottom: 24px;
  }}
  .section h2 {{
    margin: 0 0 16px 0; font-size: 20px; font-weight: 600;
    padding-bottom: 8px; border-bottom: 1px solid #393939;
  }}

  .charts-grid {{ display: grid; grid-template-columns: 1fr 2fr; gap: 24px; margin-bottom: 24px; }}
  @media (max-width: 900px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
  .chart-box {{
    background: {BG_CARD}; border-radius: 8px; padding: 24px;
  }}
  .chart-box h2 {{
    margin: 0 0 16px 0; font-size: 18px; font-weight: 600;
    padding-bottom: 8px; border-bottom: 1px solid #393939;
  }}
  canvas {{ max-height: 340px; }}

  table {{ width: 100%; border-collapse: collapse; }}
  th {{
    text-align: left; padding: 10px 14px; font-size: 13px;
    text-transform: uppercase; letter-spacing: 0.8px;
    color: {TEXT_DIM}; background: {BG}; border-bottom: 2px solid #393939;
  }}
  td {{ padding: 10px 14px; border-bottom: 1px solid #393939; font-size: 14px; }}
  .suite-row {{ cursor: pointer; transition: background 0.15s; }}
  .suite-row:hover {{ background: #393939; }}

  .badge {{
    display: inline-block; padding: 3px 12px; border-radius: 12px;
    font-size: 12px; font-weight: 600; letter-spacing: 0.5px;
    color: #fff;
  }}
  code {{ font-family: 'IBM Plex Mono', monospace; font-size: 13px; }}

  .log-block {{
    background: {BG}; color: {TEXT_DIM}; padding: 14px; border-radius: 6px;
    font-size: 12px; max-height: 320px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-all;
    border: 1px solid #393939; margin: 0;
  }}

  .pipeline {{ display: flex; flex-direction: column; align-items: flex-start; gap: 0; }}
  .pipeline-node {{
    display: flex; align-items: center; gap: 10px;
    padding: 10px 20px; border: 2px solid; border-radius: 8px;
    min-width: 180px; font-weight: 600; font-size: 14px;
  }}
  .pipeline-dot {{ width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }}
  .pipeline-arrow {{ color: {TEXT_DIM}; font-size: 18px; padding: 2px 0 2px 24px; }}

  .timing-row {{ display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }}
  .timing-label {{ width: 160px; text-align: right; font-size: 13px; color: {TEXT_DIM}; flex-shrink: 0; }}
  .timing-bar-bg {{ flex: 1; height: 22px; background: #393939; border-radius: 4px; overflow: hidden; }}
  .timing-bar {{ height: 100%; border-radius: 4px; transition: width 0.6s ease; }}
  .timing-value {{ width: 70px; font-size: 13px; font-weight: 600; }}

  .analysis-block {{
    background: {BG}; color: {TEXT_DIM}; padding: 20px; border-radius: 6px;
    font-size: 13px; max-height: 600px; overflow-y: auto;
    white-space: pre-wrap; word-break: break-word;
    border: 1px solid #393939; font-family: 'IBM Plex Mono', monospace;
    line-height: 1.5;
  }}
  .skills-block {{
    background: linear-gradient(135deg, #1a2744, #162616);
    border: 1px solid {GREEN}; border-radius: 8px;
    padding: 20px; font-size: 13px; white-space: pre-wrap;
    font-family: 'IBM Plex Mono', monospace; line-height: 1.5;
    color: #a8d8a8;
  }}

  .footer {{
    text-align: center; padding: 20px; color: #6f6f6f; font-size: 12px;
    border-top: 1px solid #393939; margin-top: 32px;
  }}
</style>
</head>
<body>
<div class="container">

  <div class="header">
    <div>
      <h1><span>Hermes</span> CI Dashboard</h1>
      <div style="color:{TEXT_DIM};font-size:14px;margin-top:4px">Multi-Agent Test Orchestration Framework</div>
    </div>
    <div class="header-meta">
      <div><strong>Run:</strong> {escape_html(run_id)}</div>
      <div><strong>Time:</strong> {escape_html(timestamp)}</div>
      <div style="margin-top:6px">
        <span class="gate-badge" style="background:{gate_col}">{gate_icon} {escape_html(gate_text)}</span>
      </div>
    </div>
  </div>

  <div class="cards">
    <div class="card">
      <div class="card-label">Total Suites</div>
      <div class="card-value" style="color:{BLUE}">{total_suites}</div>
    </div>
    <div class="card" style="border-top-color:{GREEN}">
      <div class="card-label">Passed</div>
      <div class="card-value" style="color:{GREEN}">{passed}</div>
    </div>
    <div class="card" style="border-top-color:{RED}">
      <div class="card-label">Failed</div>
      <div class="card-value" style="color:{RED}">{failed}</div>
    </div>
    <div class="card" style="border-top-color:{'#24a148' if pass_rate >= 80 else '#f1c21b' if pass_rate >= 50 else '#da1e28'}">
      <div class="card-label">Pass Rate</div>
      <div class="card-value">{pass_rate:.0f}%</div>
    </div>
    <div class="card" style="border-top-color:{CYAN}">
      <div class="card-label">Total Duration</div>
      <div class="card-value" style="font-size:24px">{total_duration:.0f}s</div>
    </div>
    <div class="card" style="border-top-color:{PURPLE}">
      <div class="card-label">Total Tokens</div>
      <div class="card-value" style="font-size:24px">{total_tokens:,}</div>
    </div>
    <div class="card" style="border-top-color:{TEAL}">
      <div class="card-label">Model</div>
      <div class="card-value" style="font-size:18px">{escape_html(str(model_used))}</div>
    </div>
  </div>

  <div class="charts-grid">
    <div class="chart-box">
      <h2>Pass / Fail Distribution</h2>
      <canvas id="pieChart"></canvas>
    </div>
    <div class="chart-box">
      <h2>Suite Duration (seconds)</h2>
      <canvas id="barChart"></canvas>
    </div>
  </div>

  <div class="section">
    <h2>Suite Results</h2>
    <div style="font-size:12px;color:{TEXT_DIM};margin-bottom:10px">Click a row to expand its log.</div>
    <table>
      <thead>
        <tr>
          <th>Suite Name</th><th>Status</th><th>Exit Code</th>
          <th>Duration</th><th>Critical</th><th>Retry</th>
        </tr>
      </thead>
      <tbody>{suite_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Agent Telemetry</h2>
    <table>
      <thead>
        <tr>
          <th>Agent ID</th><th>Model</th><th>Via</th>
          <th>Duration</th><th>Tool Calls</th><th>Prompt Tokens</th>
          <th>Completion Tokens</th><th>Total Tokens</th>
        </tr>
      </thead>
      <tbody>{agent_rows}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Agent Pipeline Flow</h2>
    <div class="pipeline">
      {pipeline_html}
    </div>
  </div>

  <div class="section">
    <h2>Timing Breakdown</h2>
    {timing_rows if timing_rows else '<div style="color:#6f6f6f">No phase timing data available.</div>'}
  </div>

  <div class="section">
    <h2>Failure Analysis Report</h2>
    <pre class="analysis-block">{escape_html(display_analysis) if display_analysis else "No analysis report available."}</pre>
  </div>

  <div class="section">
    <h2>Skills Learnt</h2>
    <div class="skills-block">{escape_html(skills_text) if skills_text else "No skills learnt section found in analysis."}</div>
  </div>

  <div class="footer">
    Generated by <strong>Hermes Dashboard Generator</strong> &mdash; {escape_html(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}
  </div>
</div>

<script>
document.addEventListener("DOMContentLoaded", function() {{
  const pieCtx = document.getElementById("pieChart").getContext("2d");
  new Chart(pieCtx, {{
    type: "doughnut",
    data: {{
      labels: ["Passed", "Failed"],
      datasets: [{{
        data: [{passed}, {failed}],
        backgroundColor: ["{GREEN}", "{RED}"],
        borderColor: ["{BG_CARD}", "{BG_CARD}"],
        borderWidth: 3
      }}]
    }},
    options: {{
      responsive: true,
      plugins: {{
        legend: {{ labels: {{ color: "{TEXT}", font: {{ size: 14 }} }} }},
        tooltip: {{ bodyFont: {{ size: 14 }} }}
      }},
      cutout: "55%"
    }}
  }});

  const barCtx = document.getElementById("barChart").getContext("2d");
  new Chart(barCtx, {{
    type: "bar",
    data: {{
      labels: {suite_names_js},
      datasets: [{{
        label: "Duration (s)",
        data: {suite_durations_js},
        backgroundColor: {suite_colors_js},
        borderRadius: 4,
        barThickness: 28
      }}]
    }},
    options: {{
      indexAxis: "y",
      responsive: true,
      plugins: {{
        legend: {{ display: false }},
        tooltip: {{ bodyFont: {{ size: 13 }} }}
      }},
      scales: {{
        x: {{
          grid: {{ color: "#393939" }},
          ticks: {{ color: "{TEXT_DIM}", font: {{ size: 12 }} }},
          title: {{ display: true, text: "Seconds", color: "{TEXT_DIM}" }}
        }},
        y: {{
          grid: {{ display: false }},
          ticks: {{ color: "{TEXT}", font: {{ size: 12 }} }}
        }}
      }}
    }}
  }});
}});

function toggleLog(id) {{
  const el = document.getElementById(id);
  if (el) el.style.display = el.style.display === "none" ? "table-row" : "none";
}}
</script>
</body>
</html>'''

    return html


def main():
    if len(sys.argv) > 1:
        run_dir = sys.argv[1]
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        run_dir = find_latest_run(script_dir)

    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        sys.exit(f"ERROR: Run directory not found: {run_dir}")

    print(f"[Hermes] Reading run data from: {run_dir}")

    html = build_html(run_dir)

    out_path = os.path.join(run_dir, "hermes_dashboard.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"[Hermes] Dashboard written to: {out_path}")
    print(f"[Hermes] Open in browser: file://{out_path}")


if __name__ == "__main__":
    main()
