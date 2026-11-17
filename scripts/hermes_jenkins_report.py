#!/usr/bin/env python3
"""
Hermes -> Jenkins report converter.

Reads a Hermes run output directory (single product, or the per-product
sub-directories written by --products) and writes files Jenkins renders
natively, so results and agent telemetry show up on the build page:

  <out>/hermes-junit.xml     JUnit: one testsuite per product, one testcase per
                             Hermes suite. Feeds the `junit` step - Test Result
                             page, per-suite history and the trend graph.
  <out>/hermes-summary.html  Static results + agent telemetry tables. No
                             JavaScript or inline styles, so it renders under
                             Jenkins' default Content-Security-Policy.
  <out>/hermes-summary.txt   One line for currentBuild.description.

Usage:
    python3 hermes_jenkins_report.py <hermes_outdir> [--out <dir>]
"""

import argparse
import html
import json
import re
import sys
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape, quoteattr

try:
    from hermes_security import get_sanitizer
    _sanitize = get_sanitizer().sanitize
except Exception:
    _sanitize = lambda text: text

LOG_TAIL_LINES = 60

# Terminal colour codes (ESC [ ... m) and every character XML 1.0 forbids.
# Suite logs carry colour output, and one ESC (0x1b) in the JUnit file made
# Jenkins reject the whole report - no tests shown at all.
_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[()][0-9A-Za-z]|\x1b[@-_]")
_XML_ILLEGAL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f￾￿]")


def _xml_safe(text: str) -> str:
    return _XML_ILLEGAL.sub("", _ANSI.sub("", str(text)))


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _find_runs(outdir: Path, product: str = "") -> list[tuple[str, Path]]:
    """(product, run_dir) for every directory holding hermes_results.json.

    A single-product run writes straight into outdir, whose name says nothing
    about the product, so the caller passes it in.
    """
    if (outdir / "hermes_results.json").exists():
        return [(product or outdir.name, outdir)]
    return [(p.parent.name, p.parent) for p in sorted(outdir.glob("*/hermes_results.json"))]


def _relpath(target: Path, base: Path) -> str:
    """Link from the summary page to target, or "" when target was not written."""
    if not target.exists():
        return ""
    try:
        return str(target.relative_to(base))
    except ValueError:
        return ""


def _log_tail(result: dict) -> str:
    log_file = result.get("log_file")
    if not log_file or not Path(log_file).exists():
        return ""
    lines = Path(log_file).read_text(encoding="utf-8", errors="replace").splitlines()
    return _xml_safe(_sanitize("\n".join(lines[-LOG_TAIL_LINES:])))


def _junit(runs: list[dict]) -> str:
    out = ['<?xml version="1.0" encoding="UTF-8"?>', "<testsuites>"]
    for run in runs:
        results = run["results"]
        failed = [r for r in results if r.get("status") not in ("passed", "skipped", "error")]
        errors = [r for r in results if r.get("status") == "error"]
        skipped = [r for r in results if r.get("status") == "skipped"]
        total_time = sum(float(r.get("duration_sec") or 0) for r in results)
        out.append(
            f'  <testsuite name={quoteattr("hermes." + run["product"])} tests="{len(results)}" '
            f'failures="{len(failed)}" errors="{len(errors)}" skipped="{len(skipped)}" '
            f'time="{total_time:.1f}">'
        )
        for r in results:
            name = _xml_safe(r.get("name", "unknown"))
            out.append(
                f'    <testcase classname={quoteattr("hermes." + run["product"])} '
                f'name={quoteattr(name)} time="{float(r.get("duration_sec") or 0):.1f}">'
            )
            status = r.get("status")
            msg = f'status={status} rc={r.get("return_code")}'
            if r.get("return_code") == 124:
                msg += " (timeout)"
            if status == "skipped":
                out.append(f"      <skipped message={quoteattr(msg)}/>")
            elif status == "error":
                out.append(f"      <error message={quoteattr(msg)}>{xml_escape(_log_tail(r))}</error>")
            elif status != "passed":
                out.append(f"      <failure message={quoteattr(msg)}>{xml_escape(_log_tail(r))}</failure>")
            if r.get("log_file"):
                out.append(f'      <system-out>{xml_escape("log: " + str(r["log_file"]))}</system-out>')
            out.append("    </testcase>")
        out.append("  </testsuite>")
    out.append("</testsuites>")
    return "\n".join(out) + "\n"


def _table(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f'<table border="1" cellpadding="4" cellspacing="0"><tr>{head}</tr>{body}</table>'


def _summary_html(runs: list[dict]) -> str:
    parts = ["<!doctype html><html><head><meta charset=\"utf-8\"><title>Hermes results</title></head><body>",
             "<h1>Hermes results</h1>"]
    overview = []
    for run in runs:
        m = run["metrics"]
        res = run["results"]
        passed = sum(1 for r in res if r.get("status") == "passed")
        overview.append([run["product"], run["gate"], len(res), passed, len(res) - passed,
                         m.get("total_agents_spawned", ""), m.get("total_tool_calls", ""),
                         m.get("total_tokens_estimate", ""),
                         m.get("timing", {}).get("total_supervisor_sec", "")])
    parts.append("<h2>Overview</h2>")
    parts.append(_table(["Product", "Gate", "Suites", "Passed", "Failed", "Agents",
                         "Tool calls", "Tokens (est.)", "Total (s)"], overview))

    for run in runs:
        m = run["metrics"]
        parts.append(f"<h2>{html.escape(run['product'])} &mdash; gate {html.escape(run['gate'])}</h2>")
        if run["dashboard"]:
            parts.append(f'<p>Full dashboard: <a href="{html.escape(run["dashboard"])}">'
                         f"hermes_dashboard.html</a></p>")
        parts.append("<h3>Suites</h3>")
        parts.append(_table(
            ["Suite", "Status", "rc", "Duration (s)", "Critical"],
            [[r.get("name"), r.get("status"), r.get("return_code"), r.get("duration_sec"),
              r.get("critical", False)] for r in run["results"]]))
        agents = m.get("agents", [])
        if agents:
            parts.append("<h3>Agent telemetry</h3>")
            parts.append(_table(
                ["Agent", "Model", "Via", "Duration (s)", "Tool calls", "Messages", "Tokens (est.)", "rc"],
                [[a.get("agent_id"), a.get("model"), a.get("via"), a.get("duration_sec"),
                  a.get("tool_calls"), a.get("messages_total"), a.get("tokens_estimate"), a.get("rc")]
                 for a in agents]))
        timing = m.get("timing", {})
        if timing:
            parts.append("<h3>Phase timing</h3>")
            parts.append(_table(["Phase", "Seconds"], [[k, v] for k, v in timing.items()]))
        for suite_name, text in run["log_analyses"]:
            parts.append(f"<h3>Log analysis &mdash; {html.escape(suite_name)}</h3>")
            parts.append(f"<pre>{html.escape(text)}</pre>")
        analysis = run["analysis"]
        if analysis:
            parts.append("<h3>Analyst summary</h3>")
            parts.append(f"<pre>{html.escape(analysis)}</pre>")
    parts.append("</body></html>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outdir", help="Hermes --outdir of the run")
    ap.add_argument("--out", help="Where to write the Jenkins files (default: <outdir>)")
    ap.add_argument("--product", default="", help="Product name for a single-product run")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    dest = Path(args.out).resolve() if args.out else outdir
    dest.mkdir(parents=True, exist_ok=True)

    found = _find_runs(outdir, args.product)
    if not found:
        print(f"[hermes-jenkins] no hermes_results.json under {outdir}", file=sys.stderr)
        (dest / "hermes-summary.txt").write_text("Hermes: no results written", encoding="utf-8")
        return 1

    runs = []
    for product, run_dir in found:
        gate_file = run_dir / "hermes_gate.txt"
        analysis_file = run_dir / "hermes_analysis.txt"
        runs.append({
            "product": product,
            "results": _load_json(run_dir / "hermes_results.json", []),
            "metrics": _load_json(run_dir / "hermes_metrics.json", {}),
            "gate": gate_file.read_text(encoding="utf-8").strip() if gate_file.exists() else "UNKNOWN",
            "analysis": _sanitize(analysis_file.read_text(encoding="utf-8", errors="replace"))
                        if analysis_file.exists() else "",
            "dashboard": _relpath(run_dir / "hermes_dashboard.html", dest),
            # Written by the supervisor's log analyser as <suite>.logai.txt
            "log_analyses": [(p.name[:-len(".logai.txt")], _sanitize(p.read_text(encoding="utf-8", errors="replace")))
                             for p in sorted(run_dir.glob("*.logai.txt"))],
        })

    (dest / "hermes-junit.xml").write_text(_junit(runs), encoding="utf-8")
    (dest / "hermes-summary.html").write_text(_summary_html(runs), encoding="utf-8")

    line = " | ".join(
        f"{r['product']}: {r['gate']} "
        f"{sum(1 for x in r['results'] if x.get('status') == 'passed')}/{len(r['results'])}"
        for r in runs)
    (dest / "hermes-summary.txt").write_text(line, encoding="utf-8")
    print(f"[hermes-jenkins] {line}")
    print(f"[hermes-jenkins] wrote {dest}/hermes-junit.xml, hermes-summary.html, hermes-summary.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
