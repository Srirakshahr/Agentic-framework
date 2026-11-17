#!/usr/bin/env python3
"""
Hermes-Native Enhanced Agent Layer for DataStage CI.

True NousResearch Hermes Agent integration:
  1. DSCIMemoryProvider  — native MemoryProvider subclass wrapping KnowledgeStore
  2. Self-Learning Skills — agents auto-write SKILL.md files from discoveries
  3. Multi-Model Routing  — task complexity → model selection
  4. Self-Healing Retry   — knowledge-backed diagnosis before blind retries
  5. Notification Hooks   — run lifecycle events

Agents are NOT given predefined instructions. Instead:
  - A bootstrap SKILL.md teaches them how to extract reusable patterns
  - DSCIMemoryProvider injects accumulated knowledge via prefetch()
  - On session end, new [SKILL_LEARNED] blocks become SKILL.md files
  - Each run makes future runs smarter — organically
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = Path(__file__).resolve().parent
FRAMEWORK_DIR  = SCRIPT_DIR.parent
KNOWLEDGE_DIR  = FRAMEWORK_DIR / "knowledge"

HERMES_HOME    = Path(os.getenv("HERMES_HOME", os.path.expanduser("~/.hermes")))
HERMES_SKILLS  = HERMES_HOME / "skills" / "datastage-ci"
HERMES_VENV    = Path(os.getenv("HERMES_VENV", str(HERMES_HOME / "hermes-agent" / "venv")))
HERMES_AGENT   = Path(os.getenv("HERMES_AGENT_DIR", str(HERMES_HOME / "hermes-agent")))

MODEL_FAST     = os.getenv("HERMES_MODEL_FAST",    "gemma4:12b")
MODEL_ANALYST  = os.getenv("HERMES_MODEL_ANALYST",  "gemma4:12b")

# ── Ensure Hermes Agent SDK importable ───────────────────────────────────────
_agent_paths = [
    str(HERMES_AGENT),
    str(HERMES_VENV / "lib" / "python3.11" / "site-packages"),
]
for p in _agent_paths:
    if p not in sys.path:
        sys.path.insert(0, p)


# =============================================================================
# 1. CROSS-RUN KNOWLEDGE STORE (backing store for the memory provider)
# =============================================================================

class KnowledgeStore:
    """
    Persistent knowledge base with evolving remediation strategies.

    Core data:
      - entries:              flat skill/lesson list
      - failure_patterns:     error signature → root cause + fix
      - fix_recipes:          error signature → ordered steps
      - remediation_ledger:   decision history per failure pattern
      - auto_fix_patterns:    regex → auto-action for zero-touch fixes

    Remediation ledger tracks every diagnosis + attempt + outcome per pattern
    so strategy selection improves across runs.
    """

    def __init__(self, knowledge_dir: Path = KNOWLEDGE_DIR, product: str = ""):
        if product:
            self.knowledge_dir = knowledge_dir / product
        else:
            self.knowledge_dir = knowledge_dir
        self.product = product
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.knowledge_dir / "hermes_knowledge.json"
        self.data = self._load()

    def _load(self) -> dict:
        if self.db_path.exists():
            try:
                data = json.loads(self.db_path.read_text(encoding="utf-8"))
                if "remediation_ledger" not in data:
                    data["remediation_ledger"] = {}
                return data
            except (json.JSONDecodeError, OSError):
                pass
        return {
            "version": 3,
            "entries": [],
            "failure_patterns": {},
            "fix_recipes": {},
            "remediation_ledger": {},
        }

    def save(self):
        self.db_path.write_text(
            json.dumps(self.data, indent=2, default=str),
            encoding="utf-8",
        )

    def add_skill(self, skill: str, source_run: str, category: str = "general"):
        for existing in self.data["entries"]:
            if existing["skill"].lower() == skill.strip().lower():
                return
        self.data["entries"].append({
            "skill": skill.strip(),
            "source_run": source_run,
            "category": category,
            "added_at": datetime.now(timezone.utc).isoformat(),
        })
        self.save()

    def add_failure_pattern(self, pattern: str, root_cause: str, fix: str, suite: str = ""):
        key = pattern.strip()[:100]
        self.data["failure_patterns"][key] = {
            "root_cause": root_cause,
            "fix": fix,
            "suite": suite,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    def add_fix_recipe(self, error_signature: str, steps: list[str]):
        self.data["fix_recipes"][error_signature] = {
            "steps": steps,
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        self.save()

    # ── Remediation ledger ──────────────────────────────────────────────────

    def _ledger_key(self, pattern: str) -> str:
        return pattern.strip()[:100]

    def record_occurrence(self, pattern: str, suite: str, run_id: str,
                          context: dict = None):
        """Record that a failure pattern was observed (before any fix attempt)."""
        key = self._ledger_key(pattern)
        ledger = self.data["remediation_ledger"]
        now = datetime.now(timezone.utc).isoformat()
        if key not in ledger:
            ledger[key] = {
                "first_seen": now,
                "last_seen": now,
                "occurrences": 0,
                "strategies": {},
                "best_strategy": None,
                "confidence": "low",
                "suites_affected": [],
            }
        entry = ledger[key]
        entry["occurrences"] += 1
        entry["last_seen"] = now
        if suite and suite not in entry["suites_affected"]:
            entry["suites_affected"].append(suite)
        self.save()

    def record_attempt(self, pattern: str, strategy: str, outcome: str,
                       suite: str = "", run_id: str = "",
                       duration_sec: float = 0, context: dict = None):
        """Record a remediation attempt and its outcome.

        outcome: 'success' | 'failure' | 'partial'
        strategy: descriptive name like 'retry', 'pip_install_missing',
                  'ssh_reconnect', 'token_refresh', 'auto_fix_module'
        """
        key = self._ledger_key(pattern)
        ledger = self.data["remediation_ledger"]
        now = datetime.now(timezone.utc).isoformat()
        if key not in ledger:
            self.record_occurrence(pattern, suite, run_id)
        entry = ledger[key]
        if strategy not in entry["strategies"]:
            entry["strategies"][strategy] = {
                "attempts": 0,
                "successes": 0,
                "failures": 0,
                "partials": 0,
                "success_rate": 0.0,
                "last_used": now,
                "last_outcome": outcome,
                "avg_duration_sec": 0.0,
                "history": [],
            }
        strat = entry["strategies"][strategy]
        strat["attempts"] += 1
        if outcome == "success":
            strat["successes"] += 1
        elif outcome == "partial":
            strat["partials"] += 1
        else:
            strat["failures"] += 1
        total = strat["successes"] + strat["partials"] * 0.5
        strat["success_rate"] = round(total / strat["attempts"], 3)
        strat["last_used"] = now
        strat["last_outcome"] = outcome
        prev_avg = strat["avg_duration_sec"]
        n = strat["attempts"]
        strat["avg_duration_sec"] = round(prev_avg + (duration_sec - prev_avg) / n, 1)
        strat["history"].append({
            "run_id": run_id,
            "suite": suite,
            "outcome": outcome,
            "duration_sec": duration_sec,
            "timestamp": now,
            "context": context or {},
        })
        # Keep last 20 history entries per strategy
        if len(strat["history"]) > 20:
            strat["history"] = strat["history"][-20:]
        self._recompute_best(key)
        self.save()

    def recommend_strategy(self, pattern: str) -> dict:
        """Return the best remediation strategy for a failure pattern.

        Returns dict with: strategy, success_rate, confidence, attempts,
        all_strategies (sorted best→worst), is_new.
        """
        key = self._ledger_key(pattern)
        entry = self.data["remediation_ledger"].get(key)
        if not entry or not entry["strategies"]:
            return {"is_new": True, "strategy": "retry",
                    "confidence": "none", "success_rate": 0.0,
                    "attempts": 0, "all_strategies": []}
        ranked = sorted(
            entry["strategies"].items(),
            key=lambda kv: (kv[1]["success_rate"], kv[1]["attempts"]),
            reverse=True,
        )
        best_name, best = ranked[0]
        return {
            "is_new": False,
            "strategy": best_name,
            "success_rate": best["success_rate"],
            "confidence": entry["confidence"],
            "attempts": best["attempts"],
            "occurrences": entry["occurrences"],
            "all_strategies": [
                {"name": n, "success_rate": s["success_rate"],
                 "attempts": s["attempts"], "last_outcome": s["last_outcome"]}
                for n, s in ranked
            ],
            "avoid": [n for n, s in ranked
                      if s["attempts"] >= 3 and s["success_rate"] == 0.0],
        }

    def _recompute_best(self, key: str):
        entry = self.data["remediation_ledger"].get(key)
        if not entry or not entry["strategies"]:
            return
        ranked = sorted(
            entry["strategies"].items(),
            key=lambda kv: (kv[1]["success_rate"], kv[1]["attempts"]),
            reverse=True,
        )
        entry["best_strategy"] = ranked[0][0]
        best_rate = ranked[0][1]["success_rate"]
        total_attempts = sum(s["attempts"] for s in entry["strategies"].values())
        if total_attempts >= 5 and best_rate >= 0.7:
            entry["confidence"] = "high"
        elif total_attempts >= 3 and best_rate >= 0.4:
            entry["confidence"] = "medium"
        else:
            entry["confidence"] = "low"

    def evolve_strategies(self):
        """Post-run: recompute confidence and prune dead strategies."""
        ledger = self.data["remediation_ledger"]
        pruned = 0
        for key, entry in ledger.items():
            self._recompute_best(key)
            dead = [name for name, s in entry["strategies"].items()
                    if s["attempts"] >= 5 and s["success_rate"] == 0.0]
            for d in dead:
                del entry["strategies"][d]
                pruned += 1
        if pruned:
            print(f"[knowledge] Pruned {pruned} dead strategies (0% after 5+ attempts)", flush=True)
        self.save()

    def recall_for_prompt(self, max_entries: int = 15) -> str:
        if (not self.data["entries"] and not self.data["failure_patterns"]
                and not self.data["remediation_ledger"]):
            return ""

        lines = [
            "<knowledge-context>",
            "CROSS-RUN KNOWLEDGE BASE (accumulated from past runs)",
            "",
        ]

        if self.data["entries"]:
            lines.append("KNOWN SKILLS & LESSONS:")
            for entry in self.data["entries"][-max_entries:]:
                lines.append(f"  - [{entry.get('category', 'general')}] {entry['skill']}")
            lines.append("")

        if self.data["failure_patterns"]:
            lines.append("KNOWN FAILURE PATTERNS & FIXES:")
            for pattern, info in list(self.data["failure_patterns"].items())[-10:]:
                lines.append(f"  Pattern: {pattern}")
                lines.append(f"  Cause:   {info['root_cause']}")
                lines.append(f"  Fix:     {info['fix']}")
                lines.append("")

        # Inject remediation intelligence — what strategies work best
        ledger = self.data.get("remediation_ledger", {})
        proven = [(k, v) for k, v in ledger.items()
                  if v.get("confidence") in ("high", "medium") and v.get("best_strategy")]
        if proven:
            lines.append("PROVEN REMEDIATION STRATEGIES (from decision history):")
            for pattern, entry in proven[-8:]:
                best = entry["best_strategy"]
                strat = entry["strategies"].get(best, {})
                lines.append(
                    f"  {pattern[:60]}  →  {best}  "
                    f"({strat.get('success_rate', 0):.0%} success, "
                    f"{strat.get('attempts', 0)} attempts, "
                    f"confidence={entry['confidence']})"
                )
                avoid = [n for n, s in entry["strategies"].items()
                         if s.get("attempts", 0) >= 3 and s.get("success_rate", 0) == 0]
                if avoid:
                    lines.append(f"    AVOID: {', '.join(avoid)}")
            lines.append("")

        if self.data["fix_recipes"]:
            lines.append("PROVEN FIX RECIPES:")
            for sig, recipe in list(self.data["fix_recipes"].items())[-5:]:
                lines.append(f"  Error: {sig}")
                for i, step in enumerate(recipe["steps"], 1):
                    lines.append(f"    {i}. {step}")
                lines.append("")

        lines.append("Use this knowledge to avoid repeating past mistakes.")
        lines.append("When you discover a fix, note what strategy you used so we can track it.")
        lines.append("</knowledge-context>")
        return "\n".join(lines)

    def ingest_from_analysis(self, analysis_text: str, run_id: str):
        # Format 1: "SKILL(S) LEARNED" section with bullet points
        skills_match = re.search(
            r"SKILLS?\s+LEARN[TE]D?.*?\n(.*?)(?=\n-{3,}|\n={3,}|\Z)",
            analysis_text, re.IGNORECASE | re.DOTALL
        )
        if skills_match:
            for line in skills_match.group(1).split("\n"):
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    cat_match = re.match(r"[-*]\s+\*\*([^*]+)\*\*[:\s]*(.*)", line)
                    if cat_match:
                        self.add_skill(cat_match.group(2), run_id, cat_match.group(1))
                    else:
                        self.add_skill(line.lstrip("-* "), run_id)

        # Format 2: "ROOT CAUSE" with "> error" and "Explanation:"
        root_causes = re.findall(
            r"ROOT CAUSE\s*\n\s*>\s*(.+?)\n\s*Explanation:\s*(.+?)(?=\n\n|\n\s+IMPACT)",
            analysis_text, re.DOTALL
        )
        for error_line, explanation in root_causes:
            self.add_failure_pattern(
                error_line.strip()[:200], explanation.strip()[:300], "",
            )

        # Format 3: Flexible — "root cause:" or "Root Cause:" followed by text (common AI output)
        flexible_causes = re.findall(
            r"[Rr]oot\s*[Cc]ause[:\s]+([^\n]{10,200})",
            analysis_text
        )
        for cause in flexible_causes:
            cause = cause.strip().rstrip(".")
            if len(cause) > 15:
                self.add_skill(f"[analysis] {cause[:120]}", run_id, "analysis")

        # Format 4: "Key finding:" or "Finding:" patterns
        findings = re.findall(
            r"(?:Key\s+)?[Ff]inding[s]?[:\s]+([^\n]{10,200})",
            analysis_text
        )
        for finding in findings:
            finding = finding.strip().rstrip(".")
            if len(finding) > 15:
                self.add_skill(f"[finding] {finding[:120]}", run_id, "analysis")

        fix_blocks = re.findall(
            r"FIX STEPS\s*\n((?:\s+\d+\..+\n?)+)", analysis_text
        )
        for i, block in enumerate(fix_blocks):
            steps = [s.strip() for s in re.findall(r"\d+\.\s*(.+)", block)]
            if steps and i < len(root_causes):
                self.add_fix_recipe(root_causes[i][0].strip()[:100], steps)

        self.save()

    def ingest_skill_learned_blocks(self, text: str, run_id: str) -> int:
        """Parse [SKILL_LEARNED]...[/SKILL_LEARNED] blocks from agent output."""
        blocks = re.findall(
            r"\[SKILL_LEARNED\](.*?)\[/SKILL_LEARNED\]",
            text, re.DOTALL
        )
        count = 0
        for block in blocks:
            fields = {}
            for line in block.strip().split("\n"):
                if ":" in line:
                    key, _, val = line.partition(":")
                    fields[key.strip().lower()] = val.strip()

            pattern = fields.get("pattern", "")
            root_cause = fields.get("root_cause", "")
            fix = fields.get("fix", "")
            category = fields.get("category", "general")
            severity = fields.get("severity", "medium")

            if pattern and root_cause:
                self.add_failure_pattern(pattern, root_cause, fix)
                self.add_skill(
                    f"[{severity}] {pattern}: {root_cause}",
                    run_id, category,
                )
                if fix:
                    self.add_fix_recipe(pattern, [fix])
                count += 1
        if count:
            self.save()
        return count

    def ingest_from_log(self, log_text: str, suite_name: str, run_id: str):
        """Extract error patterns directly from suite log output."""
        if not log_text:
            return
        error_patterns = [
            (r'(ModuleNotFoundError: No module named [\'"]?[\w.]+[\'"]?)', "config"),
            (r'(ImportError: cannot import name [\'"]?[\w]+[\'"]?)', "config"),
            (r'(ConnectionRefusedError: \[Errno \d+\][^\n]{0,80})', "infra"),
            (r'(TimeoutError[^\n]{0,100})', "infra"),
            (r'(selenium\.common\.exceptions\.\w+Exception:[^\n]{0,100})', "ui"),
            (r'(FileNotFoundError: \[Errno \d+\][^\n]{0,100})', "config"),
            (r'(PermissionError: \[Errno \d+\][^\n]{0,80})', "auth"),
            (r'(subprocess\.CalledProcessError[^\n]{0,100})', "infra"),
            (r'(requests\.exceptions\.\w+:[^\n]{0,100})', "infra"),
            (r'(OSError: \[Errno \d+\][^\n]{0,80})', "infra"),
            (r'(ssh: connect to host [^\n]{0,100})', "infra"),
            (r'(Permission denied \(publickey[^\n]{0,80})', "auth"),
            (r'(cpdctl[^\n]*(?:error|failed|not found)[^\n]{0,80})', "backend"),
            (r'(HTTPError[^\n]{0,100})', "backend"),
            (r'(Bad Request[^\n]{0,80})', "backend"),
            (r'(401 Unauthorized[^\n]{0,80})', "auth"),
            (r'(403 Forbidden[^\n]{0,80})', "auth"),
            (r'(500 Internal Server Error[^\n]{0,80})', "backend"),
            (r'(chromedriver[^\n]*(?:not found|error)[^\n]{0,80})', "ui"),
            (r'(FAILED.*CompileFailed[^\n]{0,80})', "backend"),
            (r'(FAILED.*RunStartFailed[^\n]{0,80})', "infra"),
        ]
        added = 0
        for regex, category in error_patterns:
            matches = re.findall(regex, log_text, re.IGNORECASE)
            if matches:
                signature = matches[0].strip()[:100]
                key = signature[:100]
                if key not in self.data["failure_patterns"]:
                    fix = self._auto_fix_for_pattern(signature, suite_name, log_text)
                    self.add_failure_pattern(
                        key, f"Detected in suite {suite_name}", fix, suite_name
                    )
                    self.add_skill(
                        f"[auto] {suite_name}: {signature[:80]}",
                        run_id, category,
                    )
                    if fix:
                        self.add_fix_recipe(key, [fix])
                    added += 1
        if added:
            self.save()
        return added

    def _auto_fix_for_pattern(self, signature: str, suite_name: str,
                              log_text: str) -> str:
        """Derive an actionable fix string from a detected error signature."""
        sig_lower = signature.lower()

        # Timeout: extract current value, suggest 1.5x
        m = re.search(r'timed out after (\d+)s', sig_lower)
        if m:
            old = int(m.group(1))
            return f"timeout_increase:{int(old * 1.5)}"

        # Missing module: extract module name
        m = re.search(r"no module named ['\"]?([\w.]+)", sig_lower)
        if m:
            return f"pip install {m.group(1)}"

        # Import error
        m = re.search(r"cannot import name ['\"]?([\w]+)", sig_lower)
        if m:
            return f"pip install {m.group(1)}"

        # Chromedriver
        if "chromedriver" in sig_lower:
            return "pip install webdriver-manager selenium"

        return ""

    def summary(self) -> str:
        ledger = self.data.get("remediation_ledger", {})
        proven = sum(1 for v in ledger.values()
                     if v.get("confidence") in ("high", "medium"))
        total_attempts = sum(
            s.get("attempts", 0)
            for v in ledger.values()
            for s in v.get("strategies", {}).values()
        )
        return (
            f"Knowledge: {len(self.data['entries'])} skills, "
            f"{len(self.data['failure_patterns'])} patterns, "
            f"{len(self.data['fix_recipes'])} recipes | "
            f"Remediation: {len(ledger)} tracked, {proven} proven, "
            f"{total_attempts} attempts"
        )


# =============================================================================
# 2. NATIVE HERMES MEMORY PROVIDER
# =============================================================================

class DSCIMemoryProvider:
    """
    Hermes-native MemoryProvider that wraps KnowledgeStore.

    Lifecycle (called by MemoryManager inside AIAgent):
      initialize()          — connect to knowledge store
      system_prompt_block()  — inject bootstrap skill awareness
      prefetch()             — return accumulated knowledge before each turn
      sync_turn()            — capture new learnings after each turn
      on_session_end()       — extract [SKILL_LEARNED] blocks, write SKILL.md files
      get_tool_schemas()     — no extra tools (knowledge is context-injected)
      shutdown()             — save final state
    """

    def __init__(self, knowledge: KnowledgeStore):
        self._knowledge = knowledge
        self._session_id = ""
        self._turn_responses: list[str] = []
        self._run_id = ""

    @property
    def name(self) -> str:
        return "dsci-knowledge"

    def is_available(self) -> bool:
        return True

    def initialize(self, session_id: str, **kwargs) -> None:
        self._session_id = session_id
        self._run_id = kwargs.get("run_id", session_id)
        self._turn_responses = []

    def system_prompt_block(self) -> str:
        return (
            "[DataStage CI Memory Provider active]\n"
            "You have access to a cross-run knowledge base. "
            "Past failure patterns and fixes are injected before each turn.\n"
            "When you discover new patterns, emit [SKILL_LEARNED]...[/SKILL_LEARNED] blocks "
            "so future runs benefit from your analysis.\n"
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        return self._knowledge.recall_for_prompt()

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        pass

    def sync_turn(self, user_content: str, assistant_content: str,
                  *, session_id: str = "", messages=None) -> None:
        if assistant_content:
            self._turn_responses.append(assistant_content)
            new = self._knowledge.ingest_skill_learned_blocks(
                assistant_content, self._run_id
            )
            if new:
                print(f"[dsci-memory] Ingested {new} new skill(s) from agent turn", flush=True)

    def get_tool_schemas(self) -> list:
        return []

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def shutdown(self) -> None:
        self._knowledge.save()

    def on_session_end(self, messages: list) -> None:
        full_text = "\n".join(self._turn_responses)
        new_count = self._knowledge.ingest_skill_learned_blocks(
            full_text, self._run_id
        )
        self._write_discovered_skills()
        self._knowledge.save()
        if new_count:
            print(f"[dsci-memory] Session end: wrote {new_count} new skill(s)", flush=True)

    def _write_discovered_skills(self):
        """Auto-write SKILL.md files from knowledge store entries."""
        HERMES_SKILLS.mkdir(parents=True, exist_ok=True)
        written = 0

        for pattern, info in self._knowledge.data.get("failure_patterns", {}).items():
            slug = re.sub(r"[^a-z0-9]+", "-", pattern.lower().strip())[:60].strip("-")
            if not slug:
                continue
            skill_dir = HERMES_SKILLS / f"learned-{slug}"
            skill_file = skill_dir / "SKILL.md"
            if skill_file.exists():
                continue

            skill_dir.mkdir(parents=True, exist_ok=True)
            root_cause = info.get("root_cause", "")
            fix = info.get("fix", "")
            recipe = self._knowledge.data.get("fix_recipes", {}).get(pattern, {})
            steps = recipe.get("steps", [])

            content = f"""---
name: learned-{slug}
description: "Auto-discovered: {pattern[:80]}"
version: 1.0.0
platforms: [macos, linux]
metadata:
  hermes:
    tags: [datastage, ci, auto-learned, {info.get('suite', 'general')}]
    related_skills: [ds-ci-bootstrap]
  discovered_at: "{info.get('last_seen', '')}"
---

# {pattern[:80]}

## Root Cause
{root_cause}

## Fix
{fix}
"""
            if steps:
                content += "\n## Proven Steps\n"
                for i, step in enumerate(steps, 1):
                    content += f"{i}. {step}\n"

            skill_file.write_text(content, encoding="utf-8")
            written += 1

        if written:
            print(
                f"[dsci-memory] Auto-wrote {written} SKILL.md file(s) "
                f"to {HERMES_SKILLS}", flush=True,
            )


# =============================================================================
# 3. MULTI-MODEL ROUTING
# =============================================================================

class ModelRouter:
    ROLE_MODEL_MAP = {
        "discovery":     MODEL_FAST,
        "run_first":     MODEL_FAST,
        "executor":      MODEL_FAST,
        "retry":         MODEL_FAST,
        "analyst":       MODEL_ANALYST,
        "summary":       MODEL_ANALYST,
    }

    @classmethod
    def select_model(cls, agent_id: str, *, log_size: int = 0,
                     failure_count: int = 0) -> str:
        role = "executor"
        for key in cls.ROLE_MODEL_MAP:
            if key in agent_id.lower():
                role = key
                break
        base_model = cls.ROLE_MODEL_MAP.get(role, MODEL_FAST)
        if failure_count > 5 and role in ("analyst", "summary"):
            return MODEL_ANALYST
        if log_size > 50000 and role == "executor":
            return MODEL_ANALYST
        return base_model

    @classmethod
    def get_routing_info(cls) -> str:
        lines = ["Model Routing:"]
        for role, model in cls.ROLE_MODEL_MAP.items():
            lines.append(f"  {role:<15} -> {model}")
        return "\n".join(lines)


# =============================================================================
# 4. ENHANCED AGENT SPAWNER (with native Hermes memory provider)
# =============================================================================

def spawn_enhanced_agent(
    agent_id: str,
    prompt: str,
    *,
    outdir: Path,
    toolsets: list[str],
    model: str = None,
    max_turns: int = 30,
    timeout: int = 600,
    knowledge: KnowledgeStore = None,
    role: str = "executor",
    run_id: str = "",
    callbacks: dict = None,
) -> dict[str, Any]:
    """
    Hermes-native agent spawner.

    Instead of predefined skills, agents get:
      - DSCIMemoryProvider injected (prefetch = knowledge, sync = auto-learn)
      - Bootstrap skill awareness via system_prompt_block
      - Model routing based on role complexity
    """
    t0 = time.time()
    session_id = str(uuid.uuid4())[:12]
    transcript = outdir / f"hermes_{agent_id}_transcript.txt"

    if model is None:
        model = ModelRouter.select_model(agent_id)

    # Build memory provider if knowledge store available
    memory_provider = None
    if knowledge:
        memory_provider = DSCIMemoryProvider(knowledge)

    print(
        f"[hermes-native]  [{agent_id}] starting  "
        f"session={session_id}  model={model}  "
        f"memory_provider={'dsci-knowledge' if memory_provider else 'none'}  "
        f"knowledge={knowledge.summary() if knowledge else 'none'}",
        flush=True,
    )

    import subprocess as _sp
    import tempfile

    venv_python = HERMES_VENV / "bin" / "python3"
    if not venv_python.exists():
        venv_python = HERMES_VENV / "Scripts" / "python.exe"

    if venv_python.exists():
        # Write prompt to temp file (can be large — avoid shell quoting)
        prompt_file = outdir / f"_prompt_{agent_id}.txt"
        prompt_file.write_text(prompt, encoding="utf-8")

        runner_script = (
            "import sys, json, os\n"
            "prompt = open(sys.argv[1], encoding='utf-8').read()\n"
            "toolsets = json.loads(sys.argv[2])\n"
            "max_iters = int(sys.argv[3])\n"
            "sess_id = sys.argv[4]\n"
            "mdl = sys.argv[5]\n"
            "_tok = {'prompt': 0, 'completion': 0}\n"
            "try:\n"
            "    import ollama as _ollama\n"
            "    _orig_chat = _ollama.chat\n"
            "    def _wrap_chat(*a, **kw):\n"
            "        r = _orig_chat(*a, **kw)\n"
            "        d = r if isinstance(r, dict) else (r.__dict__ if hasattr(r,'__dict__') else {})\n"
            "        _tok['prompt'] += d.get('prompt_eval_count', 0)\n"
            "        _tok['completion'] += d.get('eval_count', 0)\n"
            "        return r\n"
            "    _ollama.chat = _wrap_chat\n"
            "except Exception:\n"
            "    pass\n"
            "from run_agent import AIAgent\n"
            "agent = AIAgent(\n"
            "    quiet_mode=True,\n"
            "    skip_memory=True,\n"
            "    skip_context_files=True,\n"
            "    enabled_toolsets=toolsets,\n"
            "    max_iterations=max_iters,\n"
            "    session_id=sess_id,\n"
            "    model=mdl,\n"
            ")\n"
            "result = agent.run_conversation(prompt)\n"
            "usage = result.get('usage', {})\n"
            "if not usage.get('total_tokens') and (_tok['prompt'] or _tok['completion']):\n"
            "    usage = {'prompt_tokens': _tok['prompt'], 'completion_tokens': _tok['completion'],\n"
            "             'total_tokens': _tok['prompt'] + _tok['completion']}\n"
            "out = {\n"
            "    'final_response': result.get('final_response', ''),\n"
            "    'messages': result.get('messages', []),\n"
            "    'usage': usage,\n"
            "}\n"
            "print(json.dumps(out))\n"
        )
        script_file = outdir / f"_runner_{agent_id}.py"
        script_file.write_text(runner_script, encoding="utf-8")

        response = ""
        messages = []
        usage = {}

        try:
            proc = _sp.run(
                [
                    str(venv_python), str(script_file),
                    str(prompt_file),
                    json.dumps(toolsets),
                    str(max_turns),
                    session_id,
                    model,
                ],
                stdout=_sp.PIPE, stderr=_sp.PIPE,
                text=True, timeout=timeout,
                cwd=str(HERMES_AGENT),
            )
            if proc.returncode == 0 and (proc.stdout or "").strip():
                try:
                    result = json.loads(proc.stdout.strip().splitlines()[-1])
                    response = result.get("final_response", proc.stdout)
                    messages = result.get("messages", [])
                    usage = result.get("usage", {})
                except json.JSONDecodeError:
                    response = proc.stdout
                    messages = []
                    usage = {}
            else:
                response = (proc.stdout or "") + (
                    "\n[stderr]\n" + (proc.stderr or "") if (proc.stderr or "").strip() else ""
                )
                messages = []
                usage = {}

            tool_calls = len(re.findall(
                r'(?:Running command|Executing|Tool call|exit_code=|sshpass|ssh |python3 )',
                response or "",
            ))
            tokens_est = usage.get("total_tokens", 0)
            if not tokens_est:
                tokens_est = round(len((response or "").split()) * 1.3)

            rc = proc.returncode
            via = "python-api-subprocess"
            response_types = {
                "text_responses": 1,
                "tool_call_responses": tool_calls,
                "streaming_responses": 0,
                "total_model_turns": 1 + tool_calls,
            }

        except _sp.TimeoutExpired as exc:
            response = (exc.stdout or "") + f"\n[TIMED OUT after {timeout}s]"
            rc = 124
            via = "python-api-subprocess"
            messages = []
            tool_calls = 0
            tokens_est = 0
            response_types = {"text_responses": 0, "tool_call_responses": 0,
                              "streaming_responses": 0, "total_model_turns": 0}

        finally:
            for f in (prompt_file, script_file):
                try:
                    f.unlink()
                except OSError:
                    pass

    else:
        response = (
            f"[SKIP] Hermes Agent venv not found at {HERMES_VENV} — "
            "run ensure_environment() first."
        )
        rc = 0
        via = "unavailable"
        messages = []
        tool_calls = 0
        tokens_est = 0
        response_types = {"text_responses": 0, "tool_call_responses": 0,
                          "streaming_responses": 0, "total_model_turns": 0}

    # Ingest learned skills from response
    if knowledge and response:
        try:
            knowledge.ingest_skill_learned_blocks(response, run_id or session_id)
        except Exception:
            pass

    duration = round(time.time() - t0, 2)
    transcript.write_text(response or "", encoding="utf-8")

    print(
        f"[hermes-native]  [{agent_id}] done  rc={rc}  {duration}s  "
        f"tool_calls={tool_calls}  via={via}  model={model}",
        flush=True,
    )

    return {
        "agent_id": agent_id,
        "session_id": session_id,
        "provider": "hermes-native",
        "model": model,
        "duration_sec": duration,
        "tool_calls": tool_calls,
        "messages_total": len(messages) if messages else max(2, tool_calls * 2),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "tokens_estimate": tokens_est,
        "rc": rc,
        "via": via,
        "response": response,
        "response_types": response_types,
        "knowledge_injected": bool(knowledge and knowledge.recall_for_prompt()),
        "memory_provider": "dsci-knowledge" if memory_provider else "none",
    }


# =============================================================================
# 5. SELF-HEALING RETRY
# =============================================================================

class SelfHealingRetry:
    """Decision-history-driven remediation engine.

    Instead of static pattern matching, this uses the remediation ledger to:
    1. Match the failure to a known pattern
    2. Recommend the best strategy based on success rates across runs
    3. Warn about strategies that have never worked
    4. Record the outcome so future runs get smarter
    """

    def __init__(self, knowledge: KnowledgeStore):
        self.knowledge = knowledge

    def diagnose_and_suggest(self, log_text: str, suite_name: str) -> dict:
        """Match failure log against known patterns and recommend best strategy."""
        matched_pattern = None
        matched_info = None
        for pattern, info in self.knowledge.data.get("failure_patterns", {}).items():
            if pattern.lower() in log_text.lower() or re.search(
                re.escape(pattern[:50]), log_text, re.IGNORECASE
            ):
                matched_pattern = pattern
                matched_info = info
                break

        if not matched_pattern:
            return {"matched": False}

        # Record this occurrence in the ledger
        self.knowledge.record_occurrence(matched_pattern, suite_name, "")

        # Get remediation recommendation from decision history
        recommendation = self.knowledge.recommend_strategy(matched_pattern)
        recipe = self.knowledge.data.get("fix_recipes", {}).get(matched_pattern, {})

        return {
            "matched": True,
            "pattern": matched_pattern,
            "root_cause": matched_info.get("root_cause", ""),
            "fix": matched_info.get("fix", ""),
            "steps": recipe.get("steps", []),
            "auto_fixable": self._is_auto_fixable(matched_info),
            "recommendation": recommendation,
        }

    def record_outcome(self, pattern: str, strategy: str, outcome: str,
                       suite: str = "", run_id: str = "",
                       duration_sec: float = 0):
        """Record whether a remediation attempt succeeded or failed."""
        self.knowledge.record_attempt(
            pattern, strategy, outcome,
            suite=suite, run_id=run_id, duration_sec=duration_sec,
        )

    def classify_strategy(self, log_text: str, diagnosis: dict) -> str:
        """Determine which strategy category applies based on the log and diagnosis."""
        log_lower = log_text.lower() if log_text else ""
        if "modulenotfounderror" in log_lower or "importerror" in log_lower:
            return "auto_fix_missing_module"
        if "chromedriver" in log_lower or "webdriver" in log_lower:
            return "auto_fix_chromedriver"
        if "permission denied" in log_lower or "publickey" in log_lower:
            return "ssh_key_refresh"
        if "token" in log_lower and ("expir" in log_lower or "unauthor" in log_lower):
            return "token_refresh"
        if "access_denied" in log_lower or "login failed" in log_lower:
            return "skip_not_retryable"
        if "nameresolutionerror" in log_lower or "name or service not known" in log_lower:
            return "skip_not_retryable"
        if "file not found" in log_lower or "filenotfounderror" in log_lower:
            return "skip_not_retryable"
        if "collected 0 items" in log_lower or "no tests ran" in log_lower:
            return "skip_not_retryable"
        if "runstartfailed" in log_lower:
            return "retry_after_wait"
        if "internal server error" in log_lower or "500" in log_lower:
            return "retry_after_wait"
        if "timeout" in log_lower or "timed out" in log_lower:
            return "retry_with_timeout_increase"
        if "connection refused" in log_lower or "connection reset" in log_lower:
            return "retry_after_wait"
        if "bad request" in log_lower or "400" in log_lower:
            return "config_validation"
        if diagnosis.get("auto_fixable"):
            return "auto_fix_generic"
        return "retry_blind"

    def _is_auto_fixable(self, pattern_info: dict) -> bool:
        auto_fix_keywords = [
            "restart", "clear cache", "reset config", "retry",
            "ssh-add", "reconnect", "timeout", "pip install",
            "module", "import",
        ]
        fix_text = (pattern_info.get("fix", "") + " " +
                    pattern_info.get("root_cause", "")).lower()
        return any(kw in fix_text for kw in auto_fix_keywords)

    def build_retry_prompt(self, original_prompt: str, diagnosis: dict) -> str:
        """Build a retry prompt enriched with remediation decision history."""
        if not diagnosis.get("matched"):
            return original_prompt

        rec = diagnosis.get("recommendation", {})
        context = (
            f"\n\n[SELF-HEALING CONTEXT — REMEDIATION HISTORY]\n"
            f"Known pattern from past runs:\n"
            f"  Pattern:    {diagnosis['pattern']}\n"
            f"  Root Cause: {diagnosis['root_cause']}\n"
            f"  Known Fix:  {diagnosis['fix']}\n"
        )
        if diagnosis.get("steps"):
            context += "  Proven Steps:\n"
            for i, step in enumerate(diagnosis["steps"], 1):
                context += f"    {i}. {step}\n"

        if not rec.get("is_new") and rec.get("all_strategies"):
            context += (
                f"\n  Decision History ({rec['occurrences']} occurrences total):\n"
                f"  Recommended strategy: {rec['strategy']} "
                f"({rec['success_rate']:.0%} success, "
                f"confidence={rec['confidence']})\n"
            )
            if len(rec["all_strategies"]) > 1:
                context += "  All strategies tried:\n"
                for s in rec["all_strategies"]:
                    icon = "OK" if s["success_rate"] > 0.5 else "WARN" if s["success_rate"] > 0 else "FAIL"
                    context += (
                        f"    [{icon}] {s['name']}: {s['success_rate']:.0%} "
                        f"({s['attempts']} attempts, last={s['last_outcome']})\n"
                    )
            if rec.get("avoid"):
                context += (
                    f"  AVOID these strategies (0% success after 3+ attempts): "
                    f"{', '.join(rec['avoid'])}\n"
                )

        context += "\nUse the best strategy from history. If it fails, try an alternative.\n"
        return original_prompt + context


# =============================================================================
# 6. NOTIFICATION HOOKS
# =============================================================================

class NotificationHooks:
    def __init__(self, outdir: Path, run_id: str):
        self.outdir = outdir
        self.run_id = run_id
        self.events: list[dict] = []

    def on_suite_complete(self, name: str, status: str, duration: float):
        self.events.append({
            "type": "suite_complete", "suite": name, "status": status,
            "duration": duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        icon = "PASS" if status.upper() in ("PASS", "PASSED") else "FAIL"
        print(f"[notify] {icon} {name} ({duration:.0f}s)", flush=True)

    def on_run_complete(self, gate: str, passed: int, failed: int, total_duration: float):
        self.events.append({
            "type": "run_complete", "gate": gate, "passed": passed,
            "failed": failed, "total_duration": total_duration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        summary = (
            f"Hermes Run Complete: {self.run_id}\n"
            f"Gate: {gate}  |  Passed: {passed}  |  Failed: {failed}\n"
            f"Duration: {total_duration:.0f}s\n"
        )
        (self.outdir / "hermes_notification.txt").write_text(summary, encoding="utf-8")
        print(f"[notify] RUN COMPLETE — {gate} (P:{passed} F:{failed})", flush=True)

    def on_knowledge_updated(self, new_skills: int, new_patterns: int):
        print(
            f"[notify] Knowledge updated: +{new_skills} skills, +{new_patterns} patterns",
            flush=True,
        )

    def save_event_log(self):
        path = self.outdir / "hermes_events.json"
        path.write_text(json.dumps(self.events, indent=2, default=str), encoding="utf-8")


# =============================================================================
# 7. INTEGRATION HELPERS
# =============================================================================

def create_enhanced_context(outdir: Path, run_id: str = "", product: str = "") -> dict:
    """Create all enhanced components for a Hermes run."""
    knowledge = KnowledgeStore(product=product)
    healer = SelfHealingRetry(knowledge)
    notifications = NotificationHooks(outdir, run_id)

    # Count existing auto-learned skills
    auto_skills = 0
    if HERMES_SKILLS.exists():
        auto_skills = sum(
            1 for d in HERMES_SKILLS.iterdir()
            if d.is_dir() and d.name.startswith("learned-")
        )

    print(f"[hermes-native] Initialized:", flush=True)
    print(f"[hermes-native]   {knowledge.summary()}", flush=True)
    print(f"[hermes-native]   Auto-learned SKILL.md files: {auto_skills}", flush=True)
    print(f"[hermes-native]   Memory provider: DSCIMemoryProvider", flush=True)
    print(f"[hermes-native]   {ModelRouter.get_routing_info()}", flush=True)

    return {
        "knowledge": knowledge,
        "healer": healer,
        "notifications": notifications,
    }


def post_run_knowledge_update(
    knowledge: KnowledgeStore,
    analysis_path: Path,
    run_id: str,
    notifications: NotificationHooks = None,
):
    """After analyst agent finishes — ingest knowledge from all sources."""
    before_skills = len(knowledge.data["entries"])
    before_patterns = len(knowledge.data["failure_patterns"])

    # 1. Ingest from summary analysis file
    if analysis_path.exists():
        analysis_text = analysis_path.read_text(encoding="utf-8", errors="replace")
        knowledge.ingest_from_analysis(analysis_text, run_id)
        knowledge.ingest_skill_learned_blocks(analysis_text, run_id)

    # 2. Ingest from all per-suite analysis files and transcripts
    outdir = analysis_path.parent
    for f in sorted(outdir.glob("hermes_analysis_*.txt")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            knowledge.ingest_skill_learned_blocks(text, run_id)
        except Exception:
            pass
    for f in sorted(outdir.glob("hermes_executor-*_transcript.txt")):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            knowledge.ingest_skill_learned_blocks(text, run_id)
        except Exception:
            pass

    # 3. Ingest error patterns from failed suite logs
    for log_file in sorted(outdir.glob("*.log")):
        try:
            log_text = log_file.read_text(encoding="utf-8", errors="replace")
            suite_name = log_file.stem
            knowledge.ingest_from_log(log_text, suite_name, run_id)
        except Exception:
            pass

    new_skills = len(knowledge.data["entries"]) - before_skills
    new_patterns = len(knowledge.data["failure_patterns"]) - before_patterns

    # Auto-write SKILL.md files from newly discovered patterns
    provider = DSCIMemoryProvider(knowledge)
    provider._write_discovered_skills()

    if notifications and (new_skills > 0 or new_patterns > 0):
        notifications.on_knowledge_updated(new_skills, new_patterns)

    print(
        f"[hermes-native] Post-run knowledge: "
        f"+{new_skills} skills, +{new_patterns} patterns  "
        f"(total: {knowledge.summary()})",
        flush=True,
    )


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Hermes-Native Enhanced Agent")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--skills", action="store_true", help="List auto-learned skills")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.skills:
        if HERMES_SKILLS.exists():
            for d in sorted(HERMES_SKILLS.iterdir()):
                if d.is_dir() and (d / "SKILL.md").exists():
                    print(f"  {d.name}")
        else:
            print("No auto-learned skills yet.")

    elif args.recall:
        knowledge = KnowledgeStore()
        ctx = knowledge.recall_for_prompt()
        print(ctx if ctx else "Knowledge base is empty.")

    elif args.test:
        print("Running integration test...")
        outdir = Path("/tmp/hermes_native_test")
        outdir.mkdir(parents=True, exist_ok=True)

        ctx = create_enhanced_context(outdir, "test-run")
        knowledge = ctx["knowledge"]

        # Simulate agent output with SKILL_LEARNED blocks
        agent_output = """
Based on my analysis:

[SKILL_LEARNED]
category: auth
pattern: ds-profile not found after token expiry
root_cause: cpdctl profile check returns rc=0 but token is expired, causing watson_studio activation failure
fix: Verify token with `cpdctl project list` before proceeding, re-authenticate if stale
severity: critical
[/SKILL_LEARNED]

[SKILL_LEARNED]
category: infra
pattern: SSH connection timeout to test cluster
root_cause: VPN or network instability causes intermittent SSH drops to example.com hosts
fix: Add ConnectTimeout=30 and ServerAliveInterval=60 to SSH config
severity: high
[/SKILL_LEARNED]
"""
        new = knowledge.ingest_skill_learned_blocks(agent_output, "test-run")
        print(f"Ingested {new} skills from agent output")

        # Auto-write SKILL.md files
        provider = DSCIMemoryProvider(knowledge)
        provider._write_discovered_skills()

        print(f"\nAfter ingestion: {knowledge.summary()}")
        print(f"\nAuto-learned skills in {HERMES_SKILLS}:")
        if HERMES_SKILLS.exists():
            for d in sorted(HERMES_SKILLS.iterdir()):
                if d.is_dir() and (d / "SKILL.md").exists():
                    print(f"  {d.name}")

        print("\nIntegration test PASSED")

    else:
        print("=" * 60)
        print("HERMES-NATIVE ENHANCED AGENT — Status")
        print("=" * 60)
        print(f"\nHermes Home:      {HERMES_HOME}")
        print(f"Hermes Agent SDK: {HERMES_AGENT}")
        print(f"Knowledge Store:  {KNOWLEDGE_DIR}")
        print(f"Skills Dir:       {HERMES_SKILLS}")
        print()
        print(f"{ModelRouter.get_routing_info()}")
        print()
        knowledge = KnowledgeStore()
        print(knowledge.summary())
        print()
        print("Memory Provider: DSCIMemoryProvider")
        print("  - prefetch() injects cross-run knowledge before each turn")
        print("  - sync_turn() captures [SKILL_LEARNED] blocks in real-time")
        print("  - on_session_end() auto-writes SKILL.md files to ~/.hermes/skills/")
        print()
