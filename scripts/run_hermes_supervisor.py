#!/usr/bin/env python3
"""
Hermes Supervisor — DataStage test orchestration (6-Agent Architecture).

Single entry point: run this script and it handles everything — dependency
installation, Ollama setup, model download, Hermes Agent setup, and the
full multi-agent orchestration workflow.

Agents:
  1. DISCOVERY    — classifies suites, writes suite_inventory / bootstrap / task_plan JSON
  2. RUN_FIRST    — SSH gate: runs critical suite(s), stops run on critical failure
  3. EXECUTOR-MAC — runs mac_lane suites locally via Hermes terminal tool  (parallel with vm)
  4. EXECUTOR-VM  — runs vm_lane suites via SSH via Hermes terminal tool   (parallel with mac)
  5. RETRY        — reads failure logs, re-runs all failed suites in parallel
  6. ANALYST      — reads all logs, writes hermes_analysis.txt

All agents use the AIAgent Python API (NousResearch Hermes Agent).
If the Python API is unavailable, agents fall back to `hermes chat` CLI subprocess.
Per-role model config: HERMES_MODEL_FAST (exec/retry) and HERMES_MODEL_ANALYST.

Usage:
  python3 run_hermes_supervisor.py \\
    --remote-host api.cluster.example.com \\
    --remote-user root --ssh-password 'pass2ibm' \\
    --cluster clustername \\
    --cpd-url 'https://cpd-ds.apps.cluster.example.com' \\
    --cpd-password 'secret' \\
    --config ../configs/suites.json \\
    --max-workers 4 --max-retries 1

Local-only (no VM):
  python3 run_hermes_supervisor.py --config ../configs/suites.json

Skip auto-setup (if already installed):
  HERMES_SKIP_SETUP=1 python3 run_hermes_supervisor.py --config ../configs/suites.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from hermes_security import get_sanitizer

SCRIPT_DIR    = Path(__file__).resolve().parent
FRAMEWORK_DIR = SCRIPT_DIR.parent
PROMPTS_DIR   = FRAMEWORK_DIR / "prompts"
HERMES_INSTALL = Path.home() / ".hermes" / "hermes-agent"

# Per-role model config — change here when you switch models
def _init_model_config():
    """Deferred so .env is loaded first."""
    global HERMES_MODEL_FAST, HERMES_MODEL_ANALYST, HERMES_MODEL
    HERMES_MODEL_FAST     = os.getenv("HERMES_MODEL_FAST", "gemma4:12b")
    HERMES_MODEL_ANALYST  = os.getenv("HERMES_MODEL_ANALYST", "gemma4:12b")
    HERMES_MODEL          = HERMES_MODEL_FAST

HERMES_MODEL_FAST     = os.getenv("HERMES_MODEL_FAST", "gemma4:12b")
HERMES_MODEL_ANALYST  = os.getenv("HERMES_MODEL_ANALYST", "gemma4:12b")
HERMES_MODEL          = HERMES_MODEL_FAST
HERMES_TIMEOUT        = 7200           # seconds — wall-clock max per agent
OLLAMA_PARALLEL       = int(os.getenv("OLLAMA_NUM_PARALLEL", "4"))


# =============================================================================
# AUTO-SETUP: Ensure all prerequisites are installed before execution
# =============================================================================

def _log_setup(msg: str, level: str = "INFO"):
    symbol = {"INFO": "•", "OK": "✓", "WARN": "⚠", "FAIL": "✗", "STEP": "═"}
    print(f"  [{symbol.get(level, '•')}] {msg}", flush=True)


def _run_cmd(cmd: str, timeout: int = 300, capture: bool = True) -> tuple[int, str]:
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=capture, text=True, timeout=timeout,
        )
        return r.returncode, (r.stdout + r.stderr).strip() if capture else ""
    except subprocess.TimeoutExpired:
        return 1, "command timed out"
    except Exception as e:
        return 1, str(e)


def _detect_platform() -> dict:
    system = platform.system().lower()
    machine = platform.machine().lower()
    is_mac = system == "darwin"
    is_linux = system == "linux"
    is_windows = system == "windows"

    pkg_manager = None
    if is_linux:
        for pm in ["dnf", "yum", "apt-get"]:
            if shutil.which(pm):
                pkg_manager = pm
                break

    return {
        "system": system, "machine": machine,
        "is_mac": is_mac, "is_linux": is_linux, "is_windows": is_windows,
        "pkg_manager": pkg_manager,
    }


_BEST_PYTHON = sys.executable  # upgraded to 3.10+ if found/installed

def _find_best_python() -> str:
    """Find the highest-version Python 3.10+ binary available."""
    for candidate in ["python3.13", "python3.12", "python3.11", "python3.10"]:
        path = shutil.which(candidate)
        if path:
            rc, ver = _run_cmd(f"{path} -c \"import sys; print(sys.version_info.minor)\"")
            if rc == 0 and int(ver.strip()) >= 10:
                return path
    # Check if current python is already 3.10+
    if sys.version_info >= (3, 10):
        return sys.executable
    return ""


def _install_python311(plat: dict) -> str:
    """Install Python 3.11 and return its path."""
    if plat["is_linux"] and plat["pkg_manager"]:
        pm = plat["pkg_manager"]
        if pm in ("dnf", "yum"):
            _run_cmd(f"{pm} install -y python3.11 python3.11-pip python3.11-devel 2>/dev/null", timeout=120)
        elif pm == "apt-get":
            _run_cmd("apt-get update -qq && apt-get install -y python3.11 python3.11-pip python3.11-venv python3.11-dev 2>/dev/null", timeout=120)
    elif plat["is_mac"] and shutil.which("brew"):
        _run_cmd("brew install python@3.11", timeout=120)

    return _find_best_python()


def _check_python() -> bool:
    global _BEST_PYTHON
    v = sys.version_info
    if v.major < 3 or v.minor < 9:
        _log_setup(f"Python {v.major}.{v.minor} found — 3.9+ required", "FAIL")
        return False

    # Hermes Agent needs 3.10+ for type union syntax (X | Y)
    best = _find_best_python()
    if best:
        _BEST_PYTHON = best
        rc, ver = _run_cmd(f"{best} --version")
        _log_setup(f"Python for Hermes Agent: {ver.strip()} ({best})", "OK")
        return True

    _log_setup(f"Python {v.major}.{v.minor} — Hermes Agent needs 3.10+, installing 3.11...", "WARN")
    plat = _detect_platform()
    best = _install_python311(plat)
    if best:
        _BEST_PYTHON = best
        rc, ver = _run_cmd(f"{best} --version")
        _log_setup(f"Python 3.11 installed: {ver.strip()} ({best})", "OK")
        return True

    _log_setup(f"Python {v.major}.{v.minor} OK for supervisor, but Hermes Agent may use CLI fallback (needs 3.10+)", "WARN")
    _BEST_PYTHON = sys.executable
    return True


def _check_pip() -> bool:
    rc, _ = _run_cmd(f"{sys.executable} -m pip --version")
    if rc == 0:
        _log_setup("pip available", "OK")
        return True
    _log_setup("pip not found — installing...", "WARN")
    rc, out = _run_cmd("curl -sS https://bootstrap.pypa.io/get-pip.py | " + sys.executable)
    if rc == 0:
        _log_setup("pip installed", "OK")
        return True
    _log_setup(f"pip installation failed: {out[:200]}", "FAIL")
    return False


def _install_system_packages(plat: dict) -> bool:
    needed = []
    for cmd in ["git", "curl", "sshpass", "tmux"]:
        if not shutil.which(cmd):
            needed.append(cmd)
    if not needed:
        _log_setup("System packages (git, curl, sshpass, tmux)", "OK")
        return True

    _log_setup(f"Missing: {', '.join(needed)} — installing...", "WARN")
    if plat["is_mac"]:
        if shutil.which("brew"):
            rc, out = _run_cmd(f"brew install {' '.join(needed)}")
        else:
            _log_setup("Homebrew not found. Install missing packages manually: " + ", ".join(needed), "FAIL")
            return False
    elif plat["is_linux"] and plat["pkg_manager"]:
        pkg_map = {"apt-get": "apt-get install -y", "dnf": "dnf install -y", "yum": "yum install -y"}
        install_cmd = pkg_map[plat["pkg_manager"]]
        # sshpass may need EPEL on RHEL
        if plat["pkg_manager"] in ("dnf", "yum"):
            _run_cmd(f"{plat['pkg_manager']} install -y epel-release 2>/dev/null", timeout=120)
            pkg_names = " ".join(needed).replace("tmux", "tmux procps-ng")
        else:
            pkg_names = " ".join(needed)
        rc, out = _run_cmd(f"{install_cmd} {pkg_names}", timeout=120)
    else:
        _log_setup(f"Cannot auto-install on {plat['system']}. Install manually: {', '.join(needed)}", "FAIL")
        return False

    still_missing = [c for c in needed if not shutil.which(c)]
    if still_missing:
        _log_setup(f"Still missing after install: {', '.join(still_missing)}", "FAIL")
        return False
    _log_setup("System packages installed", "OK")
    return True


def _check_ollama(plat: dict) -> bool:
    if shutil.which("ollama"):
        _log_setup("Ollama binary found", "OK")
        return True

    _log_setup("Ollama not found — installing...", "WARN")
    if plat["is_mac"] or plat["is_linux"]:
        rc, out = _run_cmd("curl -fsSL https://ollama.com/install.sh | sh", timeout=300)
        if rc == 0 and shutil.which("ollama"):
            _log_setup("Ollama installed", "OK")
            return True
        # macOS: check if installed via app bundle
        ollama_app = Path("/usr/local/bin/ollama")
        if not ollama_app.exists():
            ollama_app = Path("/Applications/Ollama.app/Contents/Resources/ollama")
        if ollama_app.exists():
            os.environ["PATH"] = str(ollama_app.parent) + ":" + os.environ.get("PATH", "")
            _log_setup(f"Ollama found at {ollama_app}", "OK")
            return True
    elif plat["is_windows"]:
        _log_setup("Download Ollama from https://ollama.com/download/windows", "FAIL")
        return False

    _log_setup(f"Ollama installation failed: {out[:200]}", "FAIL")
    return False


def _start_ollama() -> bool:
    rc, _ = _run_cmd("curl -sf --max-time 3 http://127.0.0.1:11434/api/tags")
    if rc == 0:
        _log_setup("Ollama server running", "OK")
        return True

    _log_setup("Ollama server not running — starting...", "WARN")
    env = os.environ.copy()
    env["OLLAMA_NUM_PARALLEL"] = str(OLLAMA_PARALLEL)
    env["OLLAMA_HOST"] = "0.0.0.0"
    try:
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=open("/tmp/ollama_hermes.log", "a"),
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    except Exception as e:
        _log_setup(f"Failed to start Ollama: {e}", "FAIL")
        return False

    for attempt in range(15):
        time.sleep(2)
        rc, _ = _run_cmd("curl -sf --max-time 3 http://127.0.0.1:11434/api/tags")
        if rc == 0:
            _log_setup(f"Ollama server started (parallel={OLLAMA_PARALLEL})", "OK")
            return True

    _log_setup("Ollama server failed to start — check /tmp/ollama_hermes.log", "FAIL")
    return False


def _check_model(model_name: str) -> bool:
    base_name = model_name.split(":")[0]
    rc, out = _run_cmd("ollama list")
    if rc == 0 and base_name in out:
        _log_setup(f"Model {model_name} available", "OK")
        return True

    _log_setup(f"Model {model_name} not found — pulling (this may take several minutes)...", "WARN")
    rc, out = _run_cmd(f"ollama pull {model_name}", timeout=600)
    if rc == 0:
        _log_setup(f"Model {model_name} pulled", "OK")
        return True

    _log_setup(f"Failed to pull {model_name}: {out[:200]}", "FAIL")
    return False


def _verify_model(model_name: str) -> bool:
    rc, out = _run_cmd(f'ollama run {model_name} "respond with only: OK"', timeout=120)
    if rc == 0 and out.strip():
        _log_setup(f"Model inference test passed", "OK")
        return True
    _log_setup(f"Model inference test failed — may work during execution", "WARN")
    return True  # non-fatal


def _setup_hermes_agent() -> bool:
    hermes_home = Path.home() / ".hermes"
    agent_dir = hermes_home / "hermes-agent"
    run_agent = agent_dir / "run_agent.py"
    venv_python = agent_dir / "venv" / "bin" / "python3"
    need_venv = False

    # ── 1. Check existing install ────────────────────────────────────────
    if run_agent.exists() and venv_python.exists():
        rc, ver = _run_cmd(f"{venv_python} -c \"import sys; print(sys.version_info.minor)\"")
        if rc == 0 and int(ver.strip()) >= 11:
            _log_setup("Hermes Agent installed (Python 3.11+ venv)", "OK")
        else:
            _log_setup(f"Hermes Agent venv is Python 3.{ver.strip()} — needs 3.11+, rebuilding...", "WARN")
            _run_cmd(f"rm -rf {agent_dir / 'venv'}")
            need_venv = True
    elif run_agent.exists():
        _log_setup("Hermes Agent found but venv missing — recreating...", "WARN")
        need_venv = True
    else:
        # ── 2. Clone repo ────────────────────────────────────────────────
        _log_setup("Hermes Agent not found — cloning from GitHub...", "INFO")
        hermes_home.mkdir(parents=True, exist_ok=True)
        if agent_dir.exists():
            rc, _ = _run_cmd(f"cd {agent_dir} && git pull", timeout=60)
        else:
            rc, out = _run_cmd(
                f"git clone https://github.com/NousResearch/hermes-agent.git {agent_dir}",
                timeout=180,
            )
            if rc != 0:
                _log_setup(f"Git clone failed: {out[:200]}", "FAIL")
                return True
        need_venv = True

    # ── 3. Create venv + install deps (only if needed) ───────────────────
    if need_venv:
        _log_setup(f"Creating Hermes Agent virtualenv with {_BEST_PYTHON}...", "INFO")
        rc, out = _run_cmd(f"{_BEST_PYTHON} -m venv {agent_dir / 'venv'}")
        if rc != 0:
            _log_setup(f"venv creation failed: {out[:200]}", "FAIL")
            return True

        pip_bin = agent_dir / "venv" / "bin" / "pip"
        if not pip_bin.exists():
            pip_bin = agent_dir / "venv" / "Scripts" / "pip"

        pyproject = agent_dir / "pyproject.toml"
        reqs = agent_dir / "requirements.txt"
        if pyproject.exists():
            rc, out = _run_cmd(f"{pip_bin} install --upgrade pip && {pip_bin} install '{agent_dir}'", timeout=600)
        elif reqs.exists():
            rc, out = _run_cmd(f"{pip_bin} install --upgrade pip && {pip_bin} install -r {reqs}", timeout=300)
        else:
            rc, out = _run_cmd(f"{pip_bin} install --upgrade pip && {pip_bin} install pydantic ollama rich", timeout=300)
        if rc != 0:
            _log_setup(f"Dependency install failed: {out[:200]}", "FAIL")
            return True
        _log_setup("Hermes Agent venv + deps installed", "OK")

    # ── 4. Verify AIAgent import (install deps if missing) ────────────────
    rc, out = _run_cmd(
        f"cd {agent_dir} && {venv_python} -c \""
        f"from run_agent import AIAgent; print('IMPORT_OK')\"",
        timeout=30,
    )
    if "IMPORT_OK" not in (out or ""):
        _log_setup("AIAgent import failed — installing full dependencies...", "WARN")
        pip_bin = agent_dir / "venv" / "bin" / "pip"
        if not pip_bin.exists():
            pip_bin = agent_dir / "venv" / "Scripts" / "pip"
        pyproject = agent_dir / "pyproject.toml"
        if pyproject.exists():
            rc2, out2 = _run_cmd(f"{pip_bin} install --upgrade pip && {pip_bin} install '{agent_dir}'", timeout=600)
        else:
            rc2, out2 = _run_cmd(f"{pip_bin} install --upgrade pip && {pip_bin} install pydantic ollama rich pyyaml openai httpx requests prompt_toolkit", timeout=300)
        if rc2 != 0:
            _log_setup(f"Dependency install failed: {out2.strip()[:300]}", "FAIL")
            return True
        # Retry import
        rc, out = _run_cmd(
            f"cd {agent_dir} && {venv_python} -c \""
            f"from run_agent import AIAgent; print('IMPORT_OK')\"",
            timeout=30,
        )
        if "IMPORT_OK" not in (out or ""):
            _log_setup("AIAgent import still failing — full error:", "FAIL")
            for err_line in (out or "").strip().splitlines()[-10:]:
                _log_setup(f"  {err_line}", "FAIL")
            return True
    _log_setup("AIAgent import verified", "OK")

    # ── 5. Ensure Ollama installed ───────────────────────────────────────
    if not shutil.which("ollama"):
        _log_setup("Ollama not found — installing...", "INFO")
        plat = _detect_platform()
        if plat.get("is_linux"):
            rc, out = _run_cmd("curl -fsSL https://ollama.com/install.sh | sh", timeout=300)
        elif plat.get("is_mac") and shutil.which("brew"):
            rc, out = _run_cmd("brew install ollama", timeout=300)
        else:
            _log_setup("Install Ollama manually: https://ollama.com/download", "WARN")
            return True
        if not shutil.which("ollama"):
            _log_setup(f"Ollama install failed: {out[:200]}", "FAIL")
            return True
        _log_setup("Ollama installed", "OK")

    # ── 6. Start Ollama server if not running ────────────────────────────
    rc_ping, _ = _run_cmd("curl -sf http://localhost:11434/api/tags", timeout=5)
    if rc_ping != 0:
        _log_setup("Starting Ollama server...", "INFO")
        _run_cmd("nohup ollama serve > /tmp/ollama_hermes.log 2>&1 &", timeout=5)
        time.sleep(4)
        rc_ping2, _ = _run_cmd("curl -sf http://localhost:11434/api/tags", timeout=5)
        if rc_ping2 != 0:
            _log_setup("Ollama server failed to start — check /tmp/ollama_hermes.log", "WARN")
            return True
    _log_setup("Ollama server running", "OK")

    # ── 7. Pull model if not present ─────────────────────────────────────
    model_name = HERMES_MODEL_FAST
    model_base = model_name.split(":")[0]
    rc_list, model_list = _run_cmd("ollama list", timeout=10)
    if model_base not in (model_list or ""):
        _log_setup(f"Pulling model {model_name} (this may take several minutes)...", "INFO")
        rc, out = _run_cmd(f"ollama pull {model_name}", timeout=900)
        if rc != 0:
            _log_setup(f"Model pull failed: {out[:200]}", "FAIL")
            return True
        _log_setup(f"Model {model_name} pulled", "OK")
    else:
        _log_setup(f"Model {model_name} available", "OK")

    # ── 8. Create hermes config.yaml for Ollama provider ───────────────
    config_file = hermes_home / "config.yaml"
    if not config_file.exists():
        config_content = (
            "model:\n"
            "  api_key: ollama\n"
            "  base_url: http://127.0.0.1:11434/v1\n"
            "  context_length: 65536\n"
            f"  default: {model_name}\n"
            "  ollama_num_ctx: 65536\n"
            "  provider: ollama-launch\n"
            "providers:\n"
            "  ollama-launch:\n"
            "    api: http://127.0.0.1:11434/v1\n"
            f"    default_model: {model_name}\n"
            "    models:\n"
            f"    - {model_name}\n"
            "    name: Ollama\n"
            "toolsets:\n"
            "- hermes-cli\n"
            "- web\n"
        )
        config_file.write_text(config_content, encoding="utf-8")
        _log_setup("Hermes config.yaml created (Ollama provider)", "OK")
    else:
        _log_setup("Hermes config.yaml exists", "OK")

    _log_setup("Hermes Agent fully configured", "OK")
    return True


def _check_python_deps() -> bool:
    # Supervisor only uses stdlib — suite deps are installed on VMs later.
    # Just verify the current interpreter works.
    rc, _ = _run_cmd(f"{sys.executable} -c \"import json, subprocess; print('OK')\"")
    if rc == 0:
        _log_setup(f"Python verified: {sys.executable}", "OK")
    else:
        _log_setup(f"Python check failed: {sys.executable}", "WARN")
    return True


def _chrome_works() -> bool:
    """Check if any Chrome binary actually runs (not just exists as broken symlink)."""
    for cmd in ["google-chrome", "chromium-browser", "chromium"]:
        if shutil.which(cmd):
            rc, ver = _run_cmd(f"{cmd} --version 2>/dev/null")
            if rc == 0 and ver.strip():
                return True
    return False


def _install_chrome_linux(pm: str):
    """Remove broken Chrome install and do a clean install."""
    _run_cmd(f"{pm} remove -y google-chrome-stable 2>/dev/null")
    _run_cmd("rm -rf /opt/google/chrome")
    if pm in ("dnf", "yum"):
        _run_cmd(
            "curl -fsSL -o /tmp/chrome.rpm "
            "https://dl.google.com/linux/direct/google-chrome-stable_current_x86_64.rpm && "
            f"{pm} install -y /tmp/chrome.rpm && rm -f /tmp/chrome.rpm",
            timeout=180)
    elif pm == "apt-get":
        _run_cmd(
            "curl -fsSL -o /tmp/chrome.deb "
            "https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb && "
            "apt-get update -qq && apt-get install -y /tmp/chrome.deb && rm -f /tmp/chrome.deb",
            timeout=180)


def _check_chrome(plat: dict) -> bool:
    """Install Google Chrome / Chromium browser locally for Selenium tests."""
    if _chrome_works():
        rc, ver = _run_cmd("google-chrome --version 2>/dev/null || chromium --version 2>/dev/null || chromium-browser --version 2>/dev/null")
        _log_setup(f"Chrome browser: {ver.strip()}", "OK")
        _patch_chrome_for_root(plat)
        return True

    _log_setup("Chrome not working — clean install...", "WARN")
    if plat["is_linux"] and plat["pkg_manager"]:
        _install_chrome_linux(plat["pkg_manager"])
    elif plat["is_mac"] and shutil.which("brew"):
        _run_cmd("brew install --cask google-chrome", timeout=180)
    else:
        _log_setup("Install Google Chrome manually", "WARN")
        return True

    if _chrome_works():
        rc, ver = _run_cmd("google-chrome --version 2>/dev/null")
        _log_setup(f"Chrome installed: {ver.strip()}", "OK")
        _patch_chrome_for_root(plat)
        return True

    _log_setup("Chrome installation failed — Selenium UI tests will not work", "WARN")
    return True


def _patch_chrome_for_root(plat: dict):
    """Chrome crashes as root without --no-sandbox. Create wrapper that forces the flags."""
    if not plat["is_linux"] or os.getuid() != 0:
        return

    # Find the real Chrome binary (not a wrapper we already created)
    real_bin = None
    for cand in ["/opt/google/chrome/chrome", "/opt/google/chrome/google-chrome",
                 "/usr/lib/chromium-browser/chromium-browser", "/usr/lib/chromium/chromium"]:
        if Path(cand).exists():
            real_bin = cand
            break
    if not real_bin:
        # Try resolving from which
        rc, path = _run_cmd("readlink -f $(which google-chrome 2>/dev/null || which chromium 2>/dev/null) 2>/dev/null")
        if rc == 0 and path.strip() and Path(path.strip()).exists():
            real_bin = path.strip()
    if not real_bin:
        _log_setup("Chrome root patch — cannot find real binary", "WARN")
        return

    # Resize /dev/shm if too small (Chrome needs at least 256MB, more for concurrent tests)
    try:
        shm = Path("/dev/shm")
        if shm.exists():
            import shutil as _shutil
            shm_total = _shutil.disk_usage("/dev/shm").total
            if shm_total < 512 * 1024 * 1024:
                _run_cmd("mount -o remount,size=2G /dev/shm 2>/dev/null")
    except Exception:
        pass

    wrapper_content = (
        f"#!/bin/bash\n"
        f"exec {real_bin} --no-sandbox --disable-setuid-sandbox --no-zygote "
        f"--disable-dev-shm-usage --disable-gpu --disable-software-rasterizer "
        f"--disable-features=VizDisplayCompositor \"$@\"\n"
    )

    # Write wrappers to /usr/local/bin — always overwrite to pick up new flags
    for name in ["google-chrome", "google-chrome-stable", "chromium-browser", "chromium"]:
        wp = Path(f"/usr/local/bin/{name}")
        try:
            wp.write_text(wrapper_content)
            wp.chmod(0o755)
        except Exception:
            pass

    # Force /usr/bin/ entries to point to our wrapper — chromedriver searches /usr/bin/.
    # The RPM installs valid symlinks here → /opt/google/chrome/google-chrome (no flags).
    # We MUST replace them so chromedriver launches Chrome with --no-zygote etc.
    for name in ["google-chrome", "google-chrome-stable"]:
        usrbin = Path(f"/usr/bin/{name}")
        wrapper = Path(f"/usr/local/bin/{name}")
        if not wrapper.exists():
            continue
        try:
            if usrbin.exists() or usrbin.is_symlink():
                usrbin.unlink()
            usrbin.symlink_to(wrapper)
        except Exception:
            pass

    # Belt-and-suspenders: /etc/default/google-chrome is sourced by the RPM launcher.
    # If chromedriver somehow bypasses our wrapper, these flags still get applied.
    try:
        Path("/etc/default/google-chrome").write_text(
            'export CHROME_EXTRA_FLAGS="--no-sandbox --disable-setuid-sandbox --no-zygote '
            '--disable-dev-shm-usage --disable-gpu --disable-software-rasterizer '
            '--disable-features=VizDisplayCompositor"\n'
        )
    except Exception:
        pass

    # Start Xvfb virtual display — Chrome can crash without a display even in headless mode
    try:
        rc_xvfb, _ = _run_cmd("pgrep -f 'Xvfb :99' 2>/dev/null", timeout=5)
        if rc_xvfb != 0:
            _run_cmd("which Xvfb >/dev/null 2>&1 || (dnf install -y xorg-x11-server-Xvfb 2>/dev/null || "
                     "yum install -y xorg-x11-server-Xvfb 2>/dev/null || "
                     "apt-get install -y xvfb 2>/dev/null || true)", timeout=120)
            _run_cmd("Xvfb :99 -screen 0 1920x1080x24 -nolisten tcp &", timeout=5)
            os.environ["DISPLAY"] = ":99"
            _log_setup("Xvfb started on :99", "OK")
        else:
            os.environ["DISPLAY"] = ":99"
    except Exception:
        pass

    # Verify it works (short timeout — Chrome can hang if something is still wrong)
    rc, _ = _run_cmd("timeout 15 /usr/local/bin/google-chrome --headless --dump-dom about:blank 2>/dev/null", timeout=30)
    if rc == 0:
        _log_setup("Chrome patched for root (--no-sandbox wrapper)", "OK")
    else:
        _log_setup("Chrome root patch applied but headless test failed", "WARN")


def _check_chromedriver() -> bool:
    """Install chromedriver locally and fix webdriver-manager for Selenium tests."""
    # 1. Ensure system chromedriver is installed
    rc, ver = _run_cmd("chromedriver --version 2>/dev/null", timeout=5)
    if rc != 0:
        _log_setup("chromedriver not found — installing...", "WARN")
        try:
            import urllib.request, json, zipfile, platform
            api = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json"
            data = json.loads(urllib.request.urlopen(api, timeout=30).read())
            stable = data["channels"]["Stable"]
            cdver = stable["version"]
            arch = platform.machine()
            plat_key = {"x86_64": "linux64", "aarch64": "linux-arm64",
                        "arm64": "mac-arm64"}.get(arch, "mac-x64" if "darwin" in platform.system().lower() else "")
            if not plat_key:
                _log_setup(f"No chromedriver for {arch}", "WARN")
                return True
            url = [d["url"] for d in stable["downloads"]["chromedriver"] if d["platform"] == plat_key][0]
            _log_setup(f"Downloading chromedriver {cdver} ({plat_key})...", "INFO")
            urllib.request.urlretrieve(url, "/tmp/chromedriver.zip")
            with zipfile.ZipFile("/tmp/chromedriver.zip") as zf:
                zf.extractall("/tmp/cd_extract")
            for root, dirs, files in os.walk("/tmp/cd_extract"):
                if "chromedriver" in files:
                    src = os.path.join(root, "chromedriver")
                    dest = "/usr/bin/chromedriver" if os.getuid() == 0 else str(Path.home() / ".local/bin/chromedriver")
                    Path(dest).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest)
                    os.chmod(dest, 0o755)
                    break
            Path("/tmp/chromedriver.zip").unlink(missing_ok=True)
            shutil.rmtree("/tmp/cd_extract", ignore_errors=True)
        except Exception as e:
            _log_setup(f"chromedriver install error: {e}", "WARN")
            return True
    rc, ver = _run_cmd("chromedriver --version 2>/dev/null", timeout=5)
    if rc == 0:
        _log_setup(f"system chromedriver: {ver.strip()}", "OK")
    else:
        _log_setup("chromedriver not available", "WARN")
        return True

    # 2. Upgrade webdriver-manager and pre-populate its cache.
    #    Test scripts call ChromeDriverManager().install() which can return a stale
    #    chromedriver from ~/.wdm/. We upgrade, clear, and re-populate.
    _log_setup("Upgrading webdriver-manager...", "INFO")
    _run_cmd(f"PIP_BREAK_SYSTEM_PACKAGES=1 {sys.executable} -m pip install --quiet --upgrade webdriver-manager 2>/dev/null", timeout=120)
    wdm_cache = Path.home() / ".wdm"
    if wdm_cache.exists():
        shutil.rmtree(wdm_cache, ignore_errors=True)
        _log_setup("Cleared stale ~/.wdm cache", "OK")

    # 3. Pre-populate the cache by running ChromeDriverManager once.
    #    This ensures test scripts get a matching chromedriver.
    _log_setup("Pre-populating webdriver-manager cache...", "INFO")
    rc_wdm, wdm_out = _run_cmd(
        f"{sys.executable} -c \""
        "from webdriver_manager.chrome import ChromeDriverManager; "
        "path = ChromeDriverManager().install(); "
        "print('WDM_PATH=' + path)\"",
        timeout=120)
    if rc_wdm == 0 and "WDM_PATH=" in (wdm_out or ""):
        wdm_path = wdm_out.strip().split("WDM_PATH=")[-1]
        _log_setup(f"webdriver-manager cache populated: {wdm_path}", "OK")
    else:
        _log_setup(f"webdriver-manager cache population failed — test scripts will retry at runtime", "WARN")
        if wdm_out:
            _log_setup(f"  output: {wdm_out.strip()[:200]}", "WARN")
    return True


def _validate_project_structure() -> bool:
    required = {
        "configs": FRAMEWORK_DIR / "configs",
        "prompts": FRAMEWORK_DIR / "prompts",
        "scripts": SCRIPT_DIR,
        "knowledge": FRAMEWORK_DIR / "knowledge",
    }
    missing = []
    for name, path in required.items():
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            _log_setup(f"Created missing directory: {name}/", "INFO")
    _log_setup("Project structure validated", "OK")
    return True


def _create_ollama_systemd_service() -> bool:
    """Create systemd service for auto-start on Linux (non-fatal)."""
    plat = _detect_platform()
    if not plat["is_linux"]:
        return True
    service_path = Path("/etc/systemd/system/ollama-hermes.service")
    if service_path.exists():
        return True
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        return True
    try:
        service_path.write_text(
            f"[Unit]\nDescription=Ollama for Hermes\nAfter=network-online.target\n\n"
            f"[Service]\nType=simple\nEnvironment=OLLAMA_NUM_PARALLEL={OLLAMA_PARALLEL}\n"
            f"Environment=OLLAMA_HOST=0.0.0.0\nExecStart={ollama_bin} serve\n"
            f"Restart=always\nRestartSec=5\n\n[Install]\nWantedBy=multi-user.target\n"
        )
        _run_cmd("systemctl daemon-reload && systemctl enable ollama-hermes.service 2>/dev/null")
        _log_setup("Ollama systemd service created (auto-start on reboot)", "OK")
    except PermissionError:
        pass  # not root, skip silently
    return True


def ensure_environment() -> bool:
    """Validate and install all prerequisites. Returns True if environment is ready."""
    if os.getenv("HERMES_SKIP_SETUP", "").strip() in ("1", "true", "yes"):
        return True

    print("\n" + "═" * 70, flush=True)
    print("  HERMES FRAMEWORK — ENVIRONMENT SETUP", flush=True)
    print("═" * 70, flush=True)

    plat = _detect_platform()
    _log_setup(f"Platform: {plat['system']} / {plat['machine']}", "INFO")

    steps = [
        ("Python 3.9+",            lambda: _check_python()),
        ("pip",                     lambda: _check_pip()),
        ("System packages",         lambda: _install_system_packages(plat)),
        ("Ollama",                  lambda: _check_ollama(plat)),
        ("Ollama server",           lambda: _start_ollama()),
        ("AI model",                lambda: _check_model(HERMES_MODEL_FAST)),
        ("Model verification",      lambda: _verify_model(HERMES_MODEL_FAST)),
        ("Hermes Agent",            lambda: _setup_hermes_agent()),
        ("Python dependencies",     lambda: _check_python_deps()),
        ("Chrome browser",          lambda: _check_chrome(plat)),
        ("chromedriver",            lambda: _check_chromedriver()),
        ("Project structure",       lambda: _validate_project_structure()),
        ("Ollama auto-start",       lambda: _create_ollama_systemd_service()),
    ]

    passed = 0
    failed = 0
    fatal_fail = False

    for name, check_fn in steps:
        try:
            if check_fn():
                passed += 1
            else:
                failed += 1
                if name in ("Python 3.9+", "System packages", "Ollama", "Ollama server", "AI model"):
                    fatal_fail = True
        except Exception as e:
            _log_setup(f"{name} — unexpected error: {e}", "FAIL")
            failed += 1

    print(f"\n  Setup: {passed} passed, {failed} failed", flush=True)
    print("═" * 70, flush=True)

    if fatal_fail:
        print("\n  FATAL: Required dependencies are missing. Cannot proceed.", flush=True)
        print("  Fix the issues above and re-run.\n", flush=True)
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# Enhanced Agent Layer — NousResearch Hermes Agent integration
# Deferred import: loaded after ensure_environment() validates dependencies.
# ─────────────────────────────────────────────────────────────────────────────
_ENHANCED_LOADED = False

def _load_enhanced_agent():
    """Import hermes_enhanced_agent after environment is validated."""
    global _ENHANCED_LOADED
    global create_enhanced_context, spawn_enhanced_agent, post_run_knowledge_update
    global KnowledgeStore, SelfHealingRetry, NotificationHooks, ModelRouter
    if _ENHANCED_LOADED:
        return True
    try:
        from hermes_enhanced_agent import (
            create_enhanced_context as _cec,
            spawn_enhanced_agent as _sea,
            post_run_knowledge_update as _prku,
            KnowledgeStore as _KS,
            SelfHealingRetry as _SHR,
            NotificationHooks as _NH,
            ModelRouter as _MR,
        )
        create_enhanced_context = _cec
        spawn_enhanced_agent = _sea
        post_run_knowledge_update = _prku
        KnowledgeStore = _KS
        SelfHealingRetry = _SHR
        NotificationHooks = _NH
        ModelRouter = _MR
        _ENHANCED_LOADED = True
        return True
    except ImportError as e:
        print(f"[hermes-supervisor] WARNING: hermes_enhanced_agent not loaded: {e}", flush=True)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# AIAgent Python-API loader  (falls back to CLI if install not found)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Env builder
# ─────────────────────────────────────────────────────────────────────────────

def _env(name: str, default: str = "") -> str:
    """Read a connection setting, preferring the plain name.

    REMOTE_HOST / REMOTE_USER / REMOTE_BASE / SSH_PASSWORD are the canonical
    names; the older HERMES_-prefixed spellings are still accepted so an
    existing .env keeps working.
    """
    return os.getenv(name) or os.getenv("HERMES_" + name, default)

def build_env(remote_base: str, cpd_url: str, cpd_user: str, cpd_password: str,
              cluster: str, extra_paths: list[str], *, is_vm: bool = False,
              product: str = "datastage", needs_cpdctl: bool = False) -> str:
    if is_vm:
        base  = remote_base or "/root/DS_Hermes_Framework"
        extra = f"{base}/extra_inputs/cpd-perf-automation"
    else:
        base  = str(FRAMEWORK_DIR)
        extra = ":".join(extra_paths) if extra_paths else str(FRAMEWORK_DIR)
    env_parts = [
        f"export PATH=/usr/local/bin:$PATH",
        f"source /etc/profile.d/cpdctl_env.sh 2>/dev/null",
        f"export HERMES_PRODUCT={product}",
        f"export HERMES_REPO_ROOT={base}",
        f"export HERMES_EXTRA_DIR={extra}",
        # Quoted: these go through `bash -c`, and a password containing $, !,
        # & or a space reached cpdctl mangled - it then failed "Unauthorized".
        f"export CPD_URL={shlex.quote(cpd_url or '')}",
        f"export CPD_URL_HOST={shlex.quote((cpd_url or '').replace('https://', '').replace('http://', '').rstrip('/'))}",
        f"export CPD_USERNAME={shlex.quote(cpd_user or 'cpadmin')}",
        f"export CPD_PASSWORD={shlex.quote(cpd_password or '')}",
        f"export CLUSTER_NAME={cluster or ''}",
        f"export CLUSTER_ID={cluster or ''}",
        # Suites whose setup fetches the CPD password over SSH (e.g. pythonclient_test)
        # need the cluster root password; without it they silently use the string "None".
        f"export CLUSTER_PASS={_env('SSH_PASSWORD')}",
        # One id per supervisor run, for values the product refuses to duplicate
        # (git token names, uploaded file names). Reusing such a name makes the
        # SECOND run fail with "already exists" — a silent trap, since the first
        # run passes. Stable within a run so a suite's own steps agree.
        f"export HERMES_RUN_ID={int(time.time())}",
        f"export CPD_SCOPE=cpd",
    ]
    if needs_cpdctl:
        cpdctl_vars = [
            "export CPDCTL_ENABLE_DSJOB=true",
            "export CPDCTL_ENABLE_DATASTAGE=true",
            "export CPDCTL_ENABLE_DATASTGE=true",
        ]
        if "datastage" in product.lower():
            cpdctl_vars.insert(0, f"export HERMES_JOB0_DIR={base}/job0")
        env_parts.extend(cpdctl_vars)
    return "; ".join(env_parts) + "; "


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap remote VM
# ─────────────────────────────────────────────────────────────────────────────

def _ssh(host, user, pw, cmd, timeout=300):
    r = subprocess.run(
        ["sshpass", f"-p{pw}", "ssh",
         "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
         "-o", "LogLevel=ERROR", f"{user}@{host}", cmd],
        capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout + r.stderr).strip()


_UPLOAD_EXCLUDES = ["out", "OUTPUT", "__pycache__", ".wdm", "*.pyc",
                    "log.html", "output.xml", "report.html", "selenium-screenshot-*"]


def _scp(host, user, pw, local, remote, timeout=600):
    local_path = Path(str(local))
    if local_path.is_dir() and shutil.which("rsync"):
        cmd = ["sshpass", f"-p{pw}", "rsync", "-az",
               "-e", "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"]
        for ex in _UPLOAD_EXCLUDES:
            cmd.extend(["--exclude", ex])
        cmd.extend([str(local), f"{user}@{host}:{remote}"])
    else:
        cmd = ["sshpass", f"-p{pw}", "scp",
               "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
               "-o", "LogLevel=ERROR", "-r", str(local), f"{user}@{host}:{remote}"]
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"[bootstrap] ⚠ SCP timeout uploading {local_path.name} — skipping", flush=True)

_scp_up = _scp


def _collect_cluster_evidence(suite_name: str, ssh_func, namespace: str,
                              outdir: Path):
    """Collect cluster state evidence via SSH after a suite failure.

    Runs a small set of read-only ``oc`` commands on the remote cluster and
    saves the combined output to ``{outdir}/{suite_name}_evidence.txt``.
    Evidence collection is best-effort: all exceptions are caught silently
    so this helper never interrupts the main test run.
    """
    try:
        sections = [
            ("PODS",
             f"oc get pods -n {namespace} --no-headers 2>/dev/null | head -50"),
            ("RECENT EVENTS",
             f"oc get events -n {namespace} --sort-by=.lastTimestamp 2>/dev/null | tail -30"),
            ("PVC",
             f"oc get pvc -n {namespace} --no-headers 2>/dev/null"),
        ]
        evidence_lines: list[str] = []
        for header, cmd in sections:
            evidence_lines.append(f"=== {header} ===")
            _, output = ssh_func(cmd, timeout=30)
            evidence_lines.append(output or "(no output)")
            evidence_lines.append("")
        evidence_path = outdir / f"{suite_name}_evidence.txt"
        evidence_path.write_text(get_sanitizer().sanitize("\n".join(evidence_lines)), encoding="utf-8")
        print(f"[evidence] Collected cluster state for {suite_name}", flush=True)
    except Exception:
        pass


def bootstrap_remote_vm(*, remote_host, remote_user, ssh_password,
                        remote_base, cpd_url, cpd_user, cpd_password,
                        cluster, cpd_namespace="ds", suites_config=None,
                        extra_paths=None, product="datastage",
                        needs_cpdctl=False):
    REPO_ROOT = FRAMEWORK_DIR.parent
    LOCAL_JOB0 = REPO_ROOT / "job0"
    LOCAL_PERF = None
    if extra_paths:
        for ep in extra_paths:
            p = Path(ep)
            if p.exists():
                LOCAL_PERF = p
                break
    LOCAL_TESTS = FRAMEWORK_DIR / "tests"
    extra_dir = f"{remote_base}/extra_inputs/cpd-perf-automation"
    ssh = lambda cmd, **kw: _ssh(remote_host, remote_user, ssh_password, cmd, **kw)
    scp = lambda l, r: _scp(remote_host, remote_user, ssh_password, l, r)

    # 1: SSH check
    rc, _ = ssh("echo OK")
    if rc != 0:
        print("[bootstrap] ✗ SSH failed", flush=True)
        return False
    print("[bootstrap] ✓ SSH OK", flush=True)

    # 2: Clean stale files and create directories
    job0_dirs = f" {remote_base}/job0" if needs_cpdctl else ""
    ssh(f"rm -rf {extra_dir} && mkdir -p{job0_dirs} {extra_dir} {remote_base}/tests/performance")
    print("[bootstrap] ✓ Directories (cleaned stale)", flush=True)

    # 3: Upload job0 (only when cpdctl capability is needed)
    if needs_cpdctl and LOCAL_JOB0.exists():
        for f in LOCAL_JOB0.iterdir():
            if f.is_file():
                scp(f, f"{remote_base}/job0/")
        ssh(f"chmod +x {remote_base}/job0/*.sh 2>/dev/null || true")
        print("[bootstrap] ✓ job0", flush=True)

    # 4: Upload cpd-perf-automation (product-filtered directories)
    if LOCAL_PERF and LOCAL_PERF.exists():
        for f in LOCAL_PERF.iterdir():
            if f.is_file() and f.suffix in (".py", ".json", ".zip", ".sh", ".env", ".txt"):
                scp(f, f"{extra_dir}/")
        product_key = product.replace("-", "").replace("_", "").lower()
        for d in sorted(LOCAL_PERF.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__":
                dir_key = d.name.replace("-", "").replace("_", "").lower()
                if not product or product_key in dir_key:
                    scp(d, f"{extra_dir}/")
        print("[bootstrap] ✓ cpd-perf-automation", flush=True)

    # 5: Upload test scripts
    if LOCAL_TESTS.exists():
        for f in LOCAL_TESTS.iterdir():
            if f.is_file():
                scp(f, f"{remote_base}/tests/")
        if (LOCAL_TESTS / "performance").exists():
            for f in (LOCAL_TESTS / "performance").iterdir():
                if f.is_file():
                    scp(f, f"{remote_base}/tests/performance/")
        ssh(f"chmod +x {remote_base}/tests/*.sh {remote_base}/tests/performance/*.sh 2>/dev/null || true")
        print("[bootstrap] ✓ test scripts", flush=True)

    # 6: Generate source.env
    url = cpd_url or f"https://cpd-{cpd_namespace}.apps.{cluster}.example.com"
    source_env_content = (
        f"CPD_URL={url}\nCPD_USERNAME={cpd_user}\nCPD_PASSWORD={cpd_password}\n"
        f"CPDCTL_ENCRYPTION_KEY_PATH=/root/vpiv/key.txt\n"
    )
    if needs_cpdctl:
        source_env_content += "CPDCTL_ENABLE_DSJOB=true\nCPDCTL_ENABLE_DATASTAGE=true\n"
    ssh(f"cat > {extra_dir}/source.env <<'EOF'\n{source_env_content}EOF")
    print(f"[bootstrap] ✓ source.env → {url}", flush=True)

    # 6b: Align ALL local config files with active cluster credentials
    if LOCAL_PERF:
        cluster_overrides = {
            "cpd_url": url, "username": cpd_user, "password": cpd_password,
            "cluster": cluster, "namespace": cpd_namespace,
        }
        updated = 0

        # Update cluster_credentials.json files
        for cred_path in LOCAL_PERF.rglob("cluster_credentials.json"):
            try:
                existing = json.loads(cred_path.read_text())
            except Exception:
                existing = {}
            existing.update(cluster_overrides)
            cred_path.write_text(json.dumps(existing, indent=2) + "\n")
            updated += 1

        # Update DS_GIT_INTEGRATION_TEST/config.json
        git_config = LOCAL_PERF / "DS_GIT_INTEGRATION_TEST" / "config.json"
        if git_config.exists():
            try:
                cfg = json.loads(git_config.read_text())
            except Exception:
                cfg = {}
            cfg.setdefault("cpd", {})
            cfg["cpd"]["base_url"] = url
            cfg["cpd"]["username"] = cpd_user
            cfg["cpd"]["password"] = cpd_password
            git_config.write_text(json.dumps(cfg, indent=2) + "\n")
            updated += 1

        # Update DS_READ_WRITE_TESTS config files
        for rw_cfg_name in ["config.json", "config/config.json"]:
            rw_config = LOCAL_PERF / "DS_READ_WRITE_TESTS" / rw_cfg_name
            if rw_config.exists():
                try:
                    cfg = json.loads(rw_config.read_text())
                except Exception:
                    cfg = {}
                cfg["cpd_url"] = url
                cfg["username"] = cpd_user
                cfg["password"] = cpd_password
                cfg["cluster"] = cluster
                cfg["namespace"] = cpd_namespace
                rw_config.write_text(json.dumps(cfg, indent=2) + "\n")
                updated += 1

        # Update local source.env
        local_source_env = LOCAL_PERF / "source.env"
        env_content = (
            f"CPD_URL={url}\nCPD_USERNAME={cpd_user}\nCPD_PASSWORD={cpd_password}\n"
            f"CLUSTER_NAME={cluster}\nCPDCTL_ENCRYPTION_KEY_PATH=/root/vpiv/key.txt\n"
        )
        if needs_cpdctl:
            env_content += "CPDCTL_ENABLE_DSJOB=true\nCPDCTL_ENABLE_DATASTAGE=true\n"
        local_source_env.write_text(env_content)
        updated += 1

        print(f"[bootstrap] ✓ {updated} local config files aligned → {url}", flush=True)

    # 7: Install Python 3.11 + pip (Python 3.9 has Selenium Manager issues)
    rc_py, py_out = ssh("python3.11 --version 2>/dev/null", timeout=10)
    if rc_py != 0:
        print("[bootstrap]   Python 3.11 not found — installing...", flush=True)
        _, pkg_mgr = ssh("command -v dnf >/dev/null 2>&1 && echo dnf || "
                         "(command -v yum >/dev/null 2>&1 && echo yum) || "
                         "(command -v apt-get >/dev/null 2>&1 && echo apt-get)")
        pm = pkg_mgr.strip().split('\n')[-1].strip()
        if pm in ("dnf", "yum"):
            ssh(f"{pm} install -y python3.11 python3.11-pip python3.11-devel 2>&1 || "
                f"{pm} install -y python311 python311-pip 2>&1",
                timeout=180)
        elif pm == "apt-get":
            ssh("apt-get update -qq && "
                "apt-get install -y software-properties-common && "
                "add-apt-repository -y ppa:deadsnakes/ppa && "
                "apt-get update -qq && "
                "apt-get install -y python3.11 python3.11-venv python3.11-distutils 2>&1",
                timeout=180)
        rc_py, py_out = ssh("python3.11 --version 2>/dev/null", timeout=10)
    if rc_py == 0:
        print(f"[bootstrap] ✓ {py_out.strip()}", flush=True)
        # Make python3 point to python3.11 via /usr/local/bin (before /usr/bin in PATH)
        ssh("ln -sf /usr/bin/python3.11 /usr/local/bin/python3 && "
            "ln -sf /usr/bin/python3.11 /usr/local/bin/python && "
            "hash -r 2>/dev/null || true")
        # Ensure pip for 3.11
        ssh("python3.11 -m pip --version 2>/dev/null || "
            "curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11",
            timeout=60)
        # Make pip3 use python3.11
        ssh("ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip3 2>/dev/null || true")
        print("[bootstrap] ✓ pip for Python 3.11", flush=True)
    else:
        print(f"[bootstrap] ⚠ Python 3.11 install failed, using system python3", flush=True)
        ssh("""
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if ! python3 -m pip --version >/dev/null 2>&1; then
  PIP_BREAK_SYSTEM_PACKAGES=1 curl -sS "https://bootstrap.pypa.io/pip/${PY_VER}/get-pip.py" | PIP_BREAK_SYSTEM_PACKAGES=1 python3
fi
""", timeout=120)
        print("[bootstrap] ✓ pip (system python3)", flush=True)

    # 8: Install cpdctl
    rc, _ = ssh("command -v cpdctl")
    if rc != 0:
        _, arch = ssh("uname -m")
        a = {"s390x": "s390x", "ppc64le": "ppc64le", "aarch64": "arm64"}.get(arch.strip(), "amd64")
        ssh(f"cd /tmp && curl -sL https://github.com/IBM/cpdctl/releases/latest/download/cpdctl_linux_{a}.tar.gz "
            f"-o cpdctl.tar.gz && tar -zxf cpdctl.tar.gz && mv cpdctl /usr/local/bin/ && rm -f cpdctl.tar.gz", timeout=120)
        print(f"[bootstrap] ✓ cpdctl installed ({a})", flush=True)
    else:
        print("[bootstrap] ✓ cpdctl present", flush=True)

    # 8a2: Ensure C compiler + dev tools (needed to build bcrypt, cryptography wheels on s390x)
    rc_cc, _ = ssh("command -v cc || command -v gcc", timeout=10)
    if rc_cc != 0:
        _, pkg_mgr_chk = ssh("command -v dnf >/dev/null 2>&1 && echo dnf || "
                             "(command -v yum >/dev/null 2>&1 && echo yum) || "
                             "(command -v apt-get >/dev/null 2>&1 && echo apt-get)")
        _pm = pkg_mgr_chk.strip().split('\n')[-1].strip()
        if _pm in ("dnf", "yum"):
            ssh(f"{_pm} install -y gcc gcc-c++ libffi-devel openssl-devel 2>&1", timeout=180)
        elif _pm == "apt-get":
            ssh("apt-get update -qq && apt-get install -y gcc g++ libffi-dev libssl-dev 2>&1", timeout=180)
        print("[bootstrap] ✓ C compiler + dev headers", flush=True)
    else:
        print("[bootstrap] ✓ C compiler present", flush=True)

    # 8b: Install Python dependencies on remote VM
    # Use python3.11 -m pip to ensure packages go to 3.11
    _pip_cmd = "python3.11 -m pip" if rc_py == 0 else "pip3"
    _py_cmd = "python3.11" if rc_py == 0 else "python3"

    # Install base packages (AI-generated requirements installed later after suite generation)
    _remote_pkgs = ["requests", "plotly", "urllib3"]
    for pkg in _remote_pkgs:
        rc, out = ssh(f"PIP_BREAK_SYSTEM_PACKAGES=1 {_pip_cmd} install --quiet {pkg}", timeout=120)
        if rc != 0:
            print(f"[bootstrap] ✗ {_pip_cmd} install {pkg} failed: {out[:150]}", flush=True)
    print(f"[bootstrap] ✓ base Python deps ({_py_cmd})", flush=True)

    # 8c: Chrome/chromedriver — Selenium tests have run_local=true, so Chrome is needed
    # on THIS machine (where script runs), not on the remote host.
    # Local Chrome + chromedriver are installed by ensure_environment() steps.
    print("[bootstrap] ✓ Chrome/Selenium — handled locally (all Selenium suites have run_local=true)", flush=True)

    # 8d: Install Hermes Agent on remote VM (needed for AI-powered log analysis)
    _hermes_py = "python3.11" if rc_py == 0 else "python3"
    rc_ha, _ = ssh(f"test -f /root/.hermes/hermes-agent/run_agent.py && "
                   f"test -f /root/.hermes/hermes-agent/venv/bin/python3", timeout=10)
    if rc_ha != 0:
        print("[bootstrap]   Installing Hermes Agent on VM...", flush=True)
        ssh("mkdir -p /root/.hermes", timeout=10)
        rc_git, git_out = ssh(
            "cd /root/.hermes && "
            "(test -d hermes-agent/.git && cd hermes-agent && git pull || "
            " git clone https://github.com/NousResearch/hermes-agent.git hermes-agent)",
            timeout=120)
        if rc_git == 0:
            rc_venv, _ = ssh(
                f"{_hermes_py} -m venv /root/.hermes/hermes-agent/venv && "
                f"/root/.hermes/hermes-agent/venv/bin/pip install --upgrade pip && "
                f"/root/.hermes/hermes-agent/venv/bin/pip install pydantic ollama rich 2>&1 | tail -3",
                timeout=300)
            if rc_venv == 0:
                print("[bootstrap] ✓ Hermes Agent installed on VM", flush=True)
            else:
                print("[bootstrap] ⚠ Hermes Agent venv/deps failed — analysis will use fallback", flush=True)
        else:
            print(f"[bootstrap] ⚠ Hermes Agent git clone failed: {git_out[:200]}", flush=True)
    else:
        print("[bootstrap] ✓ Hermes Agent already installed on VM", flush=True)

    # 8e: Clean cpdctl config + encryption key
    ssh("rm -rf /root/.cpdctl && mkdir -p /root/.cpdctl && chmod 700 /root/.cpdctl && "
        "rm -f /root/vpiv/key.txt && mkdir -p /root/vpiv && echo 'dsuser' > /root/vpiv/key.txt")
    print("[bootstrap] ✓ cpdctl config cleaned", flush=True)

    # 10: Persist env vars (PATH ensures /usr/local/bin is first — python3→3.11 symlink lives there)
    env_sh = "export PATH=/usr/local/bin:$PATH\nexport CPDCTL_ENCRYPTION_KEY_PATH=/root/vpiv/key.txt\n"
    if needs_cpdctl:
        env_sh += "export CPDCTL_ENABLE_DSJOB=true\nexport CPDCTL_ENABLE_DATASTAGE=true\nexport CPDCTL_ENABLE_DATASTGE=true\n"
    ssh(f"cat > /etc/profile.d/cpdctl_env.sh <<'EOF'\n{env_sh}EOF\nchmod 644 /etc/profile.d/cpdctl_env.sh")
    print("[bootstrap] ✓ env vars persisted", flush=True)

    # 11: Verify cpdctl
    if needs_cpdctl:
        rc, out = ssh("source /etc/profile.d/cpdctl_env.sh && cpdctl dsjob --help >/dev/null 2>&1 && echo OK")
        print(f"[bootstrap] {'✓' if 'OK' in out else '✗'} dsjob", flush=True)
    else:
        rc, out = ssh("source /etc/profile.d/cpdctl_env.sh && cpdctl --help >/dev/null 2>&1 && echo OK")
        print(f"[bootstrap] {'✓' if 'OK' in out else '✗'} cpdctl", flush=True)

    # 12: Configure cpdctl profile
    setup = LOCAL_JOB0 / "setup_cpdctl_from_cluster.sh"
    if setup.exists():
        scp(setup, "/tmp/")
        cpd_url_arg = f" --cpd-url '{url}'" if url else ""
        rc, out = ssh(
            f"source /etc/profile.d/cpdctl_env.sh && chmod +x /tmp/setup_cpdctl_from_cluster.sh && "
            f"bash /tmp/setup_cpdctl_from_cluster.sh --cluster {cluster} --namespace {cpd_namespace} --cpd-user {cpd_user}{cpd_url_arg}",
            timeout=120)
        print(f"[bootstrap] {'✓' if rc == 0 else '✗'} cpdctl profile", flush=True)
        if rc != 0:
            print(f"[bootstrap]   {out[:300]}", flush=True)

    # 12b: Re-set profile with user-provided password (setup script uses oc-extracted
    #       password which may differ from --cpd-password)
    if needs_cpdctl and url and cpd_user and cpd_password:
        rc, out = ssh(
            f"source /etc/profile.d/cpdctl_env.sh && "
            f"cpdctl config profile set job0-profile --url '{url}' --username '{cpd_user}' --password '{cpd_password}' && "
            f"cpdctl config profile use job0-profile",
            timeout=30)
        print(f"[bootstrap] {'✓' if rc == 0 else '✗'} cpdctl profile override (user-provided password)", flush=True)

    # 13: Pre-create CPD projects (cpdctl 1.8.233 lacks --scope; CPD 5.4+ requires it)
    project_ids = _precreate_cpd_projects(ssh, url, cpd_user, cpd_password, suites_config or {})

    print("[bootstrap] ✓ Bootstrap complete", flush=True)
    return project_ids or True


def _precreate_cpd_projects(ssh, cpd_url, cpd_user, cpd_password, suites_config):
    """Pre-create CPD projects and return {project_name: project_id} mapping."""
    project_ids: dict[str, str] = {}
    needed = {s["cpd_project"] for s in suites_config.get("suites", []) if s.get("cpd_project")}
    if not needed:
        return project_ids
    rc, tok = ssh(
        f"curl -sk -X POST '{cpd_url}/icp4d-api/v1/authorize' "
        f"-H 'Content-Type: application/json' "
        f"-d '{{\"username\":\"{cpd_user}\",\"password\":\"{cpd_password}\"}}' "
        f"| python3 -c \"import sys,json; print(json.load(sys.stdin).get('token',''))\"",
        timeout=30)
    token = tok.strip()
    if not token:
        print("[bootstrap] ✗ CPD token failed — skipping project pre-creation", flush=True)
        return project_ids
    # Get existing projects with name→id mapping
    rc, existing_out = ssh(
        "source /etc/profile.d/cpdctl_env.sh && "
        "cpdctl project list --output json 2>/dev/null "
        "| python3 -c \"import sys,json; "
        "[print(p.get('metadata',{}).get('guid','')+' '+p.get('entity',{}).get('name','')) "
        "for p in json.load(sys.stdin).get('resources',[])]\"",
        timeout=30)
    existing: dict[str, str] = {}
    for line in existing_out.strip().splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            existing[parts[1]] = parts[0]
    created = 0
    for name in needed:
        if name in existing:
            project_ids[name] = existing[name]
            continue
        rc_c, out_c = ssh(
            f"curl -sk -X POST '{cpd_url}/v2/projects' "
            f"-H 'Authorization: Bearer {token}' "
            f"-H 'Content-Type: application/json' "
            f"-d '{{\"name\":\"{name}\","
            f"\"description\":\"Auto-created by Hermes\","
            f"\"generator\":\"cpdctl-ds\","
            f"\"storage\":{{\"type\":\"assetfiles\",\"guid\":\"not_applicable\"}},"
            f"\"scope\":{{\"bss_account_id\":\"999\"}}}}' "
            f"| python3 -c \"import sys,json; d=json.loads(sys.stdin.read() or '{{}}'); print(d.get('metadata',d).get('guid',''))\" 2>/dev/null",
            timeout=30)
        pid = out_c.strip()
        if pid:
            project_ids[name] = pid
            created += 1
    print(f"[bootstrap] ✓ CPD projects ({created} new, {len(existing)} existing)", flush=True)
    if project_ids:
        print(f"[bootstrap]   project IDs: {project_ids}", flush=True)
    return project_ids


# ─────────────────────────────────────────────────────────────────────────────
# Standalone Jenkins job runner (module-level so any lane can call it)
# ─────────────────────────────────────────────────────────────────────────────

def _run_jenkins_job_standalone(suite: dict, log_path: Path) -> tuple:
    """Trigger a Jenkins job, poll for completion, write log. Returns (rc, console_tail, result, build_url)."""
    import urllib.request, urllib.parse, http.client, http.cookiejar
    jenkins_config = suite.get("jenkins_config", "")
    job_name = suite.get("jenkins_job_name", "")

    cfg = {}
    if jenkins_config and os.path.isfile(jenkins_config):
        try:
            cfg = json.loads(Path(jenkins_config).read_text(encoding="utf-8"))
        except Exception:
            pass
    jenkins_url = cfg.get("jenkins_url", "")
    job_entry = None
    for j in cfg.get("jobs", []):
        if j.get("name") == job_name:
            job_entry = j
            break
    if not job_entry:
        log_path.write_text(f"suite={job_name}\nexit_code=1\nduration=0s\n\nERROR: Job '{job_name}' not found in config\n")
        return (1, "", "NOT_FOUND", "")

    job_path = job_entry.get("job_path", "")
    timeout_sec = job_entry.get("timeout_sec", 7200)
    params = job_entry.get("parameters", {})

    env_vars = dict(os.environ)
    env_file = FRAMEWORK_DIR / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                env_vars[k.strip()] = v.strip()

    if '{' in job_path:
        for env_k, env_v in env_vars.items():
            job_path = job_path.replace(f"{{{env_k}}}", env_v)

    resolved = {}
    for pname, pinfo in params.items():
        src = pinfo.get("source", "")
        val = pinfo.get("value", "")
        template = pinfo.get("template", "")
        if src in ("hardcoded", "default"):
            resolved[pname] = str(val)
        elif src.startswith("env:"):
            env_key = src.split(":", 1)[1]
            env_val = env_vars.get(env_key, str(val) if val != "auto" else "")
            if template and env_val:
                resolved[pname] = template.replace(f"{{{env_key}}}", env_val)
            else:
                resolved[pname] = env_val
        elif val == "auto":
            resolved[pname] = env_vars.get(pname, "")
        else:
            resolved[pname] = str(val)

    base_url = f"{jenkins_url}/{job_path}"

    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))

    jenkins_user = env_vars.get("JENKINS_USER", cfg.get("jenkins_user", ""))
    jenkins_token = env_vars.get("JENKINS_API_TOKEN", cfg.get("jenkins_api_token", ""))
    auth_headers = {}
    if jenkins_user and jenkins_token:
        import base64
        cred = base64.b64encode(f"{jenkins_user}:{jenkins_token}".encode()).decode()
        auth_headers["Authorization"] = f"Basic {cred}"

    crumb_header = {}
    try:
        crumb_url = f"{jenkins_url}/crumbIssuer/api/json"
        crumb_req = urllib.request.Request(crumb_url)
        for k, v in auth_headers.items():
            crumb_req.add_header(k, v)
        with opener.open(crumb_req, timeout=10) as resp:
            crumb_data = json.loads(resp.read().decode())
            crumb_header[crumb_data["crumbRequestField"]] = crumb_data["crumb"]
            print(f"[hermes-supervisor]   [jenkins] CSRF crumb acquired (session cookie stored)", flush=True)
    except Exception as e:
        print(f"[hermes-supervisor]   [jenkins] no CSRF crumb: {e}", flush=True)

    def _jenkins_open(url, method="GET", data=None):
        r = urllib.request.Request(url, data=data, method=method)
        for k, v in {**auth_headers, **crumb_header}.items():
            r.add_header(k, v)
        return opener.open(r, timeout=30)

    prev_build = None
    try:
        with _jenkins_open(f"{base_url}/lastBuild/buildNumber") as resp:
            prev_build = resp.read().decode().strip()
    except Exception:
        pass

    trigger_data = urllib.parse.urlencode(resolved).encode()
    try:
        req = urllib.request.Request(f"{base_url}/buildWithParameters", data=trigger_data, method="POST")
        for k, v in {**auth_headers, **crumb_header}.items():
            req.add_header(k, v)
        with opener.open(req, timeout=30) as resp:
            http_code = resp.status
            resp_body = resp.read().decode(errors="replace")[:500]
    except urllib.error.HTTPError as e:
        http_code = e.code
        resp_body = ""
    except Exception as e:
        log_path.write_text(f"suite={job_name}\nexit_code=1\nduration=0s\n\nERROR: Failed to trigger: {e}\n")
        return (1, "", "TRIGGER_FAILED", base_url)

    if http_code == 200 and ("<html" in resp_body.lower() or "login" in resp_body.lower()):
        log_path.write_text(f"suite={job_name}\nexit_code=1\nduration=0s\n\nERROR: HTTP 200 but got login page - authentication failed. Set JENKINS_USER and JENKINS_API_TOKEN in .env\n")
        print(f"[hermes-supervisor]   [jenkins] HTTP 200 but login page returned — auth failed", flush=True)
        return (1, "", "AUTH_FAILED", base_url)

    if http_code not in (200, 201, 302):
        log_path.write_text(f"suite={job_name}\nexit_code=1\nduration=0s\n\nERROR: Trigger returned HTTP {http_code}\n")
        print(f"[hermes-supervisor]   [jenkins] HTTP {http_code}", flush=True)
        return (1, "", f"HTTP_{http_code}", base_url)

    print(f"[hermes-supervisor]   [jenkins] triggered {job_name} (HTTP {http_code})", flush=True)

    time.sleep(15)
    build_number = None
    for attempt in range(5):
        try:
            with _jenkins_open(f"{base_url}/lastBuild/buildNumber") as resp:
                build_number = resp.read().decode().strip()
                break
        except Exception:
            time.sleep(5)

    if build_number and prev_build and build_number == prev_build:
        print(f"[hermes-supervisor]   [jenkins] WARNING: build #{build_number} same as before trigger — job may not have started", flush=True)
        time.sleep(15)
        try:
            with _jenkins_open(f"{base_url}/lastBuild/buildNumber") as resp:
                build_number = resp.read().decode().strip()
        except Exception:
            pass
        if build_number == prev_build:
            log_path.write_text(f"suite={job_name}\nexit_code=1\nduration=30s\n\nERROR: No new build created after trigger — job was not actually triggered. Check Jenkins auth credentials.\n")
            print(f"[hermes-supervisor]   [jenkins] ✗ no new build created — trigger was not real", flush=True)
            return (1, "", "NOT_TRIGGERED", base_url)

    if not build_number:
        log_path.write_text(f"suite={job_name}\nexit_code=1\nduration=15s\n\nERROR: Could not get build number\n")
        return (1, "", "NO_BUILD_NUMBER", base_url)

    print(f"[hermes-supervisor]   [jenkins] build #{build_number} started", flush=True)
    build_url = f"{base_url}/{build_number}"

    t_start = time.time()
    build_result = "UNKNOWN"
    while (time.time() - t_start) < timeout_sec:
        try:
            with _jenkins_open(f"{build_url}/api/json?tree=result,building") as resp:
                data = json.loads(resp.read().decode())
            if not data.get("building", True):
                build_result = data.get("result", "UNKNOWN")
                break
        except Exception:
            pass
        print(f"[hermes-supervisor]   [jenkins] build #{build_number} still running ({round(time.time()-t_start)}s)...", flush=True)
        time.sleep(30)
    else:
        build_result = "TIMEOUT"

    def _fetch_console():
        # progressiveText in offset-sized pieces: a single consoleText GET of a
        # multi-MB log is often cut off mid-chunk (http.client.IncompleteRead).
        parts, start = [], 0
        for _ in range(10000):
            try:
                with _jenkins_open(f"{build_url}/logText/progressiveText?start={start}") as resp:
                    try:
                        body = resp.read()
                    except http.client.IncompleteRead as e:
                        body = e.partial
                    more = resp.headers.get("X-More-Data", "").lower() == "true"
                    nxt = int(resp.headers.get("X-Text-Size", start + len(body)))
            except Exception:
                break
            parts.append(body)
            if nxt <= start or (not more and nxt == start + len(body)):
                break
            start = nxt
        return b"".join(parts).decode(errors="replace")

    full_log = _fetch_console()
    console_tail = ("\n".join(full_log.splitlines()[-200:]) if full_log
                    else "(could not fetch console log)")

    rc = 0 if build_result == "SUCCESS" else (124 if build_result == "TIMEOUT" else 1)
    dur = round(time.time() - t_start)

    def _mask(text):
        # Jobs may echo their password parameters in plain text, so every
        # sensitive value is scrubbed before anything is written.
        for pname, pinfo in params.items():
            if pinfo.get("sensitive") and resolved.get(pname):
                text = text.replace(resolved[pname], "********")
        return get_sanitizer().sanitize(text)

    masked_console = _mask(console_tail)

    # Keep the whole console, not just the tail: long jobs put their results
    # far above the last 200 lines, and log analysers need all of it.
    console_path = log_path.with_name(f"{log_path.stem}.console.log")
    if full_log:
        console_path.write_text(_mask(full_log), encoding="utf-8")
        print(f"[hermes-supervisor]   [jenkins] full console saved → {console_path} "
              f"({len(full_log.splitlines())} lines)", flush=True)

    # AI agent log analysis: ask Ollama to verify Jenkins result
    ai_verdict = None
    if console_tail and console_tail != "(could not fetch console log)":
        log_snippet = "\n".join(console_tail.splitlines()[-150:])
        ai_prompt = (
            f"You are a Jenkins log analyst. Jenkins reported build result: {build_result}\n\n"
            f"Analyze the console log below and determine the REAL result.\n"
            f"Look for: error messages, failed tests, 401/403 errors, exceptions, "
            f"scenarios marked as FAILED/ERROR, and misleading success messages.\n\n"
            f"Console log:\n```\n{log_snippet}\n```\n\n"
            f"Respond with ONLY a JSON object:\n"
            f'{{"real_result": "PASSED" or "FAILED", '
            f'"reason": "one-line explanation", '
            f'"failures_found": ["list of specific failures"]}}'
        )
        try:
            import urllib.request
            ai_req = urllib.request.Request(
                "http://127.0.0.1:11434/api/generate",
                data=json.dumps({"model": "gemma4:12b", "prompt": ai_prompt, "stream": False}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(ai_req, timeout=120) as ai_resp:
                ai_raw = json.loads(ai_resp.read().decode()).get("response", "")
            json_start = ai_raw.find("{")
            json_end = ai_raw.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                ai_verdict = json.loads(ai_raw[json_start:json_end])
                print(f"[hermes-supervisor]   [jenkins] AI log analysis: {ai_verdict.get('real_result', '?')} — {ai_verdict.get('reason', '')}", flush=True)
        except Exception as e:
            print(f"[hermes-supervisor]   [jenkins] AI log analysis skipped: {e}", flush=True)

    hermes_override = False
    override_note = ""
    if ai_verdict and ai_verdict.get("real_result") == "FAILED" and build_result == "SUCCESS":
        failures = [str(f).strip() for f in (ai_verdict.get("failures_found") or [])
                    if str(f).strip()]
        reason = ai_verdict.get("reason", "")

        # A 12B model does not get to overturn a deterministic build result on
        # its say-so. The override is one-way - it can only turn SUCCESS into
        # FAILED, never the reverse - so every hallucination costs a red build
        # and nothing corrects in the other direction. The prompt also asks it
        # to hunt for "error messages" and "misleading success messages", and
        # a log saying "0 errors" or retrying a step once satisfies that
        # reading.
        #
        # So: it must QUOTE something that is actually in the log. Each cited
        # failure is normalised for whitespace and checked against the same
        # snippet the model saw. A citation that is not in the log is not
        # evidence, it is a guess, and a guess must not flip a build.
        haystack = " ".join(log_snippet.split()).lower()
        grounded = [f for f in failures
                    if " ".join(f.split()).lower()[:60] in haystack]

        # Even grounded, a quoted line proves only that the line exists - a
        # retried step or a logged-and-handled error is quoted just as easily
        # as a real failure, and that is how passing builds kept going red.
        # The Jenkins result stands unless the override is asked for.
        override_enabled = os.getenv("HERMES_AI_OVERRIDE", "").lower() in ("1", "true", "yes")

        if grounded and override_enabled:
            hermes_override = True
            build_result = "FAILED_BY_LOG_ANALYSIS"
            rc = 1
            print(f"[hermes-supervisor]   [jenkins] Hermes overriding Jenkins SUCCESS → FAILED", flush=True)
            override_note = (
                f"\n⚠️  HERMES AI LOG ANALYSIS OVERRIDE\n"
                f"Jenkins reported: SUCCESS\n"
                f"AI verdict: FAILED — {reason}\n"
                f"Failures found in the log: {', '.join(grounded)}\n"
                f"Hermes result: FAILED\n\n"
            )
        else:
            # Said out loud rather than dropped: the opinion may still be
            # worth a human's attention, it just cannot change the result.
            why = ("cited nothing" if not failures else
                   "cited failures that do not appear in the log" if not grounded else
                   "override is off (set HERMES_AI_OVERRIDE=1 to let it fail the suite)")
            print(f"[hermes-supervisor]   [jenkins] AI said FAILED but {why} "
                  f"- keeping Jenkins SUCCESS", flush=True)
            override_note = (
                f"\nNOTE: AI log analysis disagreed with Jenkins but was not "
                f"acted on - it {why}.\n"
                f"AI reason: {reason}\n"
                f"Claimed failures: {', '.join(failures) if failures else '(none)'}\n\n"
            )

    log_path.write_text(
        f"suite={job_name}\nexit_code={rc}\nduration={dur}s\n\n"
        f"Jenkins build result: {build_result}\n"
        f"Build URL: {build_url}/console\n"
        f"Full console: {console_path if full_log else '(not fetched)'}\n"
        f"{override_note}\n"
        f"--- Console Output (last 200 lines) ---\n{masked_console}\n",
        encoding="utf-8")

    return (rc, console_tail, build_result, build_url)


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic suite runner (used for run_first + retries)
# ─────────────────────────────────────────────────────────────────────────────

def run_suite_direct(suite: dict, *, outdir: Path,
                     remote_host: str, remote_user: str,
                     ssh_password: str,
                     vm_env_str: str, mac_env_str: str,
                     is_vm: bool,
                     project_ids: dict = None,
                     cpd_namespace: str = "") -> dict:
    """Run a single suite via subprocess (deterministic, no AI)."""
    name     = suite["name"]

    # Jenkins-type suites: delegate to _run_jenkins_job (defined later in main())
    if suite.get("type") == "jenkins":
        log_path = outdir / f"{name}.log"
        print(f"[hermes-supervisor]   [jenkins] {name} — executing", flush=True)
        t0 = time.time()
        rc, console_tail, build_result, build_url = _run_jenkins_job_standalone(suite, log_path)
        dur = round(time.time() - t0)
        status = "passed" if rc == 0 else "failed"
        print(f"[hermes-supervisor]   [jenkins] {'✓' if rc == 0 else '✗'} {name}  rc={rc}  {dur}s  result={build_result}", flush=True)
        return {
            "name": name, "status": status, "return_code": rc,
            "duration_sec": dur, "critical": suite.get("critical", False),
            "log_file": str(log_path),
        }

    env_str  = vm_env_str if is_vm else mac_env_str
    env_str  = "export PIP_BREAK_SYSTEM_PACKAGES=1; " + env_str
    cmd      = suite.get("cmd") or suite.get("command") or ""
    timeout  = suite.get("timeout_sec", 3600)
    cpd_proj = suite.get("cpd_project", "")
    if cpd_proj and project_ids and cpd_proj in project_ids:
        env_str += f"export CPD_PROJECT_ID='{project_ids[cpd_proj]}'; "
    tags = suite.get("tags", [])
    if not is_vm and ("selenium" in tags or "ui" in tags):
        # On Linux, only point at :99 if Xvfb is actually serving it. Bootstrap
        # sets DISPLAY=:99 even when Xvfb failed to install/start, and suites
        # that go headful whenever DISPLAY is set then die with
        # "session not created: Chrome instance exited".
        if sys.platform != "darwin" and not Path("/tmp/.X11-unix/X99").exists():
            env_str += "unset DISPLAY; export HEADLESS=1; "
        else:
            env_str += "export DISPLAY=':99'; "
    log_path = outdir / f"{name}.log"

    # Inject per-suite output directory so robot output files don't collide
    suite_outdir = outdir / name
    suite_outdir.mkdir(parents=True, exist_ok=True)
    if "python3 -m robot " in cmd:
        cmd = cmd.replace("python3 -m robot ", f"python3 -m robot -d '{suite_outdir}' ", 1)

    if is_vm and remote_host:
        shell_cmd = [
            "sshpass", f"-p{ssh_password}",
            "ssh", "-o", "StrictHostKeyChecking=no",
            f"{remote_user}@{remote_host}",
            f"{env_str}{cmd}",
        ]
    else:
        # Kill only orphaned chromedriver processes before local Selenium tests
        # (don't kill Chrome itself — the test's Close All Browsers handles that)
        if "selenium" in tags or "ui" in tags:
            try:
                subprocess.run(["pkill", "-9", "-f", "chromedriver"],
                               capture_output=True, timeout=5)
            except Exception:
                pass
            time.sleep(5)

        shell_cmd = ["bash", "-c", f"{env_str}{cmd}"]

    print(f"[hermes-supervisor]   → {name}  (timeout={timeout}s)", flush=True)
    t0 = time.time()
    try:
        proc = subprocess.run(shell_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, timeout=timeout)
        rc  = proc.returncode
        out = proc.stdout or ""
    except subprocess.TimeoutExpired as exc:
        rc  = 124
        raw = exc.stdout or b""
        out = (raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw)
        out += f"\n[TIMED OUT after {timeout}s]"
    except Exception as exc:
        rc  = 1
        out = str(exc)

    duration = round(time.time() - t0)
    out = get_sanitizer().sanitize(out)
    log_path.write_text(
        f"suite={name}\nstart={datetime.now(timezone.utc).isoformat()}\n"
        f"exit_code={rc}\nduration={duration}s\n\n{out}",
        encoding="utf-8",
    )
    # Override rc=0 if log contains failure keywords (scripts may exit 0 on internal failure)
    rc, status = _check_log_for_failures(log_path, rc)
    if rc != 0 and is_vm and remote_host and cpd_namespace:
        _collect_cluster_evidence(name, lambda cmd, **kw: _ssh(remote_host, remote_user, ssh_password, cmd, **kw), cpd_namespace, outdir)
    print(f"[hermes-supervisor]   {'✓' if rc == 0 else '✗'} {name}  rc={rc}  {duration}s", flush=True)
    return {
        "name": name, "status": status, "return_code": rc,
        "duration_sec": duration, "critical": suite.get("critical", False),
        "log_file": str(log_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Parse executor-agent JSON response → suite results
# ─────────────────────────────────────────────────────────────────────────────

_LOG_FAILURE_PATTERNS = re.compile(
    r'Login timeout|Login error|Login failed|TEST FAILED|RunStartFailed'
    r'|❌|✗ FAILED|exit_code=[1-9]|\[IMPORT\] Failed'
    r'|Status code\s*=\s*[1-9]|WORKFLOW.*FAILED|all .* flows? .*failed'
    r'|Failed:\s*[1-9]|Successful:\s*0'
    r'|ModuleNotFoundError|ImportError|No module named',
    re.IGNORECASE
)

def _check_log_for_failures(log_path: Path, rc: int) -> tuple[int, str]:
    """If log contains failure keywords, override rc=0 to rc=1 and status=failed."""
    if rc == 0 and log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        if _LOG_FAILURE_PATTERNS.search(text):
            return 1, "failed"
    return rc, ("passed" if rc == 0 else ("error" if rc == 124 else "failed"))




# ─────────────────────────────────────────────────────────────────────────────
# Discovery artifacts (fallback writer — used if discovery agent fails)
# ─────────────────────────────────────────────────────────────────────────────

def write_discovery_artifacts(outdir: Path, remote_host: str, remote_user: str,
                               remote_base: str, suites: list[dict],
                               manifest: dict) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rh  = (f"YES — {remote_user}@{remote_host}  base={remote_base}"
           if remote_host else "NO — Mac-only run")
    suite_map = {s["name"]: s for s in suites}
    inventory = []
    for lane, names in [("run_first", manifest["run_first"]),
                        ("mac_lane",  manifest["mac_lane"]),
                        ("vm_lane",   manifest["vm_lane"])]:
        for name in names:
            s = suite_map.get(name, {"name": name})
            inventory.append({
                "name": name, "lane": lane,
                "critical": s.get("critical", False),
                "run_local": s.get("run_local", False),
                "timeout_sec": s.get("timeout_sec", 3600),
                "what_it_tests": ", ".join(s.get("tags", [])) or name,
                "dependencies": [], "risks": [],
            })
    (outdir / "suite_inventory.json").write_text(json.dumps({
        "generated_at": now, "remote_host": rh,
        "total_suites": len(inventory), "suites": inventory,
    }, indent=2), encoding="utf-8")
    (outdir / "bootstrap_manifest.json").write_text(json.dumps({
        "required": bool(remote_host), "remote_host": rh, "remote_base": remote_base or "(none)",
        "reason": "Remote VM must have framework, CPD env, Python deps and cpdctl.",
        "steps": [
            {"step": "upload_framework", "required": True,  "why": "Framework scripts must exist on VM."},
            {"step": "write_cpd_env",    "required": True,  "why": "CPD URL/user/password env vars needed."},
            {"step": "install_py_deps",  "required": True,  "why": "Python packages required."},
            {"step": "setup_cpdctl",     "required": True,  "why": "cpdctl needed to import/run DataStage jobs."},
        ],
        "vm_lane_suites": manifest["vm_lane"],
        "mac_lane_notes": "mac_lane suites run locally without SSH.",
    }, indent=2), encoding="utf-8")
    (outdir / "hermes_task_plan.json").write_text(json.dumps({
        "generated_at": now, "remote_host": rh, "model": HERMES_MODEL,
        "lanes": {
            "run_first": {"suites": manifest["run_first"], "description": "Sequential gate."},
            "mac_lane":  {"suites": manifest["mac_lane"],  "description": "Run locally (no SSH)."},
            "vm_lane":   {"suites": manifest["vm_lane"],   "description": "Run on remote VM via SSH."},
        },
        "gate_criteria": {
            "FAIL": "any critical suite failed",
            "PASS_WITH_RISK": "only non-critical suites failed",
            "PASS": "all suites passed",
        },
        "estimated_risks": ["Remote host connectivity", "Timeouts in long-running perf tests"],
    }, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Lane split + gate
# ─────────────────────────────────────────────────────────────────────────────

def enforce_lane_split_deterministic(suites: list[dict], *, remote_host: str = "") -> dict:
    """Fallback: deterministic lane split from config fields."""
    if not remote_host:
        return {
            "run_first": [s["name"] for s in suites if s.get("run_first")],
            "mac_lane":  [s["name"] for s in suites if not s.get("run_first")],
            "vm_lane":   [],
        }
    return {
        "run_first": [s["name"] for s in suites if s.get("run_first")],
        "mac_lane":  [s["name"] for s in suites if s.get("run_local") and not s.get("run_first")],
        "vm_lane":   [s["name"] for s in suites if not s.get("run_local") and not s.get("run_first")],
    }


def enforce_lane_split(suites: list[dict], *, remote_host: str = "",
                       outdir: Path = None, knowledge: object = None,
                       run_id: str = "", agent_telemetry: list = None) -> dict:
    """Agent-based lane split: spawns a discovery agent to classify suites,
    with deterministic fallback on failure."""
    if not remote_host:
        return enforce_lane_split_deterministic(suites, remote_host=remote_host)

    suite_descriptions = "\n".join(
        f"  {i+1}. name={s['name']}, cmd={s['cmd'][:100]}, "
        f"tags={s.get('tags', [])}, critical={s.get('critical', False)}"
        for i, s in enumerate(suites)
    )
    suite_names = {s["name"] for s in suites}

    prompt = (
        "You are a test orchestration agent for IBM DataStage CI.\n"
        "Classify these test suites into three lanes:\n\n"
        "- run_first: suites marked as critical gate checks (must pass before others run)\n"
        "- mac_lane: suites that need a local browser (Selenium/Chrome/UI) or local Mac resources\n"
        "- vm_lane: suites that run remotely via SSH (cpdctl CLI, Python perf scripts on a cluster)\n\n"
        f"Suites:\n{suite_descriptions}\n\n"
        "Rules:\n"
        "- Every suite must appear in exactly ONE lane\n"
        "- Use suite names only (not numbers)\n"
        "- Respond with ONLY a JSON object, no explanation:\n"
        '  {"run_first": [...], "mac_lane": [...], "vm_lane": [...]}'
    )

    try:
        meta = spawn_enhanced_agent(
            "discovery", prompt,
            outdir=outdir or Path("/tmp"),
            toolsets=["file"],
            max_turns=5,
            timeout=120,
            knowledge=knowledge,
            run_id=run_id,
            role="discovery",
        )
        if agent_telemetry is not None:
            agent_telemetry.append(meta)

        response = meta.get("response", "")
        # Parse JSON from response (handle markdown code blocks)
        json_text = response
        if "```" in json_text:
            match = re.search(r"```(?:json)?\s*\n?(.*?)```", json_text, re.DOTALL)
            json_text = match.group(1).strip() if match else response
        # Try to find JSON object in response
        json_match = re.search(r"\{[^{}]*\"run_first\"[^{}]*\}", json_text, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)

        manifest = json.loads(json_text)

        # Validate: every suite name must appear exactly once
        agent_names = set(
            manifest.get("run_first", []) +
            manifest.get("mac_lane", []) +
            manifest.get("vm_lane", [])
        )
        # A set alone accepts a suite listed in two lanes, and that suite then
        # ran twice - for a Jenkins suite, two builds of a multi-hour job.
        listed = (manifest.get("run_first", []) + manifest.get("mac_lane", []) +
                  manifest.get("vm_lane", []))
        duplicated = {n for n in listed if listed.count(n) > 1}
        if agent_names != suite_names or duplicated:
            missing = suite_names - agent_names
            extra = agent_names - suite_names
            print(f"[hermes-supervisor]   [discovery-agent] WARN: mismatch — "
                  f"missing={missing or 'none'}, extra={extra or 'none'}, "
                  f"in two lanes={duplicated or 'none'}. "
                  f"Using deterministic fallback.", flush=True)
            return enforce_lane_split_deterministic(suites, remote_host=remote_host)

        print(f"[hermes-supervisor]   [discovery-agent] Agent split validated OK "
              f"({meta['duration_sec']}s, {meta['tokens_estimate']} tokens)", flush=True)
        return manifest

    except Exception as e:
        print(f"[hermes-supervisor]   [discovery-agent] Agent failed ({e}). "
              f"Using deterministic fallback.", flush=True)
        return enforce_lane_split_deterministic(suites, remote_host=remote_host)


def decide_gate(results: list[dict]) -> str:
    failed   = [r for r in results if r.get("status") != "passed"]
    critical = [r for r in failed  if r.get("critical")]
    if not failed:  return "PASS"
    if critical:    return "FAIL"
    return "PASS_WITH_RISK"


# ─────────────────────────────────────────────────────────────────────────────
# Prompt renderers
# ─────────────────────────────────────────────────────────────────────────────

def _read_prompt(fname: str) -> str:
    return (PROMPTS_DIR / fname).read_text(encoding="utf-8")












def render_summary_prompt(outdir: Path, results: list[dict],
                          agent_telemetry: list[dict] = None) -> str:
    passed = [r for r in results if r.get("status") == "passed"]
    failed = [r for r in results if r.get("status") != "passed"]
    gate   = (outdir / "hermes_gate.txt").read_text(encoding="utf-8").strip() \
             if (outdir / "hermes_gate.txt").exists() else "UNKNOWN"
    # Format agent telemetry table for the prompt
    telemetry_lines = []
    if agent_telemetry:
        telemetry_lines.append(f"{'Agent':<35} {'Model':<12} {'Duration':>8} {'Tools':>6} {'Prompt Tok':>11} {'Compl Tok':>10} {'Total Tok':>10}")
        telemetry_lines.append("-" * 100)
        for a in agent_telemetry:
            telemetry_lines.append(
                f"{a['agent_id']:<35} {a.get('model',''):<12} {a['duration_sec']:>7.1f}s "
                f"{a['tool_calls']:>6} {a.get('prompt_tokens',0):>11} "
                f"{a.get('completion_tokens',0):>10} {a['tokens_estimate']:>10}"
            )
        telemetry_lines.append("-" * 100)
        telemetry_lines.append(
            f"{'TOTAL':<35} {'':<12} {sum(a['duration_sec'] for a in agent_telemetry):>7.1f}s "
            f"{sum(a['tool_calls'] for a in agent_telemetry):>6} "
            f"{sum(a.get('prompt_tokens',0) for a in agent_telemetry):>11} "
            f"{sum(a.get('completion_tokens',0) for a in agent_telemetry):>10} "
            f"{sum(a['tokens_estimate'] for a in agent_telemetry):>10}"
        )
    telemetry_str = "\n  ".join(telemetry_lines) if telemetry_lines else "No telemetry data available"
    return _read_prompt("hermes_summary.prompt.txt").format(
        outdir       = str(outdir),
        gate         = gate,
        n_passed     = len(passed),
        passed_names = [r["name"] for r in passed],
        n_failed     = len(failed),
        failed_names = [r["name"] for r in failed],
        agent_telemetry = telemetry_str,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry helpers
# ─────────────────────────────────────────────────────────────────────────────

def check_ollama_available() -> bool:
    try:
        r = subprocess.run(["ollama", "list"], stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, timeout=5)
        return r.returncode == 0
    except Exception:
        return False


def _run_log_analyser(outdir: Path, results: list[dict]) -> None:
    """Run the AI log analyser on each failed suite's log, whatever the product.

    HERMES_LOGAI_ALL=1 analyses passing suites too. Each report is saved as
    <suite>.logai.txt in outdir, so Jenkins archives and shows it.
    """
    analyse_all = os.getenv("HERMES_LOGAI_ALL", "").lower() in ("1", "true", "yes")
    targets = [r for r in results if analyse_all or r.get("status") != "passed"]
    if not targets:
        return
    logai_dir = Path(os.getenv("HERMES_LOGAI_DIR", "/root/cp4d_ai_log_analyser"))
    if not logai_dir.is_dir():
        # Said out loud: a missing analyser used to skip silently, which
        # looked exactly like the analysis having found nothing.
        print(f"\n[hermes-supervisor] Log analyser SKIPPED for {len(targets)} suite(s): "
              f"{logai_dir} not found (set HERMES_LOGAI_DIR to its location)", flush=True)
        return
    model = os.getenv("HERMES_MODEL_FAST", os.getenv("HERMES_MODEL", "gemma4:12b"))
    print(f"\n[hermes-supervisor] Log analyser: {len(targets)} suite(s), model={model}", flush=True)
    for r in targets:
        name = r["name"]
        # Jenkins suites keep their full console beside the summary log;
        # it holds results the 200-line summary cuts off.
        log = outdir / f"{name}.console.log"
        if not log.exists():
            log = outdir / f"{name}.log"
        if not log.exists():
            print(f"[hermes-supervisor]   [{name}] no log to analyse", flush=True)
            continue
        print(f"[hermes-supervisor]   [{name}] analysing {log.name}...", flush=True)
        try:
            proc = subprocess.run(
                ["python3.11", "-m", "logai", "analyse", str(log), "--model", model],
                cwd=str(logai_dir), timeout=7200, capture_output=True, text=True,
            )
            report = proc.stdout.strip()
            if proc.returncode != 0 and proc.stderr.strip():
                report += ("\n" if report else "") + proc.stderr.strip()
        except Exception as e:
            report = f"log analyser failed: {e}"
        if report:
            report = get_sanitizer().sanitize(report)
            print(report, flush=True)
            (outdir / f"{name}.logai.txt").write_text(report, encoding="utf-8")
            print(f"[hermes-supervisor]   [{name}] analysis saved → {name}.logai.txt", flush=True)


def _load_bob_stats(outdir: Path) -> dict:
    """Load Bob statistics from generate_suites.py output."""
    product = os.getenv("HERMES_PRODUCT", "").lower()
    candidates = [
        FRAMEWORK_DIR / "configs" / product / "bob_stats.json" if product else None,
        FRAMEWORK_DIR / "configs" / "bob_stats.json",
    ]
    for p in candidates:
        if p and p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


def build_metrics(outdir: Path, results: list[dict], gate: str,
                  agent_telemetry: list[dict], phase_timing: dict,
                  args: argparse.Namespace) -> dict:
    passed = sum(1 for r in results if r.get("status") == "passed")
    return {
        "run_id":        outdir.name,
        "started_at":    datetime.fromtimestamp(
                             phase_timing["started_ts"], tz=timezone.utc).isoformat(),
        "gate":          gate,
        "suites_total":  len(results),
        "suites_passed": passed,
        "suites_failed": len(results) - passed,
        # Per-agent breakdown
        "agents": [
            {
                "agent_id":       a["agent_id"],
                "session_id":     a["session_id"],
                "provider":       a["provider"],
                "model":          a["model"],
                "via":            a["via"],
                "duration_sec":   a["duration_sec"],
                "tool_calls":     a["tool_calls"],
                "messages_total": a["messages_total"],
                "tokens_estimate":a["tokens_estimate"],
                "rc":             a["rc"],
                "response_types": a.get("response_types", {}),
            }
            for a in agent_telemetry
        ],
        "total_agents_spawned": len(agent_telemetry),
        # Aggregate token/tool stats
        "total_tool_calls":      sum(a["tool_calls"]      for a in agent_telemetry),
        "total_tokens_estimate": sum(a["tokens_estimate"] for a in agent_telemetry),
        # Aggregate model response types
        "total_response_types": {
            "text_responses":      sum(a.get("response_types", {}).get("text_responses", 0) for a in agent_telemetry),
            "tool_call_responses": sum(a.get("response_types", {}).get("tool_call_responses", 0) for a in agent_telemetry),
            "streaming_responses": sum(a.get("response_types", {}).get("streaming_responses", 0) for a in agent_telemetry),
            "total_model_turns":   sum(a.get("response_types", {}).get("total_model_turns", 0) for a in agent_telemetry),
        },
        # Timing breakdown
        "timing": {
            "remote_prep_sec":       phase_timing.get("remote_prep_sec", 0),
            "discovery_agent_sec":   phase_timing.get("discovery_agent_sec", 0),
            "run_first_sec":         phase_timing.get("run_first_sec", 0),
            "execution_parallel_sec":phase_timing.get("execution_parallel_sec", 0),
            "retry_sec":             phase_timing.get("retry_sec", 0),
            "analyst_agent_sec":     phase_timing.get("analyst_agent_sec", 0),
            "total_supervisor_sec":  phase_timing.get("total_supervisor_sec", 0),
        },
        # Environment
        "ollama_available": check_ollama_available(),
        "remote_host":      args.remote_host or None,
        "max_workers":      args.max_workers,
        "max_retries":      args.max_retries,
        # Per-suite failure categories
        "failure_categories": {
            r["name"]: (
                "timeout"        if r["return_code"] == 124 else
                "script_missing" if r["return_code"] == 127 else
                "failed"
            )
            for r in results if r.get("status") != "passed"
        },
        "suite_durations": {r["name"]: r["duration_sec"] for r in results},
        # Bob Shell statistics (from suite generation)
        "bob_stats": _load_bob_stats(outdir),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_env_file():
    """Load .env file from framework root if it exists. Always overrides existing env vars."""
    for env_path in (FRAMEWORK_DIR / ".env", SCRIPT_DIR / ".env"):
        if env_path.exists():
            print(f"[hermes-supervisor] Loading config from {env_path}", flush=True)
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip()
                if key:
                    os.environ[key] = val
            return
    print("[hermes-supervisor] No .env file found (optional — using CLI args / env vars)", flush=True)


def _print_diagnosis(outdir: Path, knowledge) -> None:
    """Scan suite logs in outdir and print a diagnosis report using the knowledge store."""
    suite_logs = sorted(
        f for f in outdir.glob("*.log") if not f.name.startswith("hermes_")
    )
    if not suite_logs:
        return

    patterns = knowledge.data.get("failure_patterns", {}) if knowledge else {}
    recipes = knowledge.data.get("fix_recipes", {}) if knowledge else {}
    ledger = knowledge.data.get("remediation_ledger", {}) if knowledge else {}

    print(f"\n{'═' * 70}", flush=True)
    print(f"  HERMES DIAGNOSIS — {outdir.name}", flush=True)
    print(f"  Logs: {len(suite_logs)} suite(s)  Knowledge: {len(patterns)} patterns", flush=True)
    print(f"{'═' * 70}\n", flush=True)

    diagnosed = 0
    unmatched = []

    for log_file in suite_logs:
        suite_name = log_file.stem
        text = log_file.read_text(encoding="utf-8", errors="replace")

        header_lines = text.split("\n", 5)
        rc_line = [l for l in header_lines if l.startswith("exit_code=")]
        rc = int(rc_line[0].split("=", 1)[1]) if rc_line else -1
        dur_line = [l for l in header_lines if l.startswith("duration=")]
        duration = dur_line[0].split("=", 1)[1] if dur_line else "?"

        if rc == 0:
            print(f"  ✓  {suite_name:<55} PASSED  ({duration})", flush=True)
            continue

        print(f"\n  ✗  {suite_name:<55} FAILED  (rc={rc}, {duration})", flush=True)

        matches = []
        for pattern, info in patterns.items():
            check = pattern.strip('"').strip("'")
            if check.lower() in text.lower():
                matches.append((pattern, info))

        if matches:
            diagnosed += 1
            for i, (pattern, info) in enumerate(matches, 1):
                print(f"       Match {i}: {pattern[:80]}", flush=True)
                print(f"         Root cause: {info.get('root_cause', 'unknown')}", flush=True)
                print(f"         Fix:        {info.get('fix', 'no fix recorded')}", flush=True)
                recipe = recipes.get(pattern, {})
                steps = recipe.get("steps", [])
                if steps:
                    print(f"         Steps:", flush=True)
                    for s in steps:
                        print(f"           - {s}", flush=True)
                entry = ledger.get(pattern, {})
                if entry:
                    for strat, detail in entry.get("strategies", {}).items():
                        sr = detail.get("success_rate", 0)
                        att = detail.get("attempts", 0)
                        print(f"         Strategy '{strat}': {att} attempts, {sr*100:.0f}% success", flush=True)
        else:
            unmatched.append(suite_name)
            error_lines = [l.strip() for l in text.split("\n")
                           if any(kw in l.lower() for kw in
                                  ["error", "failed", "fatal", "no such file",
                                   "not found", "unauthorized", "denied", "timeout"])]
            if error_lines:
                print(f"       No known pattern matched. Key error lines:", flush=True)
                for el in error_lines[:5]:
                    print(f"         > {el[:120]}", flush=True)
            else:
                print(f"       No known pattern matched, no obvious error lines found.", flush=True)

    total_failed = diagnosed + len(unmatched)
    print(f"\n{'─' * 70}", flush=True)
    print(f"  DIAGNOSIS SUMMARY", flush=True)
    print(f"    Total suites:       {len(suite_logs)}", flush=True)
    print(f"    Passed:             {len(suite_logs) - total_failed}", flush=True)
    print(f"    Failed (diagnosed): {diagnosed}", flush=True)
    print(f"    Failed (unknown):   {len(unmatched)}", flush=True)
    if unmatched:
        print(f"    Unknown failures:   {', '.join(unmatched)}", flush=True)
    print(f"{'─' * 70}\n", flush=True)


def main() -> int:
    _load_env_file()
    _init_model_config()

    ap = argparse.ArgumentParser(
        description="Hermes Supervisor — multi-product AI test orchestration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--config",            default="configs/suites_generated.json")
    ap.add_argument("--outdir",            default=str(SCRIPT_DIR / "out" / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"))
    ap.add_argument("--remote-host",       default=_env("REMOTE_HOST", ""),      dest="remote_host")
    ap.add_argument("--remote-user",       default=_env("REMOTE_USER", "root"),  dest="remote_user")
    ap.add_argument("--remote-base",       default=_env("REMOTE_BASE", "/root/DS_Hermes_Framework"), dest="remote_base")
    ap.add_argument("--ssh-password",      default=_env("SSH_PASSWORD", ""),     dest="ssh_password")
    ap.add_argument("--cluster",           default=_env("CLUSTER_NAME", ""))
    ap.add_argument("--cpd-url",           default=os.getenv("CPD_URL", ""))
    ap.add_argument("--cpd-user",          default=os.getenv("CPD_USERNAME", "cpadmin"))
    ap.add_argument("--cpd-password",      default=os.getenv("CPD_PASSWORD", ""))
    ap.add_argument("--extra-path",        action="append", default=[], dest="extra_paths")
    ap.add_argument("--max-workers",       type=int, default=int(os.getenv("HERMES_MAX_WORKERS", "4")),  dest="max_workers")
    ap.add_argument("--max-retries",       type=int, default=int(os.getenv("HERMES_MAX_RETRIES", "1")),  dest="max_retries")
    ap.add_argument("--max-turns",         type=int, default=30, dest="max_turns",
                    help="Max tool turns per Hermes agent")
    ap.add_argument("--skip-setup",        action="store_true", dest="skip_setup",
                    help="Skip auto-setup of prerequisites (Ollama, Hermes Agent, etc.)")
    ap.add_argument("--product",           default=os.getenv("HERMES_PRODUCT", "datastage"),
                    help="Product to test (e.g. datastage, wml, wsl). Default: datastage")
    ap.add_argument("--products",          default="",
                    help="Comma-separated list of products for multi-product runs (overrides --product)")
    ap.add_argument("--cpd-namespace",     default=os.getenv("CPD_NAMESPACE", "ds"), dest="cpd_namespace",
                    help="OpenShift namespace for CPD. Default: ds")
    ap.add_argument("--execution",         default=os.getenv("HERMES_EXECUTION", "sequential"), choices=["sequential", "parallel"],
                    help="Execution mode for multi-product runs. Default: sequential")
    args = ap.parse_args()

    # Apply extra-path from env if not given on CLI
    if not args.extra_paths and os.getenv("HERMES_EXTRA_DIR"):
        args.extra_paths = [os.getenv("HERMES_EXTRA_DIR")]

    # ── Resolve product list ──────────────────────────────────────────────
    if args.products:
        product_list = [p.strip().lower() for p in args.products.split(",") if p.strip()]
    else:
        product_list = [args.product.strip().lower()]

    # ── Auto-setup: validate and install all prerequisites ──────────────
    if args.skip_setup:
        os.environ["HERMES_SKIP_SETUP"] = "1"
    if not ensure_environment():
        return 1
    if not _load_enhanced_agent():
        print("[hermes-supervisor] FATAL: Cannot load hermes_enhanced_agent.", flush=True)
        return 1

    # ── Multi-product execution ─────────────────────────────────────────
    if len(product_list) > 1:
        print(f"\n[hermes-supervisor] Multi-product run: {product_list}  execution={args.execution}", flush=True)
        all_gates = []
        if args.execution == "sequential":
            for product in product_list:
                rc = _run_product(product, args)
                all_gates.append((product, rc))
        else:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(product_list)) as pool:
                futures = {pool.submit(_run_product, p, args): p for p in product_list}
                for fut in futures:
                    product = futures[fut]
                    all_gates.append((product, fut.result()))
        print(f"\n{'═' * 70}", flush=True)
        print(f"  MULTI-PRODUCT SUMMARY", flush=True)
        for product, rc in all_gates:
            print(f"    {product:<20} {'PASS' if rc == 0 else 'FAIL'}", flush=True)
        print(f"{'═' * 70}", flush=True)
        return 0 if all(rc == 0 for _, rc in all_gates) else 1

    return _run_product(product_list[0], args)


def _run_product(product: str, args) -> int:
    """Run full Hermes orchestration for a single product."""
    base_outdir = Path(args.outdir).resolve()
    if args.products:
        outdir = base_outdir / product
    else:
        outdir = base_outdir
    outdir.mkdir(parents=True, exist_ok=True)
    started_ts  = time.time()
    phase_timing: dict[str, Any] = {"started_ts": started_ts}
    agent_telemetry: list[dict]  = []

    # ── Enhanced Agent Layer init ────────────────────────────────────────
    run_id = outdir.name
    enhanced_ctx = create_enhanced_context(outdir, run_id, product=product)
    knowledge    = enhanced_ctx["knowledge"]
    healer       = enhanced_ctx["healer"]
    notifications = enhanced_ctx["notifications"]

    print("\n" + "═" * 70, flush=True)
    print(f"  HERMES SUPERVISOR — product={product}  fast={HERMES_MODEL_FAST}  analyst={HERMES_MODEL_ANALYST}", flush=True)
    print(f"  Hermes-Native: memory_provider=DSCIMemoryProvider | "
          f"knowledge={len(knowledge.data['entries'])} skills, "
          f"{len(knowledge.data['failure_patterns'])} patterns | "
          f"self-healing={'on' if knowledge.data['failure_patterns'] else 'cold-start'} | "
          f"auto-skill-writing=on", flush=True)
    print("═" * 70 + "\n", flush=True)

    gen_script = SCRIPT_DIR / "generate_suites.py"
    product_configs_dir = FRAMEWORK_DIR / "configs" / product
    product_configs_dir.mkdir(parents=True, exist_ok=True)
    generated_suites = product_configs_dir / "suites_generated.json"
    generated_reqs_s390x = product_configs_dir / "requirements_s390x.txt"
    generated_reqs_x86   = product_configs_dir / "requirements_x86.txt"

    phase_timing["remote_prep_sec"] = round(time.time() - started_ts, 2)

    # ── AI Suite Generation (runs BEFORE bootstrap to determine lanes) ──
    _gen_done = False
    if gen_script.exists() and args.extra_paths and not args.skip_setup:
        print("\n[hermes-supervisor] ── AI SUITE GENERATION (local) ──", flush=True)
        t_gen = time.time()
        gen_env = os.environ.copy()
        gen_env["HERMES_EXTRA_DIR"] = args.extra_paths[0]
        gen_env["HERMES_PRODUCT"] = product
        cpd_url_val = gen_env.get("CPD_URL", "")
        if cpd_url_val:
            gen_env["CPD_URL_HOST"] = cpd_url_val.replace("https://", "").replace("http://", "").rstrip("/")
        local_job0 = FRAMEWORK_DIR.parent / "job0"
        if local_job0.exists() and "datastage" in product.lower():
            gen_env["HERMES_JOB0_DIR"] = str(local_job0)
        gen_rc = subprocess.run(
            [sys.executable, str(gen_script)],
            cwd=str(SCRIPT_DIR), env=gen_env, timeout=3600,
        )
        phase_timing["suite_generation_sec"] = round(time.time() - t_gen, 2)
        if gen_rc.returncode == 0 and generated_suites.exists():
            print(f"[hermes-supervisor]   ✓ AI suite generation complete ({phase_timing['suite_generation_sec']}s)", flush=True)
            _gen_done = True
        else:
            print(f"[hermes-supervisor]   ⚠ AI suite generation failed, falling back to --config", flush=True)

    # Load suites: prefer AI-generated, fall back to --config
    if generated_suites.exists():
        config_path = generated_suites
    else:
        config_path = Path(args.config).resolve()
    cfg    = json.loads(config_path.read_text(encoding="utf-8"))
    suites = cfg.get("suites", [])
    for s in suites:
        cmd = s.get("cmd", "")
        if "HEADLESS=1 " in cmd:
            cmd = cmd.replace("HEADLESS=1 ", "")
        # CPD rejects creating a git token whose name already exists: the dialog
        # stays open with Continue disabled and every later click is intercepted,
        # so the run fails minutes later looking like a UI problem. A suite that
        # does not set a token name would reuse its own hardcoded default and
        # collide on the second run. Give it a per-run name instead.
        # (Generation is asked to do this too, but a 12B model does it unreliably.)
        if ".robot" in cmd and not re.search(r'-v\s+\w*token_?name:', cmd, re.IGNORECASE):
            cmd = re.sub(r'(\S+\.robot)', r'-v TOKEN_NAME:tok$HERMES_RUN_ID \1', cmd, count=1)
        s["cmd"] = cmd
    if not suites:
        print("ERROR: no suites found in config", file=sys.stderr)
        return 2

    # Determine capabilities from suites (agent-provided, not hardcoded)
    has_vm_suites = any(not s.get("run_local") and not s.get("run_first") for s in suites)
    needs_cpdctl = any(s.get("capabilities", {}).get("needs_cpdctl") for s in suites)
    needs_selenium = any(s.get("capabilities", {}).get("needs_selenium") for s in suites)
    needs_chrome = any(s.get("capabilities", {}).get("needs_chrome") for s in suites)

    test_type = cfg.get("test_type", "unknown")
    print(f"[hermes-supervisor] config    = {config_path}", flush=True)
    print(f"[hermes-supervisor] test_type = {test_type} ({'Robot Framework suites' if test_type == 'robot' else 'Python scripts' if test_type == 'python' else 'auto-detected'})", flush=True)
    print(f"[hermes-supervisor] outdir    = {outdir}", flush=True)
    print(f"[hermes-supervisor] suites    = {len(suites)}", flush=True)
    print(f"[hermes-supervisor] vm_needed = {has_vm_suites}  cpdctl = {needs_cpdctl}  selenium = {needs_selenium}", flush=True)
    print(f"[hermes-supervisor] model     = {HERMES_MODEL}", flush=True)
    if args.remote_host:
        print(f"[hermes-supervisor] remote  = {args.remote_user}@{args.remote_host}  base={args.remote_base}", flush=True)
    else:
        print("[hermes-supervisor] remote  = none — Mac-only run", flush=True)

    # ── Step 0: BOOTSTRAP REMOTE VM (only if VM-lane suites exist) ────────
    cpd_project_ids: dict[str, str] = {}
    if args.remote_host and has_vm_suites:
        print("\n[hermes-supervisor] ── STEP 0: BOOTSTRAP REMOTE VM ─────────────────", flush=True)
        t_boot = time.time()
        _config_path = Path(args.config).resolve()
        _static_config = json.loads(_config_path.read_text(encoding="utf-8")) if _config_path.exists() else {"suites": []}
        ok = bootstrap_remote_vm(
            remote_host=args.remote_host, remote_user=args.remote_user,
            ssh_password=args.ssh_password, remote_base=args.remote_base,
            cpd_url=args.cpd_url, cpd_user=args.cpd_user,
            cpd_password=args.cpd_password, cluster=args.cluster,
            cpd_namespace=args.cpd_namespace,
            suites_config=_static_config,
            extra_paths=args.extra_paths,
            product=product,
            needs_cpdctl=needs_cpdctl,
        )
        phase_timing["bootstrap_sec"] = round(time.time() - t_boot, 2)
        if not ok:
            print("[hermes-supervisor]   ✗  Bootstrap FAILED", flush=True)
            return 3
        if isinstance(ok, dict):
            cpd_project_ids = ok
        print(f"[hermes-supervisor]   ✓  Bootstrap complete ({phase_timing['bootstrap_sec']}s)", flush=True)
    elif args.remote_host and not has_vm_suites:
        print("\n[hermes-supervisor] ── STEP 0: All suites are LOCAL — skipping VM bootstrap", flush=True)
    else:
        print("\n[hermes-supervisor] ── STEP 0: No remote host — skipping bootstrap", flush=True)

    # Install AI-generated requirements on each target
    def _read_reqs(filepath):
        if not filepath.exists():
            return []
        return [l.strip() for l in filepath.read_text().splitlines()
                if l.strip() and not l.strip().startswith("#")]

    # s390x VM: VM-lane suite packages (via SSH to --remote-host)
    s390x_reqs = _read_reqs(generated_reqs_s390x)
    if args.remote_host and s390x_reqs:
        print("\n[hermes-supervisor] ── INSTALL REQUIREMENTS (s390x VM) ─────────────", flush=True)
        _pkg_list = " ".join(s390x_reqs)
        rc_inst, _ = _ssh(args.remote_host, args.remote_user, args.ssh_password,
                          f"PIP_BREAK_SYSTEM_PACKAGES=1 python3.11 -m pip install --quiet {_pkg_list} 2>/dev/null || "
                          f"PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install --quiet {_pkg_list}", timeout=300)
        if rc_inst == 0:
            print(f"[hermes-supervisor]   ✓ s390x VM: installed {len(s390x_reqs)} packages", flush=True)
        else:
            print(f"[hermes-supervisor]   ⚠ s390x VM: some packages failed", flush=True)

    # x86 VM: LOCAL-lane suite packages (installed locally where supervisor runs)
    x86_reqs = _read_reqs(generated_reqs_x86)
    if x86_reqs:
        print("\n[hermes-supervisor] ── INSTALL REQUIREMENTS (x86 VM — local) ──────", flush=True)
        _pkg_list = " ".join(x86_reqs)
        rc_inst = subprocess.run(
            f"PIP_BREAK_SYSTEM_PACKAGES=1 {sys.executable} -m pip install --quiet {_pkg_list}",
            shell=True, capture_output=True, text=True, timeout=300)
        if rc_inst.returncode == 0:
            print(f"[hermes-supervisor]   ✓ x86 local: installed {len(x86_reqs)} packages", flush=True)
        else:
            print(f"[hermes-supervisor]   ⚠ x86 local: some packages failed", flush=True)

    # ── Step 1: Read prompt files ─────────────────────────────────────────────
    print("\n[hermes-supervisor] ── STEP 1: READ PROMPTS ───────────────────────", flush=True)
    for fname in ["hermes_bootstrap.prompt.txt", "hermes_discovery.prompt.txt",
                  "hermes_run_first.prompt.txt",
                  "hermes_executor_mac.prompt.txt", "hermes_executor_vm.prompt.txt",
                  "hermes_retry.prompt.txt", "hermes_summary.prompt.txt"]:
        p = PROMPTS_DIR / fname
        if p.exists():
            print(f"[hermes-supervisor]   ✓  {fname}  ({p.stat().st_size:,} bytes)", flush=True)
        else:
            print(f"[hermes-supervisor]   ✗  {fname}  (missing)", flush=True)

    # ── Step 2: AGENT 1 — DISCOVERY ──────────────────────────────────────────
    print("\n[hermes-supervisor] ── STEP 2: AGENT 1 — DISCOVERY ───────────────", flush=True)
    t_disc = time.time()
    manifest  = enforce_lane_split(suites, remote_host=args.remote_host,
                                   outdir=outdir, knowledge=knowledge,
                                   run_id=run_id, agent_telemetry=agent_telemetry)
    suite_map = {s["name"]: s for s in suites}
    write_discovery_artifacts(outdir, args.remote_host, args.remote_user,
                              args.remote_base, suites, manifest)
    phase_timing["discovery_agent_sec"] = round(time.time() - t_disc, 2)
    print(f"[hermes-supervisor]   ✓  suite_inventory.json, bootstrap_manifest.json, hermes_task_plan.json", flush=True)

    print(f"\n[hermes-supervisor] ── STEP 3: LANE SPLIT ─────────────────────────", flush=True)
    print(f"[hermes-supervisor]   run_first : {manifest['run_first']}", flush=True)
    print(f"[hermes-supervisor]   mac_lane  : {manifest['mac_lane']}", flush=True)
    print(f"[hermes-supervisor]   vm_lane   : {manifest['vm_lane']}", flush=True)

    vm_env_str  = build_env(args.remote_base, args.cpd_url, args.cpd_user,
                            args.cpd_password, args.cluster, args.extra_paths, is_vm=True, product=product, needs_cpdctl=needs_cpdctl)
    mac_env_str = build_env(args.remote_base, args.cpd_url, args.cpd_user,
                            args.cpd_password, args.cluster, args.extra_paths, is_vm=False, product=product, needs_cpdctl=needs_cpdctl)
    if args.ssh_password:
        mac_env_str += f"export SSH_PASSWORD='{args.ssh_password}'; "
    results: list[dict] = []

    # ── Step 4: RUN_FIRST — direct execution + AI analysis (sequential gate) ─
    print("\n[hermes-supervisor] ── STEP 4: RUN_FIRST (sequential gate) ─", flush=True)
    t_rf = time.time()
    for name in manifest["run_first"]:
        # Direct execution (reliable)
        is_vm = not suite_map[name].get("run_local", False)
        direct_result = run_suite_direct(
            suite_map[name], outdir=outdir,
            remote_host=args.remote_host, remote_user=args.remote_user,
            ssh_password=args.ssh_password,
            vm_env_str=vm_env_str, mac_env_str=mac_env_str,
            is_vm=is_vm, project_ids=cpd_project_ids,
            cpd_namespace=args.cpd_namespace,
        )

        # AI agent analyzes the result
        log_path = outdir / f"{name}.log"
        log_snippet = ""
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            log_snippet = text[-2000:] if len(text) > 2000 else text
        gate_skill_instructions = ""
        if direct_result["status"] != "passed":
            gate_skill_instructions = (
                "\n\nSince this gate test FAILED, emit a [SKILL_LEARNED] block:\n"
                "[SKILL_LEARNED]\n"
                "category: <infra|auth|ui|backend|config>\n"
                "pattern: <the error signature, max 100 chars>\n"
                "root_cause: <why it failed>\n"
                "fix: <how to fix it>\n"
                "severity: critical\n"
                "[/SKILL_LEARNED]\n"
            )
        analysis_prompt = (
            f"You are a gate-check analyst agent. Critical suite '{name}' was executed.\n\n"
            f"Log (last 2000 chars):\n```\n{log_snippet}\n```\n\n"
            f"Exit code: {direct_result['return_code']}\n"
            f"Duration: {direct_result['duration_sec']}s\n\n"
            f"Analyze whether this gate test passed. Write analysis to {outdir}/hermes_analysis_{name}.txt "
            f"then respond with the JSON below."
            f"{gate_skill_instructions}\n\n"
            f'{{"name": "{name}", "rc": {direct_result["return_code"]}, '
            f'"duration_sec": {direct_result["duration_sec"]}, '
            f'"status": "{direct_result["status"]}"}}'
        )
        rf_meta = spawn_enhanced_agent(f"run_first_{name}", analysis_prompt,
                              outdir=outdir, toolsets=["file"],
                              max_turns=5, timeout=300,
                              knowledge=knowledge, run_id=run_id,
                              role="run_first")
        agent_telemetry.append(rf_meta)
        print(f"[hermes-supervisor]   [{name}] exec={direct_result['duration_sec']}s "
              f"rc={direct_result['return_code']}  analysis={rf_meta['duration_sec']}s", flush=True)

        results.append(direct_result)
        # Auto-extract patterns from gate failure logs
        if direct_result["status"] != "passed" and log_snippet and knowledge:
            try:
                n_new = knowledge.ingest_from_log(log_snippet, name, run_id)
                if n_new:
                    print(f"[hermes-supervisor]   [{name}] +{n_new} patterns from gate log", flush=True)
            except Exception:
                pass
        notifications.on_suite_complete(name, direct_result["status"], direct_result["duration_sec"])
        # A failed critical suite fails the gate (decide_gate), but does not
        # end the run: stopping here skipped every other suite and the log
        # analysis, so one failure left nothing else to look at.
        if direct_result["status"] != "passed" and suite_map[name].get("critical"):
            print(f"[hermes-supervisor] CRITICAL {name} FAILED — gate will be FAIL, continuing", flush=True)
    phase_timing["run_first_sec"] = round(time.time() - t_rf, 2)

    # ── Step 5: PER-SUITE AGENTS — one agent per suite, all in parallel ──────
    # Each agent only handles 1 suite (3 turns: run → write log → return JSON).
    # This avoids qwen3:14b spending minutes reasoning between each suite.
    print(f"\n[hermes-supervisor] ── STEP 5: PER-SUITE AGENTS (all parallel) ─", flush=True)
    t_exec = time.time()

    mac_names = manifest["mac_lane"]
    vm_names  = manifest["vm_lane"]

    # VM preflight — fix cpdctl config before any VM suite runs
    def run_vm_preflight():
        preflight = FRAMEWORK_DIR / "scripts" / "vm_preflight.sh"
        if not preflight.exists() or not args.remote_host:
            return
        cpd_user = args.cpd_user or "cpadmin"
        cmd = (
            f"sshpass -p'{args.ssh_password}' ssh -o StrictHostKeyChecking=no "
            f"{args.remote_user}@{args.remote_host} "
            f"'CPD_URL={args.cpd_url} CPD_USERNAME={cpd_user} "
            f"CPD_PASSWORD={args.cpd_password} bash -s' < {preflight}"
        )
        print(f"[hermes-supervisor]   [vm-preflight] running vm_preflight.sh ...", flush=True)
        rc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=120)
        if rc.returncode == 0:
            print(f"[hermes-supervisor]   [vm-preflight] PASS", flush=True)
        else:
            print(f"[hermes-supervisor]   [vm-preflight] WARN (rc={rc.returncode}): {rc.stderr.strip()[:200]}", flush=True)

    run_vm_preflight()

    def _run_jenkins_job(suite: dict, log_path: Path) -> tuple:
        return _run_jenkins_job_standalone(suite, log_path)

    def run_single_suite(name: str, is_vm: bool) -> list[dict]:
        """Hybrid executor: direct execution → AI analysis."""
        agent_id = f"executor-{'vm' if is_vm else 'mac'}-{name}"

        # Jenkins-type suites: direct Python execution (trigger, poll, log)
        if suite_map[name].get("type") == "jenkins":
            print(f"[hermes-supervisor]   [jenkins] {name} — executing", flush=True)
            t0 = time.time()
            log_path = outdir / f"{name}.log"
            rc, console_tail, build_result, build_url = _run_jenkins_job(suite_map[name], log_path)
            dur = round(time.time() - t0)
            status = "passed" if rc == 0 else "failed"
            result = {"name": name, "status": status, "return_code": rc,
                      "duration_sec": dur, "critical": suite_map[name].get("critical", False),
                      "log_file": str(log_path)}
            notifications.on_suite_complete(name, status, dur)
            print(f"[hermes-supervisor]   [jenkins] {'✓' if rc == 0 else '✗'} {name}  "
                  f"rc={rc}  {dur}s  result={build_result}", flush=True)
            return [result]

        # 1. Execute directly (reliable, no AI guessing)
        direct_result = run_suite_direct(
            suite_map[name], outdir=outdir,
            remote_host=args.remote_host, remote_user=args.remote_user,
            ssh_password=args.ssh_password,
            vm_env_str=vm_env_str, mac_env_str=mac_env_str,
            is_vm=is_vm, project_ids=cpd_project_ids,
            cpd_namespace=args.cpd_namespace,
        )

        # 2. AI agent analyzes the output (reads log, classifies, writes transcript)
        log_path = outdir / f"{name}.log"
        log_snippet = ""
        if log_path.exists():
            text = log_path.read_text(encoding="utf-8", errors="replace")
            log_snippet = text[-2000:] if len(text) > 2000 else text
        skill_block_instructions = ""
        if direct_result["status"] != "passed":
            skill_block_instructions = (
                "\n\nSince this suite FAILED, emit a [SKILL_LEARNED] block so future runs can self-heal. "
                "Format:\n"
                "[SKILL_LEARNED]\n"
                "category: <infra|auth|ui|backend|config>\n"
                "pattern: <the error signature, max 100 chars>\n"
                "root_cause: <why it failed>\n"
                "fix: <how to fix it>\n"
                "severity: <critical|high|medium|low>\n"
                "[/SKILL_LEARNED]\n"
            )
        analysis_prompt = (
            f"You are a test analyst agent. Suite '{name}' was executed and produced this log:\n\n"
            f"```\n{log_snippet}\n```\n\n"
            f"Exit code: {direct_result['return_code']}\n"
            f"Duration: {direct_result['duration_sec']}s\n\n"
            f"Analyze the output. Write a brief analysis to {outdir}/hermes_analysis_{name}.txt "
            f"then respond with the JSON below."
            f"{skill_block_instructions}\n\n"
            f'[{{"name": "{name}", "rc": {direct_result["return_code"]}, '
            f'"duration_sec": {direct_result["duration_sec"]}, '
            f'"status": "{direct_result["status"]}", '
            f'"analysis": "<your 1-line summary>"}}]'
        )
        meta = spawn_enhanced_agent(agent_id, analysis_prompt,
                           outdir=outdir, toolsets=["file"],
                           max_turns=5, timeout=300,
                           knowledge=knowledge, run_id=run_id,
                           role="executor")
        agent_telemetry.append(meta)
        # Ensure analysis file exists even if agent didn't use file tool
        analysis_file = outdir / f"hermes_analysis_{name}.txt"
        if not analysis_file.exists() and meta.get("response"):
            analysis_file.write_text((meta["response"] or "")[:2000], encoding="utf-8")
        print(f"[hermes-supervisor]   [{agent_id}] exec={direct_result['duration_sec']}s "
              f"rc={direct_result['return_code']}  analysis={meta['duration_sec']}s  "
              f"tools={meta['tool_calls']}", flush=True)
        # Auto-extract error patterns from log (works even if AI doesn't emit SKILL_LEARNED)
        if direct_result["status"] != "passed" and log_snippet and knowledge:
            try:
                n_new = knowledge.ingest_from_log(log_snippet, name, run_id)
                if n_new:
                    print(f"[hermes-supervisor]   [{agent_id}] +{n_new} patterns from log", flush=True)
            except Exception:
                pass
        notifications.on_suite_complete(name, direct_result["status"], direct_result["duration_sec"])
        return [direct_result]

    # Run VM suites (all parallel in pool) and mac suites (sequential in own thread)
    # Separate thread ensures mac lane starts immediately, not queued behind VM workers
    vm_tasks  = [(n, True)  for n in vm_names]
    mac_tasks = [(n, False) for n in mac_names]
    mac_results: list[dict] = []

    def _run_mac_lane():
        for n, is_vm in mac_tasks:
            try:
                res = run_single_suite(n, is_vm)
                mac_results.extend(res)
                print(f"[hermes-supervisor]   ✓ {n} completed (mac)", flush=True)
            except Exception as e:
                print(f"[hermes-supervisor]   suite {n} exception (mac): {e}", flush=True)

    mac_thread = threading.Thread(target=_run_mac_lane, name="mac-lane")
    mac_thread.start()
    print(f"[hermes-supervisor]   mac_lane started in parallel ({len(mac_tasks)} suites, sequential)", flush=True)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        vm_futures = {pool.submit(run_single_suite, n, is_vm): n
                      for n, is_vm in vm_tasks}
        for fut in as_completed(vm_futures):
            name = vm_futures[fut]
            try:
                res = fut.result()
                results.extend(res)
                print(f"[hermes-supervisor]   ✓ {name} completed (vm)", flush=True)
            except Exception as e:
                print(f"[hermes-supervisor]   suite {name} exception (vm): {e}", flush=True)

    mac_thread.join()
    results.extend(mac_results)

    phase_timing["execution_parallel_sec"] = round(time.time() - t_exec, 2)

    # ── Step 6: RETRY failed suites (sequential direct execution + AI analysis) ─
    t_retry = time.time()
    if args.max_retries > 0:
        for retry_round in range(1, args.max_retries + 1):
            # A suite opts out with "retry": false - e.g. a many-hour Jenkins
            # job whose rerun would double the regression's length.
            # Jenkins suites default to no retry: each retry is another full
            # build on the Jenkins side.
            still_failed = [r for r in results if r["status"] != "passed"
                            and suite_map.get(r["name"], {}).get(
                                "retry", suite_map.get(r["name"], {}).get("type") != "jenkins")]
            if not still_failed:
                break
            print(f"\n[hermes-supervisor] ── STEP 6: RETRY round {retry_round}/{args.max_retries} (sequential) ─", flush=True)
            print(f"[hermes-supervisor]   retrying: {[r['name'] for r in still_failed]}", flush=True)

            retry_results_all = []
            for fail_r in still_failed:
                name = fail_r["name"]
                is_vm = name in manifest["vm_lane"] or name in manifest.get("run_first", [])
                retry_log = outdir / f"{name}.retry{retry_round}.log"

                # Self-healing: diagnose failure and get remediation recommendation
                orig_log = outdir / f"{name}.log"
                diagnosis = {"matched": False}
                log_text = ""
                if orig_log.exists():
                    log_text = orig_log.read_text(encoding="utf-8", errors="replace")
                    diagnosis = healer.diagnose_and_suggest(log_text, name)
                if diagnosis.get("matched"):
                    rec = diagnosis.get("recommendation", {})
                    strategy = healer.classify_strategy(log_text, diagnosis)
                    print(f"[hermes-supervisor]   → retry {name}  (self-healing: {diagnosis['pattern'][:60]})", flush=True)
                    print(f"[hermes-supervisor]     cause: {diagnosis.get('root_cause', '')[:80]}", flush=True)
                    if not rec.get("is_new"):
                        print(f"[hermes-supervisor]     strategy: {rec.get('strategy', '?')} "
                              f"({rec.get('success_rate', 0):.0%} success, "
                              f"confidence={rec.get('confidence', '?')}, "
                              f"{rec.get('occurrences', 0)} prior occurrences)", flush=True)
                        if rec.get("avoid"):
                            print(f"[hermes-supervisor]     avoid: {', '.join(rec['avoid'])}", flush=True)
                    else:
                        print(f"[hermes-supervisor]     strategy: {strategy} (first time — no history)", flush=True)
                else:
                    strategy = healer.classify_strategy(log_text, diagnosis)
                    if strategy == "retry_blind":
                        print(f"[hermes-supervisor]   → retry {name}  (no known pattern)", flush=True)
                    else:
                        print(f"[hermes-supervisor]   → retry {name}  (auto-classified: {strategy})", flush=True)

                # ── Pre-retry: skip deterministic failures ─────────────────
                if strategy == "skip_not_retryable":
                    print(f"[hermes-supervisor]   ✗ skipping retry for {name} — deterministic failure (not retryable)", flush=True)
                    direct_result = {"status": "failed", "return_code": fail_r.get("return_code", 1)}
                    retry_duration = 0.0
                    pattern_key = diagnosis.get("pattern", name)
                    healer.record_outcome(pattern_key, strategy, "failure",
                                          suite=name, run_id=run_id, duration_sec=0)
                    retry_results_all.append({
                        "name": name, "status": "failed",
                        "return_code": fail_r.get("return_code", 1),
                        "duration_sec": 0, "critical": fail_r.get("critical", False),
                        "log_file": str(retry_log), "retry": True,
                        "skipped_reason": "deterministic failure",
                    })
                    continue

                # ── Pre-retry: spawn healer agent to apply fixes ──────────
                # The agent reads the knowledge store (fix_recipes, auto_fix_patterns,
                # failure_patterns), figures out what to do, and executes fixes using
                # shell/ssh tools. It also updates the knowledge store with new recipes.
                knowledge_summary = ""
                if knowledge:
                    recipes = json.dumps(knowledge.data.get("fix_recipes", {}), indent=2)
                    patterns = json.dumps(knowledge.data.get("auto_fix_patterns", {}), indent=2)
                    failure_pats = json.dumps(knowledge.data.get("failure_patterns", {}), indent=2)
                    knowledge_summary = (
                        f"\n\nKnowledge Store — fix_recipes:\n```json\n{recipes}\n```\n"
                        f"\nauto_fix_patterns:\n```json\n{patterns}\n```\n"
                        f"\nfailure_patterns:\n```json\n{failure_pats}\n```\n"
                    )

                suite_cfg = suite_map.get(name, {})
                suite_config = json.dumps(suite_cfg, indent=2)
                heal_hints = suite_cfg.get("self_heal_hints", [])
                pre_retry = suite_cfg.get("retry_strategy", {}).get("pre_retry_actions", [])
                pre_actions = suite_cfg.get("pre_actions", [])

                hint_instructions = ""
                if heal_hints:
                    matched_hints = []
                    for hint in heal_hints:
                        pat = hint.get("error_pattern", "")
                        if pat and log_text and re.search(pat, log_text, re.IGNORECASE):
                            matched_hints.append(hint)
                    if matched_hints:
                        hint_instructions = (
                            "\n\nMATCHED SELF-HEAL HINTS (from suite analysis — apply these FIRST):\n"
                        )
                        for i, h in enumerate(matched_hints, 1):
                            hint_instructions += (
                                f"  {i}. Pattern: {h.get('error_pattern', '')}\n"
                                f"     Fix: {h.get('fix', '')}\n"
                                f"     Category: {h.get('category', 'unknown')}\n"
                            )
                    elif heal_hints:
                        hint_instructions = (
                            f"\n\nSuite has {len(heal_hints)} self-heal hints (none matched this error). "
                            f"Review them in the suite config for possible partial matches.\n"
                        )

                pre_action_instructions = ""
                if pre_retry or pre_actions:
                    actions = list(set(pre_retry + pre_actions))
                    pre_action_instructions = (
                        f"\n\nPRE-RETRY ACTIONS to apply before re-running:\n"
                    )
                    for act in actions:
                        if act == "dismiss_walkme_overlay" or act == "dismiss_overlays":
                            pre_action_instructions += (
                                f"  - {act}: Inject JS to remove WalkMe overlays. "
                                f"Write a helper script that runs: "
                                f"document.querySelectorAll('[id*=walkme],[class*=walkme]').forEach(e => e.remove())\n"
                            )
                        elif act == "increase_wait_timeout":
                            pre_action_instructions += (
                                f"  - {act}: Add --variable TIMEOUT:120 to the robot command\n"
                            )
                        elif act == "refresh_page":
                            pre_action_instructions += (
                                f"  - {act}: The test should refresh the page before retrying UI actions\n"
                            )
                        elif act == "re_authenticate":
                            pre_action_instructions += (
                                f"  - {act}: Re-login to CPD before retrying\n"
                            )
                        else:
                            pre_action_instructions += f"  - {act}\n"

                healer_prompt = (
                    f"You are a self-healing agent. Suite '{name}' failed with strategy='{strategy}'.\n"
                    f"Suite config:\n```json\n{suite_config}\n```\n\n"
                    f"Failure log (last 1500 chars):\n```\n{log_text[-1500:] if log_text else 'no log'}\n```\n"
                    f"{hint_instructions}"
                    f"{pre_action_instructions}"
                    f"{knowledge_summary}\n"
                    f"Your job: FIX the problem before the suite is retried.\n\n"
                    f"PRIORITY ORDER:\n"
                    f"1. Apply any MATCHED SELF-HEAL HINTS above — these are the most specific fixes\n"
                    f"2. Apply PRE-RETRY ACTIONS — these prevent known recurring issues\n"
                    f"3. Check fix_recipes in the knowledge store\n"
                    f"4. Fall back to general fixes below\n\n"
                    f"You have shell tools. Use them to apply fixes:\n"
                    f"- If element click intercepted / walkme overlay: write a JS injection script to "
                    f"remove overlay elements, and modify the robot command to run it first\n"
                    f"- If element not visible / timeout: write a JSON file {outdir}/healer_{name}.json with "
                    f'{{"action": "timeout_increase", "new_timeout_sec": <new value>, '
                    f'"extra_robot_args": "--variable TIMEOUT:120"}}. '
                    f"Use 1.5x the current timeout.\n"
                    f"- If missing module: run 'pip install <module>' using the shell tool\n"
                    f"- If chromedriver: run 'pip install webdriver-manager selenium' then install driver\n"
                    f"- If auth/token expired: run cpdctl re-auth commands\n"
                    f"- If connection error: just note it (retry may resolve)\n\n"
                    f"After applying fixes, write to {outdir}/healer_{name}.json:\n"
                    f'{{"action": "<type>", "fix_applied": "<what you did>", '
                    f'"recipe": {{"pattern": "<error signature>", "steps": ["<actionable fix step>"]}}}}\n\n'
                    f"If you applied a timeout increase, the steps should be like: "
                    f'["timeout_increase:<new_seconds>"]\n'
                    f"If you installed a package: [\"pip install <pkg>\"]\n"
                    f"Respond with a 1-line summary of what you did."
                )
                healer_toolsets = ["file"]
                healer_meta = spawn_enhanced_agent(
                    f"healer_r{retry_round}_{name}", healer_prompt,
                    outdir=outdir, toolsets=healer_toolsets,
                    max_turns=10, timeout=180,
                    knowledge=knowledge, run_id=run_id,
                    role="executor",
                )
                agent_telemetry.append(healer_meta)

                # Read healer output and apply suite config changes
                healer_file = outdir / f"healer_{name}.json"
                if healer_file.exists():
                    try:
                        healer_result = json.loads(healer_file.read_text(encoding="utf-8"))
                        action = healer_result.get("action", "")
                        if action == "timeout_increase":
                            new_t = healer_result.get("new_timeout_sec", 0)
                            if new_t:
                                suite_map[name]["timeout_sec"] = new_t
                                print(f"[hermes-supervisor]   ✓ healer: timeout → {new_t}s", flush=True)
                        # Ingest recipe into knowledge store
                        recipe = healer_result.get("recipe", {})
                        if recipe and knowledge:
                            pat = recipe.get("pattern", "")
                            steps = recipe.get("steps", [])
                            if pat and steps:
                                knowledge.add_fix_recipe(pat, steps)
                                print(f"[hermes-supervisor]   ✓ healer: recipe saved → {pat[:60]}", flush=True)
                        print(f"[hermes-supervisor]   ✓ healer fix applied: {healer_result.get('fix_applied', 'unknown')}", flush=True)
                    except (json.JSONDecodeError, KeyError) as e:
                        print(f"[hermes-supervisor]   ⚠ healer output not parseable: {e}", flush=True)
                else:
                    print(f"[hermes-supervisor]   ⚠ healer did not write {healer_file.name}", flush=True)

                # 1. Direct execution (reliable)
                t_retry_suite = time.time()
                direct_result = run_suite_direct(
                    suite_map[name], outdir=outdir,
                    remote_host=args.remote_host, remote_user=args.remote_user,
                    ssh_password=args.ssh_password,
                    vm_env_str=vm_env_str, mac_env_str=mac_env_str,
                    is_vm=is_vm, project_ids=cpd_project_ids,
                    cpd_namespace=args.cpd_namespace,
                )
                retry_duration = round(time.time() - t_retry_suite, 2)

                # Copy log to retry-specific file
                orig_log = outdir / f"{name}.log"
                if orig_log.exists():
                    retry_log.write_text(
                        f"retry_round={retry_round}\nstrategy={strategy}\n" +
                        orig_log.read_text(encoding="utf-8", errors="replace"),
                        encoding="utf-8",
                    )

                # Record remediation outcome in the ledger
                outcome = "success" if direct_result["status"] == "passed" else "failure"
                pattern_key = diagnosis.get("pattern", name)
                healer.record_outcome(
                    pattern_key, strategy, outcome,
                    suite=name, run_id=run_id, duration_sec=retry_duration,
                )
                print(f"[hermes-supervisor]   [{name}] remediation: {strategy} → {outcome} "
                      f"(recorded in ledger)", flush=True)

                # 2. AI agent analyzes the retry output
                log_snippet = ""
                if orig_log.exists():
                    text = orig_log.read_text(encoding="utf-8", errors="replace")
                    log_snippet = text[-2000:] if len(text) > 2000 else text

                retry_skill_instructions = ""
                if direct_result["status"] != "passed":
                    retry_skill_instructions = (
                        "\n\nSince this retry FAILED, emit a [SKILL_LEARNED] block:\n"
                        "[SKILL_LEARNED]\n"
                        "category: <infra|auth|ui|backend|config>\n"
                        "pattern: <the error signature, max 100 chars>\n"
                        "root_cause: <why the retry still failed>\n"
                        "fix: <what should be tried differently>\n"
                        "severity: <critical|high|medium|low>\n"
                        "[/SKILL_LEARNED]\n"
                    )
                analysis_prompt = (
                    f"Suite '{name}' was retried (round {retry_round}, strategy={strategy}). Log:\n\n"
                    f"```\n{log_snippet}\n```\n\n"
                    f"Exit code: {direct_result['return_code']}, Duration: {direct_result['duration_sec']}s\n"
                    f"Analyze whether the retry fixed the issue. Write analysis to "
                    f"{outdir}/hermes_analysis_{name}_retry{retry_round}.txt\n"
                    f"Then respond with the JSON below."
                    f"{retry_skill_instructions}\n\n"
                    f'[{{"name": "{name}", "rc": {direct_result["return_code"]}, '
                    f'"duration_sec": {direct_result["duration_sec"]}, '
                    f'"status": "{direct_result["status"]}", "retry": true, '
                    f'"strategy": "{strategy}", "strategy_outcome": "{outcome}", '
                    f'"analysis": "<1-line summary>"}}]'
                )
                # Inject remediation history context
                analysis_prompt = healer.build_retry_prompt(analysis_prompt, diagnosis)
                retry_meta = spawn_enhanced_agent(f"retry_r{retry_round}_{name}", analysis_prompt,
                                         outdir=outdir, toolsets=["file"],
                                         max_turns=5, timeout=300,
                                         knowledge=knowledge, run_id=run_id,
                                         role="retry")
                agent_telemetry.append(retry_meta)

                # Write analysis if agent didn't use file tool
                analysis_file = outdir / f"hermes_analysis_{name}_retry{retry_round}.txt"
                if not analysis_file.exists():
                    analysis_file.write_text((retry_meta["response"] or "")[:2000], encoding="utf-8")

                print(f"[hermes-supervisor]   {'✓' if direct_result['return_code'] == 0 else '✗'} "
                      f"retry {name}  rc={direct_result['return_code']}  {direct_result['duration_sec']}s", flush=True)

                # Update results
                retry_results_all.append(direct_result)

            for rr in retry_results_all:
                results = [rr if x["name"] == rr["name"] else x for x in results]
    phase_timing["retry_sec"] = round(time.time() - t_retry, 2)

    # ── Step 7: Write results + gate ──────────────────────────────────────────
    (outdir / "hermes_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    gate = decide_gate(results)
    (outdir / "hermes_gate.txt").write_text(gate, encoding="utf-8")
    print(f"\n[hermes-supervisor] gate = {gate}", flush=True)

    # ── Step 8: AGENT 4 — ANALYST ─────────────────────────────────────────────
    print("\n[hermes-supervisor] ── STEP 8: AGENT 4 — ANALYST ─────────────────", flush=True)
    t_analyst = time.time()
    analyst_prompt = render_summary_prompt(outdir, results, agent_telemetry=agent_telemetry)
    analyst_meta   = spawn_enhanced_agent("analyst", analyst_prompt,
                                 outdir=outdir, toolsets=["file"],
                                 max_turns=args.max_turns,
                                 knowledge=knowledge, run_id=run_id,
                                 role="analyst")
    agent_telemetry.append(analyst_meta)
    phase_timing["analyst_agent_sec"] = round(time.time() - t_analyst, 2)

    # If the analyst agent didn't write hermes_analysis.txt, save its response
    analysis_path = outdir / "hermes_analysis.txt"
    if not analysis_path.exists() and analyst_meta.get("response"):
        analysis_path.write_text(analyst_meta["response"], encoding="utf-8")
        print("[hermes-supervisor]   fallback: wrote analyst response → hermes_analysis.txt", flush=True)

    # ── Post-run: ingest new knowledge + evolve remediation strategies ───
    post_run_knowledge_update(knowledge, analysis_path, run_id, notifications)
    knowledge.evolve_strategies()

    # ── Inline diagnosis: match failures against knowledge store ───────────
    _print_diagnosis(outdir, knowledge)

    # ── Log analyser: every failed suite, any product ──────────────────────
    _run_log_analyser(outdir, results)

    # ── Metrics ────────────────────────────────────────────────────────────────
    elapsed = round(time.time() - started_ts)
    phase_timing["total_supervisor_sec"] = elapsed
    metrics = build_metrics(outdir, results, gate, agent_telemetry, phase_timing, args)
    (outdir / "hermes_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # ── Artifact index ────────────────────────────────────────────────────────
    print("\n[hermes-supervisor] Artifacts:", flush=True)
    for fname in [
        "suite_inventory.json", "bootstrap_manifest.json",
        "hermes_task_plan.json", "hermes_results.json",
        "hermes_gate.txt", "hermes_analysis.txt",
        "hermes_discovery_transcript.txt",
        "hermes_run_first_job0_test_transcript.txt",
        "hermes_executor-mac_transcript.txt",
        "hermes_executor-vm_transcript.txt",
        "hermes_retry_r1_transcript.txt",
        "hermes_analyst_transcript.txt",
        "hermes_metrics.json",
        "hermes_dashboard.html",
        "hermes_notification.txt",
        "hermes_events.json",
    ]:
        p = outdir / fname
        if p.exists():
            print(f"  ✓  {fname:<50} ({p.stat().st_size:,} bytes)", flush=True)
        else:
            print(f"  ✗  {fname:<50} (not written)", flush=True)

    # Agent summary
    # Note: tool_calls counts OpenAI function-call blocks. qwen3 emits natural-language
    # tool invocations that Hermes parses and executes — these show as 0 here but
    # real execution is evidenced by suite log files written under outdir.
    suites_with_logs = len([f for f in outdir.glob("*.log")
                             if not f.name.startswith("hermes_")])
    print("\n[hermes-supervisor] Agent telemetry:", flush=True)
    print(f"  {'agent':<35} {'model':<12} {'via':<14} {'dur(s)':>7} {'tools':>6} {'prompt_tok':>10} {'compl_tok':>10} {'total_tok':>10}", flush=True)
    print(f"  {'-'*35} {'-'*12} {'-'*14} {'-'*7} {'-'*6} {'-'*10} {'-'*10} {'-'*10}", flush=True)
    for a in agent_telemetry:
        print(f"  {a['agent_id']:<35} {a.get('model',''):<12} {a['via']:<14} {a['duration_sec']:>7.1f} "
              f"{a['tool_calls']:>6} {a.get('prompt_tokens',0):>10} "
              f"{a.get('completion_tokens',0):>10} {a['tokens_estimate']:>10}", flush=True)
    print(f"  {'TOTAL':<35} {'':<12} {'':<14} {sum(a['duration_sec'] for a in agent_telemetry):>7.1f} "
          f"{sum(a['tool_calls'] for a in agent_telemetry):>6} "
          f"{sum(a.get('prompt_tokens',0) for a in agent_telemetry):>10} "
          f"{sum(a.get('completion_tokens',0) for a in agent_telemetry):>10} "
          f"{sum(a['tokens_estimate'] for a in agent_telemetry):>10}", flush=True)
    print(f"  total_agents_spawned = {len(agent_telemetry)}", flush=True)
    print(f"  suites_executed      = {suites_with_logs}  "
          f"(suite log files written — real execution evidence)", flush=True)

    # Bob Shell statistics
    bob_stats = _load_bob_stats(outdir)
    if bob_stats and bob_stats.get("calls", 0) > 0:
        print(f"\n[hermes-supervisor] IBM Bob Statistics:", flush=True)
        print(f"  Total calls:    {bob_stats['calls']}", flush=True)
        print(f"  Successful:     {bob_stats.get('success', 0)}", flush=True)
        print(f"  Errors:         {bob_stats.get('errors', 0)}", flush=True)
        print(f"  Timeouts:       {bob_stats.get('timeouts', 0)}", flush=True)
        print(f"  Setup time:     {bob_stats.get('setup_time_sec', 0)}s", flush=True)
        print(f"  Total API time: {round(bob_stats.get('total_time_sec', 0), 2)}s", flush=True)
        jobs = bob_stats.get("jobs_analyzed", [])
        if jobs:
            avg = round(bob_stats["total_time_sec"] / len(jobs), 2)
            print(f"  Avg per job:    {avg}s", flush=True)
            for j in jobs:
                print(f"    ✓ {j['job']} ({j['time_sec']}s)", flush=True)
        errors = bob_stats.get("error_details", [])
        if errors:
            for e in errors:
                print(f"    ✗ {e.get('job', '?')}: {e.get('error', '?')}", flush=True)

    # ── Run-complete notification + event log ───────────────────────────
    passed_count = sum(1 for r in results if r["status"] == "passed")
    failed_count = len(results) - passed_count
    notifications.on_run_complete(gate, passed_count, failed_count, elapsed)
    notifications.save_event_log()

    # ── Generate HTML dashboard if generator is available ─────────────
    try:
        from generate_html_report import build_html
        html = build_html(str(outdir))
        (outdir / "hermes_dashboard.html").write_text(html, encoding="utf-8")
        print(f"  ✓  hermes_dashboard.html (auto-generated)", flush=True)
    except Exception as html_err:
        print(f"  ✗  hermes_dashboard.html (skipped: {html_err})", flush=True)

    # Clean up leaked temp files from agent spawner
    for pattern in ("_prompt_*.txt", "_runner_*.py"):
        for tmp in outdir.glob(pattern):
            try:
                tmp.unlink()
            except OSError:
                pass

    print(f"\n[hermes-supervisor] DONE — product={product}  gate={gate}  elapsed≈{elapsed}s", flush=True)
    print(f"[hermes-supervisor] Knowledge ({product}): {knowledge.summary()}", flush=True)
    return 0 if gate in {"PASS", "PASS_WITH_RISK"} else 1


if __name__ == "__main__":
    sys.exit(main())
