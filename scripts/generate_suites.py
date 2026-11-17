#!/usr/bin/env python3
"""
generate_suites.py — AI-Agent-powered suite & requirements discovery.

Uses Bob AI to analyze test directories and produce:
  1. configs/suites.json               — suite definitions
  2. configs/bootstrap_requirements.txt — pip deps for bootstrap
"""

import json
import os
import re
import shutil
import sys
import time
import subprocess
from pathlib import Path
from hermes_security import get_sanitizer, load_env_secure

SCRIPT_DIR    = Path(__file__).resolve().parent
FRAMEWORK_DIR = SCRIPT_DIR.parent
HERMES_PRODUCT = os.getenv("HERMES_PRODUCT", "").lower()
CONFIGS_DIR   = FRAMEWORK_DIR / "configs" / HERMES_PRODUCT if HERMES_PRODUCT else FRAMEWORK_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
OLLAMA_MODEL  = os.getenv("HERMES_MODEL_FAST", "gemma4:12b")
OLLAMA_URL    = os.getenv("OLLAMA_HOST", "http://localhost:11434")
_extra = os.getenv("HERMES_EXTRA_DIR")
if not _extra:
    print("[generate_suites] ERROR: HERMES_EXTRA_DIR not set. Pass --extra-path to the supervisor.", file=sys.stderr)
    sys.exit(1)
TEST_ROOT = Path(_extra)


def _load_env_vars():
    """Read env vars: prefer encrypted .env.enc, fall back to plaintext .env."""
    return load_env_secure()

_ENV_VARS = _load_env_vars()
BOB_API_KEY   = os.getenv("BOB_API_KEY", "") or _ENV_VARS.get("BOB_API_KEY", "")
# Exported by run_hermes_supervisor.build_env rather than living in .env.
_DERIVED_ENV = {"CPD_URL_HOST", "CLUSTER_NAME", "CLUSTER_PASS", "HERMES_RUN_ID"}

# Robot variables whose .env counterpart is named differently. Without these the
# matcher returns None and the -v flag is silently dropped, which is how
# pythonclient_test lost CLUSTER_NAME/CLUSTER_PASS and ended up SSH-ing with an
# empty password (the suite then types the literal string "None" at login).
_ENV_ALIASES = {
    "CLUSTER_NAME": "CLUSTER_NAME",       # derived, exported by the supervisor
    "CLUSTER_PASS": "CLUSTER_PASS",       # derived, from SSH_PASSWORD
    "CLUSTER_USER": "REMOTE_USER",
    # Robot's ${url} is a bare host (resource.robot prepends https:// itself).
    # Without this alias the GIT_ prefix is tried first and ${url} resolves to
    # GIT_URL — pointing the CPD login at the git repo.
    "URL": "CPD_URL_HOST",
    "CPD_HOST": "CPD_URL_HOST",
    "BRANCH_NAME": "GIT_BRANCH",
    "DOMAIN_URL": "GIT_DOMAIN",
    "GIT_REPO": "GIT_URL",
    "REPO_USERNAME": "GIT_REPO_USERNAME",
}


def _match_env_var(robot_var):
    """Find the best matching .env variable for a robot variable name."""
    upper = robot_var.upper()
    alias = _ENV_ALIASES.get(upper)
    if alias:
        # Accept the legacy HERMES_-prefixed spelling if the plain name is absent.
        for cand in (alias, "HERMES_" + alias):
            if cand in _ENV_VARS or cand in _DERIVED_ENV:
                return cand
    if upper in _ENV_VARS or upper in _DERIVED_ENV:
        return upper
    for prefix in ("GIT_", "CPD_", "DB2_"):
        cand = prefix + upper
        if cand in _ENV_VARS:
            return cand
    if upper.startswith("GITHUB"):
        alt = "GIT" + upper[6:]
        if alt in _ENV_VARS:
            return alt
    if upper == "CPD_HOST":
        return "CPD_URL_HOST"
    return None


def ollama_generate(prompt, timeout=120, fmt="json"):
    """Use /api/generate for fast single-turn responses.
    Sanitizes the prompt to scrub secrets before sending to the model."""
    sanitizer = get_sanitizer(_ENV_VARS)
    original_len = len(prompt)
    prompt = sanitizer.sanitize(prompt)
    scrubbed = original_len - len(prompt)
    if scrubbed:
        sanitizer.audit_log("ollama_generate", original_len, scrubbed)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        # repeat_penalty guards against degenerate loops: at temperature 0.1 the
        # decoder is near-greedy, and highly self-similar targets (long runs of
        # "-v name:$VAR") can lock into repeating a single token until num_predict.
        "options": {
            "temperature": 0.1,
            "num_predict": 2048,
            "num_ctx": 16384,
            "repeat_penalty": 1.3,
            "repeat_last_n": 256,
        },
    }
    if fmt:
        payload["format"] = fmt
    try:
        import urllib.request
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode()).get("response", "")
    except Exception as e:
        return f"ERROR: {e}"


_BOB_READY = None  # cached result of _ensure_bob_ready()
_bob_stats = {
    "calls": 0, "success": 0, "errors": 0, "timeouts": 0,
    "total_time_sec": 0.0, "setup_time_sec": 0.0,
    "jobs_analyzed": [], "error_details": [],
}


def _ensure_bob_ready():
    """Auto-install Bob Shell CLI, check Node.js, accept license. Returns True if ready."""
    global _BOB_READY
    if _BOB_READY is not None:
        return _BOB_READY
    _BOB_READY = False
    setup_start = time.time()

    # 1. Check Node.js >= 22.15
    def _install_node():
        """Install Node.js 22 using the platform-appropriate method."""
        import platform
        if platform.system() == "Darwin":
            r = subprocess.run(["brew", "install", "node@22"], capture_output=True, text=True, timeout=300)
            if r.returncode != 0:
                print(f"  [Bob] WARN: brew install node@22 failed: {r.stderr[:200]}", flush=True)
                return False
            subprocess.run(["brew", "link", "--overwrite", "node@22"], capture_output=True, text=True, timeout=60)
            print("  [Bob] Node.js 22 installed.", flush=True)
            return True
        arch = platform.machine()
        node_arch = "x64" if arch == "x86_64" else arch
        # Strategy 1: dnf module (RHEL 8/9)
        if shutil.which("dnf"):
            r = subprocess.run(
                "dnf module enable -y nodejs:22 2>/dev/null && dnf install -y nodejs",
                shell=True, capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                print("  [Bob] Node.js 22 installed via dnf module.", flush=True)
                return True
            print(f"  [Bob] dnf module failed, trying binary download...", flush=True)
        # Strategy 2: direct binary download (no repo needed)
        r = subprocess.run(
            f"curl -fsSL https://nodejs.org/dist/v22.16.0/node-v22.16.0-linux-{node_arch}.tar.xz"
            f" | tar -xJ --strip-components=1 -C /usr/local",
            shell=True, capture_output=True, text=True, timeout=180,
        )
        if r.returncode == 0:
            print("  [Bob] Node.js 22 installed via binary download.", flush=True)
            return True
        print(f"  [Bob] WARN: binary download also failed: {r.stderr[:200]}", flush=True)
        return False

    try:
        node_ver = subprocess.run(
            ["node", "-v"], capture_output=True, text=True, timeout=10
        ).stdout.strip().lstrip("v")
        parts = [int(x) for x in node_ver.split(".")[:2]]
        if parts[0] < 22 or (parts[0] == 22 and parts[1] < 15):
            print(f"  [Bob] Node.js {node_ver} too old (need 22.15+). Installing node@22...", flush=True)
            if not _install_node():
                return False
    except FileNotFoundError:
        print("  [Bob] WARN: node not found. Installing node@22...", flush=True)
        if not _install_node():
            return False
    except Exception as e:
        print(f"  [Bob] WARN: Node.js check failed: {e}", flush=True)
        return False

    # 2. Check if bob CLI exists; install if missing
    bob_path = shutil.which("bob")
    if not bob_path:
        print("  [Bob] bob CLI not found. Installing Bob Shell...", flush=True)
        try:
            dl = subprocess.run(
                ["curl", "-fsSL", "https://bob.example.com/download/bobshell.sh"],
                capture_output=True, text=True, timeout=60
            )
            if dl.returncode != 0 or not dl.stdout.strip():
                print(f"  [Bob] WARN: failed to download bobshell.sh", flush=True)
                return False
            inst = subprocess.run(
                ["bash"], input=dl.stdout, capture_output=True, text=True, timeout=120
            )
            if inst.returncode != 0:
                print(f"  [Bob] WARN: install failed: {inst.stderr[:200]}", flush=True)
                return False
            print("  [Bob] Bob Shell installed.", flush=True)
        except Exception as e:
            print(f"  [Bob] WARN: install error: {e}", flush=True)
            return False

    # 3. Accept license (idempotent)
    try:
        subprocess.run(
            ["bob", "--accept-license"],
            capture_output=True, text=True, timeout=30,
            env={**os.environ, "BOBSHELL_API_KEY": BOB_API_KEY},
        )
    except Exception:
        pass

    _BOB_READY = True
    _bob_stats["setup_time_sec"] = round(time.time() - setup_start, 2)
    print(f"  [Bob] Ready. (setup took {_bob_stats['setup_time_sec']}s)", flush=True)
    return True


def bob_generate(prompt, timeout=120, job_name=""):
    """Use Bob Shell CLI for AI generation (IBM Bob API key required)."""
    _bob_stats["calls"] += 1
    if not _ensure_bob_ready():
        _bob_stats["errors"] += 1
        _bob_stats["error_details"].append({"job": job_name, "error": "setup failed"})
        return "ERROR: Bob Shell setup failed"
    sanitizer = get_sanitizer(_ENV_VARS)
    prompt = sanitizer.sanitize(prompt)
    t0 = time.time()
    try:
        bob_key = BOB_API_KEY.strip().strip("'\"")
        proc = subprocess.run(
            ["bob", "run", "--format", "json", "--accept-license",
             "--disable-mcp", "--disable-subagents", "--max-turns", "1",
             prompt],
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "BOBSHELL_API_KEY": bob_key},
        )
        elapsed = round(time.time() - t0, 2)
        _bob_stats["total_time_sec"] += elapsed
        output = proc.stdout.strip()
        stderr = proc.stderr.strip()
        if stderr:
            print(f"    [Bob] stderr: {stderr[:200]}", flush=True)
        if not output:
            _bob_stats["errors"] += 1
            _bob_stats["error_details"].append({"job": job_name, "error": f"empty output rc={proc.returncode}", "time_sec": elapsed})
            return "ERROR: Bob returned empty response"

        def _strip_fences(s):
            """Strip markdown code fences (```json ... ```) from a string."""
            s = s.strip()
            s = re.sub(r'^```(?:json)?\s*\n?', '', s)
            s = re.sub(r'\n?```\s*$', '', s)
            return s.strip()

        def _extract_json_obj(s):
            """Try to extract a JSON object from text."""
            s = _strip_fences(s)
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return json.dumps(parsed)
            except (json.JSONDecodeError, TypeError):
                pass
            m = re.search(r'\{[^{}]*"description"[^{}]*\}', s, re.DOTALL)
            if not m:
                m = re.search(r'\{.*\}', s, re.DOTALL)
            if m:
                try:
                    json.loads(m.group())
                    return m.group()
                except (json.JSONDecodeError, TypeError):
                    pass
            return None

        # Try parsing Bob's JSON wrapper format
        try:
            result = json.loads(output)
            # Bob wraps as {"last_message": "```json\n{...}\n```", "status": "success", ...}
            inner = (result.get("last_message") or result.get("response")
                     or result.get("text") or result.get("output")
                     or result.get("message", ""))
            if isinstance(inner, str) and inner:
                extracted = _extract_json_obj(inner)
                if extracted:
                    _bob_stats["success"] += 1
                    _bob_stats["jobs_analyzed"].append({"job": job_name, "time_sec": elapsed})
                    return extracted
                _bob_stats["success"] += 1
                _bob_stats["jobs_analyzed"].append({"job": job_name, "time_sec": elapsed})
                return _strip_fences(inner)
            elif isinstance(inner, dict):
                _bob_stats["success"] += 1
                _bob_stats["jobs_analyzed"].append({"job": job_name, "time_sec": elapsed})
                return json.dumps(inner)
            # Maybe the whole result IS the analysis (no wrapper)
            if any(k in result for k in ("description", "tags", "critical")):
                _bob_stats["success"] += 1
                _bob_stats["jobs_analyzed"].append({"job": job_name, "time_sec": elapsed})
                return output
            _bob_stats["success"] += 1
            _bob_stats["jobs_analyzed"].append({"job": job_name, "time_sec": elapsed})
            return output
        except json.JSONDecodeError:
            extracted = _extract_json_obj(output)
            if extracted:
                _bob_stats["success"] += 1
                _bob_stats["jobs_analyzed"].append({"job": job_name, "time_sec": elapsed})
                return extracted
            _bob_stats["errors"] += 1
            _bob_stats["error_details"].append({"job": job_name, "error": f"unparseable: {output[:100]}", "time_sec": elapsed})
            return output
    except subprocess.TimeoutExpired:
        elapsed = round(time.time() - t0, 2)
        _bob_stats["total_time_sec"] += elapsed
        _bob_stats["timeouts"] += 1
        _bob_stats["error_details"].append({"job": job_name, "error": "timeout", "time_sec": elapsed})
        return "ERROR: Bob Shell timed out"
    except FileNotFoundError:
        _bob_stats["errors"] += 1
        _bob_stats["error_details"].append({"job": job_name, "error": "bob CLI not found"})
        return "ERROR: bob CLI not found after install attempt"
    except Exception as e:
        elapsed = round(time.time() - t0, 2)
        _bob_stats["total_time_sec"] += elapsed
        _bob_stats["errors"] += 1
        _bob_stats["error_details"].append({"job": job_name, "error": str(e)[:100], "time_sec": elapsed})
        return f"ERROR: {e}"



def validate_suite_cmds(suites):
    """Catch model-degenerated shell commands before they reach a config file.

    format="json" only guarantees the *envelope* parses; it says nothing about
    whether a string value is a runnable shell command. A repetition loop that
    emits "-v-v-v..." until num_predict, or a leaked markdown fence, still
    produces valid JSON containing an unrunnable command.
    """
    problems = []
    for suite in suites:
        if suite.get("type") == "jenkins":
            continue
        name = suite.get("name", "<unnamed>")
        cmd = suite.get("cmd", "")
        if not cmd:
            problems.append((name, "empty cmd"))
            continue
        if "`" in cmd:
            problems.append((name, f"contains {cmd.count('`')} backtick(s) — likely a leaked markdown fence"))
            continue
        if re.search(r'(-v){3,}', cmd) or re.search(r'(.{2,12}?)\1{9,}', cmd):
            problems.append((name, "repeated-token run — model degeneration"))
            continue
        # Ask bash whether it parses, without executing it. Most cmds are
        # wrapped as: bash -c '<real command>'. To bash the wrapper is a single
        # quoted string, so an unbalanced quote or $( inside it parses fine —
        # unwrap and check the inner command too, or the check is cosmetic.
        candidates = [cmd]
        inner = re.match(r"^bash -c '(.*)'\s*$", cmd, re.S)
        if inner:
            candidates.append(inner.group(1))
        failed = False
        for variant in candidates:
            try:
                proc = subprocess.run(["bash", "-n", "-c", variant],
                                      capture_output=True, text=True, timeout=10)
                if proc.returncode != 0:
                    problems.append((name, f"bash -n: {proc.stderr.strip().splitlines()[-1][:80]}"))
                    failed = True
                    break
            except Exception as e:
                problems.append((name, f"could not validate: {e}"))
                failed = True
                break
        if failed:
            continue

        # A literal hostname means the command targets something other than the
        # cluster under test — almost always a stale default copied from a suite.
        for host in re.findall(r'https?://([A-Za-z0-9.\-]+)', cmd):
            if not host.startswith('$') and '.' in host and 'github' not in host:
                problems.append((name, f"hardcoded host '{host}' — should use $CPD_URL_HOST"))
                break

        # The .robot file named in the command must actually exist — a misspelled
        # filename makes the whole suite unrunnable and is easy for a model to emit.
        m = re.search(r'([\w.\-]+\.robot)', cmd)
        dirpath = suite.get("path") or suite.get("dir") or ""
        if m and dirpath:
            target = Path(dirpath) / m.group(1)
            if not target.exists():
                problems.append((name, f"references missing file '{m.group(1)}'"))
                continue
            # A suite directory often holds both the test file and a *_resource.robot
            # of shared keywords. Running the resource gives
            # "Suite '...' contains no tests or tasks" and Robot exits 252.
            try:
                if "*** Test Cases ***" not in target.read_text(errors="replace"):
                    problems.append((name, f"'{m.group(1)}' has no *** Test Cases *** "
                                           f"— it is a resource file, not a suite"))
                    continue
            except Exception:
                pass

        # -v flags must name a variable the suite (or a resource it imports)
        # declares, else Robot ignores them silently and the variable keeps its
        # empty default — no error, and a confusing failure much later.
        declared = _declared_variables(dirpath, m.group(1) if m else None)
        passed = {v.upper() for v in re.findall(r'-v\s+(\w+):', cmd)}
        if declared:
            unknown = sorted(p for p in passed if p not in declared)
            if unknown:
                problems.append((name, f"-v names not declared by the suite: {', '.join(unknown[:4])}"))

        # A variable with no default that DOES have an environment counterpart is
        # a credential the suite needs. Omitting it is what produced
        # "InvalidURL: 'https:///icp4d-api/...': No host supplied".
        # Variables with no env match (optional gitlab/bitbucket tokens) are skipped.
        if m and dirpath:
            resolvable = {rv for rv in _empty_default_variables(dirpath, m.group(1))
                          if _match_env_var(rv)}
            missing = sorted(rv for rv in resolvable if rv.upper() not in passed)
            if missing:
                problems.append((name, f"required variables not passed: {', '.join(missing[:5])}"))
    return problems


def _empty_default_variables(dirpath, robot_filename, _depth=0):
    """Variables with no usable default, following Resource imports.

    ${url}/${username}/${password} are declared in resource.robot, not in the
    suite files, so a scan limited to the suite's own Variables table never sees
    them and the generated command silently omits the credentials.
    """
    if not (robot_filename and dirpath) or _depth > 3:
        return set()
    path = Path(dirpath) / robot_filename
    if not path.exists():
        return set()
    text = path.read_text(errors="replace")

    names = set()
    if "*** Variables ***" in text:
        section = text.split("*** Variables ***", 1)[1].split("***")[0]
        names |= {m.group(1) for m in re.finditer(
            r'^\$\{(\w+)\}[ \t]*(?:\$\{EMPTY\})?[ \t]*(?:#.*)?$', section, re.MULTILINE)}
    for m in re.finditer(r'^Resource\s+(\S+)', text, re.MULTILINE | re.IGNORECASE):
        names |= _empty_default_variables(str(path.parent), m.group(1), _depth + 1)
    return names


def _declared_variables(dirpath, robot_filename, _depth=0):
    """Variable names a suite declares, including those from Resource imports.

    Suites routinely rely on ${url}/${username}/${password} declared in
    resource.robot, so a check that reads only the suite file reports false
    positives. Returns an empty set when nothing can be read, so validation
    degrades to the syntax checks rather than inventing problems.
    """
    if not (robot_filename and dirpath) or _depth > 3:
        return set()
    path = Path(dirpath) / robot_filename
    if not path.exists():
        return set()
    text = path.read_text(errors="replace")

    names = set()
    if "*** Variables ***" in text:
        section = text.split("*** Variables ***", 1)[1].split("***")[0]
        names |= {m.group(1).upper() for m in re.finditer(r'^\$\{(\w+)\}', section, re.MULTILINE)}

    # Follow Resource imports (one hop is normally enough; depth-capped anyway).
    for m in re.finditer(r'^Resource\s+(\S+)', text, re.MULTILINE | re.IGNORECASE):
        names |= _declared_variables(str(path.parent), m.group(1), _depth + 1)
    return names


def extract_json(text):
    text = text.strip()
    # Direct parse first (works when format=json forces clean output)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip markdown code fences
    m2 = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', text)
    if m2:
        try:
            return json.loads(m2.group(1).strip())
        except json.JSONDecodeError:
            pass
    # Find first { ... } block (greedy to handle nested braces)
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    start = -1
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Discovery — no AI needed, pure filesystem scan
# ═════════════════════════════════════════════════════════════════════════════

BROWSER_OVERRIDES = set()


def _matches_product(name):
    """Check if a directory/file name is relevant to HERMES_PRODUCT."""
    if not HERMES_PRODUCT:
        return True
    name_lower = name.lower().replace("-", "").replace("_", "")
    product_lower = HERMES_PRODUCT.replace("-", "").replace("_", "")
    return product_lower in name_lower


def _find_test_files(directory):
    """Find all test files (.py, .robot, and .sh) in a directory."""
    py = list(directory.rglob("*.py"))
    robot = [f for f in directory.rglob("*.robot") if f.name != "resource.robot"]
    sh = list(directory.rglob("*.sh"))
    return py + robot + sh


def _find_requirements(directory):
    """Return the actual requirements filename (case-sensitive) or None."""
    for f in directory.iterdir():
        if f.is_file() and f.name.lower() == "requirements.txt":
            return f
    return None

def _has_requirements(directory):
    """Check for requirements file (any case)."""
    return _find_requirements(directory) is not None


_SKIP = {"__pycache__", "venv", ".venv", "source_env.sh"}
_SUB_SKIP = {"__pycache__", "venv", ".venv", "Resources"}


def _collect_dir(directory, name_prefix, parent_reqs, results):
    """Recursively collect test candidates from a directory.

    If the dir has .robot test files → one candidate per .robot file.
    If the dir has NO direct .py files but has child dirs → recurse
      (it's a container like Utility/ or DS_Performance_Tests/).
    Otherwise → one candidate for the entire directory.
    """
    files = _find_test_files(directory)
    if not files:
        return

    robot_files = [f for f in files if f.suffix == ".robot"]
    resource_files = [f for f in robot_files if "resource" in f.name.lower()]
    test_robots = [f for f in robot_files if f not in resource_files]

    has_reqs = _has_requirements(directory) or parent_reqs

    if test_robots:
        res_names = [str(f.relative_to(directory)) for f in resource_files]
        for rf in test_robots:
            results.append({
                "path": str(directory),
                "name": f"{name_prefix}/{rf.stem}",
                "suite_dir": name_prefix,
                "is_dir": True,
                "files": [str(rf.relative_to(directory))] + res_names,
                "has_reqs": has_reqs,
            })
        return

    sh_files = [f for f in files if f.suffix == ".sh"]
    if sh_files:
        for sf in sh_files:
            results.append({
                "path": str(directory),
                "name": f"{name_prefix}/{sf.stem}",
                "suite_dir": name_prefix,
                "is_dir": True,
                "files": [str(sf.relative_to(directory))],
                "has_reqs": has_reqs,
            })
        return

    direct_scripts = list(directory.glob("*.py")) + list(directory.glob("*.sh"))
    children = [d for d in sorted(directory.iterdir())
                if d.is_dir() and not d.name.startswith(".")
                and d.name not in _SUB_SKIP and d.name not in _SKIP
                and _find_test_files(d)]

    if children and not direct_scripts:
        for child in children:
            _collect_dir(child, f"{name_prefix}/{child.name}",
                         has_reqs, results)
    else:
        results.append({
            "path": str(directory),
            "name": name_prefix,
            "is_dir": True,
            "files": [str(f.relative_to(directory)) for f in files],
            "has_reqs": has_reqs,
        })


def discover(root):
    """Find test directories and standalone scripts — fully generic."""
    results = []

    for item in sorted(root.iterdir()):
        if item.name.startswith(".") or item.name in _SKIP:
            continue
        if "copy" in item.name.lower():
            continue
        if HERMES_PRODUCT and not _matches_product(item.name):
            continue

        if item.is_dir():
            subdirs = [d for d in sorted(item.iterdir())
                       if d.is_dir() and not d.name.startswith(".")
                       and d.name not in _SUB_SKIP
                       and _find_test_files(d)]
            # Exclude Library directories referenced by .robot files
            robot_dirs = [d for d in subdirs if list(d.rglob("*.robot"))]
            lib_dir_names = set()
            for d in robot_dirs:
                for rf in d.rglob("*.robot"):
                    try:
                        content = rf.read_text(errors="replace")
                        for m in re.findall(r'Library\s+\.\./([\w]+)/', content):
                            lib_dir_names.add(m)
                    except Exception:
                        pass
            if lib_dir_names:
                subdirs = [d for d in subdirs if d.name not in lib_dir_names]

            if subdirs:
                for sub in subdirs:
                    parent_reqs = _has_requirements(item)
                    sub_reqs = _has_requirements(sub) or parent_reqs
                    root_py = [f for f in sub.glob("*.py")
                               if "copy" not in f.name.lower()]
                    root_robot = [f for f in sub.glob("*.robot")
                                  if "resource" not in f.name.lower()]
                    child_dirs = [d for d in sorted(sub.iterdir())
                                  if d.is_dir()
                                  and not d.name.startswith(".")
                                  and d.name not in _SUB_SKIP
                                  and d.name not in _SKIP
                                  and _find_test_files(d)]
                    if child_dirs:
                        root_sh = [f for f in sub.glob("*.sh")]
                        for rf in root_py + root_robot + root_sh:
                            results.append({
                                "path": str(rf),
                                "name": f"{item.name}/{sub.name}/{rf.stem}",
                                "is_dir": False,
                                "files": [rf.name],
                                "has_reqs": sub_reqs,
                            })
                        for cd in child_dirs:
                            _collect_dir(cd, f"{item.name}/{sub.name}/{cd.name}",
                                         sub_reqs, results)
                    else:
                        _collect_dir(sub, f"{item.name}/{sub.name}",
                                     parent_reqs, results)
            else:
                root_files = [f for f in item.glob("*.py")
                              if f.name not in _SKIP
                              and "copy" not in f.name.lower()]
                root_files += [f for f in item.glob("*.robot")
                               if f.name != "resource.robot"]
                for rf in root_files:
                    results.append({
                        "path": str(rf),
                        "name": f"{item.name}/{rf.stem}",
                        "is_dir": False,
                        "files": [rf.name],
                        "has_reqs": False,
                    })
                if not root_files:
                    files = _find_test_files(item)
                    if files:
                        results.append({
                            "path": str(item),
                            "name": item.name,
                            "is_dir": True,
                            "files": [str(f.relative_to(item)) for f in files],
                            "has_reqs": _has_requirements(item),
                        })
        elif item.suffix in (".py", ".robot"):
            results.append({
                "path": str(item),
                "name": item.stem,
                "is_dir": False,
                "files": [item.name],
                "has_reqs": False,
            })

    # Post-discovery: exclude dirs that are Library imports of .robot files
    # (resolve paths so Datastage/Utility isn't confused with WSL-WML/Utility)
    robot_results = [r for r in results
                     if any(f.endswith('.robot') for f in r['files'])]
    if robot_results:
        lib_abs_paths = set()
        for r in robot_results:
            base = Path(r['path'])
            for rel in r['files']:
                if not rel.endswith('.robot'):
                    continue
                fp = base / rel if r['is_dir'] else Path(r['path'])
                try:
                    content = fp.read_text(errors='replace')
                    robot_dir = fp.parent
                    for m in re.findall(r'Library\s+(\.\./[\w/]+)', content):
                        lib_path = (robot_dir / m.split("/")[0]
                                    / m.split("/")[1]).resolve()
                        lib_abs_paths.add(str(lib_path))
                except Exception:
                    pass
        if lib_abs_paths:
            before = len(results)
            results = [r for r in results
                       if any(f.endswith('.robot') for f in r['files'])
                       or not any(str(Path(r['path']).resolve()).startswith(lp)
                                  for lp in lib_abs_paths)]
            excluded = before - len(results)
            if excluded:
                print(f"  Excluded {excluded} Library dir(s)", flush=True)

    return results


def _discover_jenkins_jobs(root):
    """Scan TEST_ROOT for jenkins_jobs.json files and return suite entries."""
    jenkins_suites = []
    seen_jobs = set()
    for jf in sorted(root.rglob("jenkins_jobs.json")):
        if HERMES_PRODUCT:
            rel = jf.relative_to(root)
            top_dir = rel.parts[0] if rel.parts else ""
            if not _matches_product(top_dir):
                continue
        try:
            cfg = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            continue
        jobs = cfg.get("jobs", [])
        if not jobs:
            continue
        product_parts = []
        rel = jf.relative_to(root)
        for p in rel.parts[:-1]:
            product_parts.append(p)
        prefix = "_".join(product_parts) if product_parts else jf.parent.name
        prefix = re.sub(r'[^a-z0-9_]', '_', prefix.lower()).strip("_")

        for job in jobs:
            job_name = job.get("name", "unknown")
            if job_name in seen_jobs:
                continue
            seen_jobs.add(job_name)
            suite_name = re.sub(r'_+', '_', f"{prefix}_jenkins_{job_name}")

            # Jenkins = complex job → use Bob for analysis
            job_summary = json.dumps({
                "name": job_name,
                "job_path": job.get("job_path", ""),
                "description": job.get("description", ""),
                "parameters": {k: {"source": v.get("source"), "description": v.get("description", "")}
                               for k, v in job.get("parameters", {}).items()},
            }, indent=2)
            analysis_prompt = (
                f"You are a JSON-only responder. Analyze this Jenkins job config and return a JSON object.\n\n"
                f"Jenkins job config:\n{job_summary}\n\n"
                f"Return exactly one JSON object with these fields:\n"
                f'  "description": a clear one-line description of what this job tests,\n'
                f'  "tags": a list of relevant tags (e.g. ["core", "api", "model", "deployment"]),\n'
                f'  "critical": true if this is a critical/gate test, false otherwise,\n'
                f'  "estimated_duration_min": estimated duration in minutes\n\n'
                f"No text before or after the JSON."
            )

            ai_analysis = {}
            ai_source = "defaults"

            # Use Bob for Jenkins analysis
            print(f"    [Bob] analyzing Jenkins job: {job_name}...", flush=True)
            bob_resp = bob_generate(analysis_prompt, timeout=60, job_name=job_name)
            try:
                ai_analysis = json.loads(bob_resp) if not bob_resp.startswith("ERROR") else {}
            except (json.JSONDecodeError, TypeError):
                try:
                    m = re.search(r'\{.*\}', bob_resp, re.DOTALL)
                    ai_analysis = json.loads(m.group()) if m else {}
                except Exception:
                    ai_analysis = {}
            if ai_analysis:
                ai_source = "bob"
                print(f"    [Bob] ✓ {job_name}: {ai_analysis.get('description', '')[:60]}", flush=True)
            else:
                print(f"    [Bob] ✗ {job_name}: no valid response, using defaults", flush=True)

            jenkins_suites.append({
                "name": suite_name,
                "type": "jenkins",
                "jenkins_config": str(jf),
                "jenkins_job_name": job_name,
                # The job's own timeout_sec is a floor: a model guess must not
                # cut a long Jenkins run (e.g. 14h fastpath) short.
                "timeout_sec": max(int(job.get("timeout_sec", 7200)),
                                   int(ai_analysis.get("estimated_duration_min") or 0) * 60),
                "run_local": True,
                # From jenkins_jobs.json only, never the model's guess: a
                # critical suite that fails stops the whole run, and Bob
                # called a 6-hour fastpath job "critical", which ended a
                # regression the moment one fastpath flow failed.
                "critical": bool(job.get("critical", False)),
                "retry": bool(job.get("retry", True)),
                "tags": ai_analysis.get("tags", ["jenkins"]),
                "description": ai_analysis.get("description",
                               job.get("description", f"Jenkins job: {job_name}")),
                "ai_source": ai_source,
            })
        print(f"  Found jenkins_jobs.json: {jf} ({len(jobs)} job(s))", flush=True)
    return jenkins_suites


# ═════════════════════════════════════════════════════════════════════════════
# Static analysis — fast, no AI
# ═════════════════════════════════════════════════════════════════════════════

def static_scan(candidate):
    """Read code and extract facts: imports, entry points, deps."""
    base = Path(candidate["path"])
    info = {
        "uses_selenium": False,
        "uses_cpdctl": False,
        "uses_pytest": False,
        "imports": set(),
        "entry_point": None,
        "requirements": [],
    }

    # Determine entry point first (needed for browser detection)
    if not candidate["is_dir"]:
        raw_name = candidate["name"].split("/")[-1]
        matching = [f for f in candidate["files"] if f.startswith(raw_name)]
        if matching:
            info["entry_point"] = matching[0]
        elif any(f.endswith(".sh") for f in candidate["files"]):
            info["entry_point"] = next(f for f in candidate["files"] if f.endswith(".sh"))
        else:
            info["entry_point"] = raw_name + ".py"
    else:
        for name in ("login.py", "main.py", "run.py", "run_automation.py",
                     "test_web_automation.py"):
            if name in candidate["files"]:
                info["entry_point"] = name
                break
        if not info["entry_point"]:
            first_file = candidate["files"][0] if candidate["files"] else ""
            if first_file.endswith(".robot"):
                info["entry_point"] = first_file
            else:
                root_files = sorted(f for f in candidate["files"] if "/" not in f and not f.startswith("conftest"))
                test_py = [f for f in root_files if f.startswith("test_")]
                sh_files = [f for f in root_files if f.endswith(".sh")]
                root_py = [f for f in root_files if f.endswith(".py")]
                if test_py:
                    info["entry_point"] = test_py[0]
                elif sh_files:
                    info["entry_point"] = sh_files[0]
                elif root_py:
                    info["entry_point"] = root_py[0]
                else:
                    all_files = sorted(f for f in candidate["files"] if not f.endswith("conftest.py") and not f.endswith("__init__.py"))
                    if all_files:
                        info["entry_point"] = all_files[0]

    # Read requirements.txt — walk up from candidate dir to find it (case-sensitive)
    if candidate.get("has_reqs"):
        search_dir = base if candidate["is_dir"] else base.parent
        found = None
        d = search_dir
        for _ in range(4):
            req_file = _find_requirements(d)
            if req_file:
                found = req_file
                break
            if d == d.parent:
                break
            d = d.parent
        if found:
            info["requirements_path"] = str(found)
            for line in found.read_text(errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    pkg = re.split(r'[>=<!\[]', line)[0].strip()
                    if pkg:
                        info["requirements"].append(pkg)
                break

    # Scan test files for imports, cpdctl, pytest, robot libraries
    files_to_scan = candidate["files"][:10]
    for rel in files_to_scan:
        fp = base / rel if candidate["is_dir"] else Path(candidate["path"])
        try:
            code = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        if re.search(r'(?:subprocess|run|Popen|call).*cpdctl|["\']cpdctl["\']|command.*cpdctl|import.*cpdctl', code):
            info["uses_cpdctl"] = True
        if "pytest" in code or "conftest" in rel:
            info["uses_pytest"] = True

        # Detect zip file argument (sys.argv or argparse)
        if ("sys.argv" in code or "add_argument" in code) and "zip" in code.lower():
            info["uses_zip_arg"] = True

        # Extract specific zip filename from code (e.g. "T2_Sizing_Project_2.zip")
        for zm in re.findall(r'["\']([A-Za-z0-9_-]+\.zip)["\']', code):
            info.setdefault("zip_names", []).append(zm)

        # Extract PROJECT_NAME constant
        pm = re.search(r'PROJECT_NAME\s*=\s*["\']([^"\']+)["\']', code)
        if pm:
            info["cpd_project"] = pm.group(1)

        # Detect argparse CLI args (--cpd-url, --username, --password, --zip)
        for am in re.findall(r'add_argument\(\s*["\'](-{1,2}[\w-]+)["\']', code):
            info.setdefault("cli_args", []).append(am)
        # Detect argparse positional args (e.g. "zip_path")
        for am in re.findall(r'add_argument\(\s*["\'](\w+)["\']', code):
            if "zip" in am.lower():
                info["uses_zip_arg"] = True

        for m in re.findall(r'^(?:from|import)\s+([\w.]+)', code, re.MULTILINE):
            top = m.split(".")[0]
            if top not in ("os", "sys", "json", "time", "re", "pathlib",
                           "subprocess", "datetime", "logging", "uuid",
                           "base64", "argparse", "textwrap", "shutil",
                           "importlib", "collections", "typing",
                           "concurrent", "threading", "tempfile"):
                info["imports"].add(top)

        # Robot Framework: extract Library declarations
        for m in re.findall(r'^Library\s+([\w.]+)', code, re.MULTILINE):
            info["imports"].add(m.split(".")[0])

    # Parse HERMES HINTS from docstrings / comments
    hints = {}
    for rel in files_to_scan:
        fp = base / rel if candidate["is_dir"] else Path(candidate["path"])
        try:
            code = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        hm = re.search(r'HERMES HINTS[:\s]*\n((?:[ \t#]*\w+:.*\n)+)', code)
        if hm:
            for hl in hm.group(1).splitlines():
                hl = hl.strip().lstrip("#").strip()
                if ":" in hl:
                    hk, hv = hl.split(":", 1)
                    hk, hv = hk.strip(), hv.strip()
                    if hv.lower() in ("true", "false"):
                        hints[hk] = hv.lower() == "true"
                    elif hv.isdigit():
                        hints[hk] = int(hv)
                    else:
                        hints[hk] = hv
            break
    info["hermes_hints"] = hints

    # Detect browser usage by tracing entry point + its local imports
    def _reads_browser(filepath):
        try:
            code = filepath.read_text(encoding="utf-8", errors="replace")
            return ("selenium" in code or "webdriver" in code
                    or "playwright" in code)
        except Exception:
            return False

    if candidate["is_dir"] and info["entry_point"]:
        ep = base / info["entry_point"]
        if _reads_browser(ep):
            info["uses_selenium"] = True
        else:
            # Check files imported by the entry point; a shell entry point has
            # no imports, so follow the local .py files it invokes instead.
            try:
                ep_code = ep.read_text(encoding="utf-8", errors="replace")
                if ep.suffix == ".sh":
                    local_mods = re.findall(r'([\w./-]+\.py)\b', ep_code)
                else:
                    local_mods = [m.split(".")[0] + ".py" for m in
                                  re.findall(r'^\s*(?:from|import)\s+([\w.]+)', ep_code, re.MULTILINE)]
                for local_mod in local_mods:
                    local_path = base / local_mod
                    if local_path.exists() and _reads_browser(local_path):
                        info["uses_selenium"] = True
                        break
            except Exception:
                pass
    elif not candidate["is_dir"]:
        if _reads_browser(Path(candidate["path"])):
            info["uses_selenium"] = True

    # Robot: follow Resource imports to detect SeleniumLibrary
    if not info["uses_selenium"]:
        for rel in candidate.get("files", []):
            if not rel.endswith(".robot") or "resource" in rel.lower():
                continue
            fp = base / rel if candidate["is_dir"] else Path(candidate["path"])
            try:
                for res_name in re.findall(r'^Resource\s+(\S+\.robot)', fp.read_text(errors="replace"), re.MULTILINE):
                    res_path = fp.parent / res_name
                    if res_path.exists() and _reads_browser(res_path):
                        info["uses_selenium"] = True
                        break
            except Exception:
                pass
            if info["uses_selenium"]:
                break

    # Check for shell entry points
    if candidate["is_dir"]:
        for sh in ("run_full_test.sh", "run.sh"):
            if (base / sh).exists():
                info["shell_entry"] = sh
                break

    info["imports"] = sorted(info["imports"])
    return info


# ═════════════════════════════════════════════════════════════════════════════
# AI Agent — short focused prompt per suite
# ═════════════════════════════════════════════════════════════════════════════

def _read_robot_content(candidate):
    """Read Settings/Variables sections from .robot files (not full content).

    Sends a focused summary to Bob: Library/Resource imports, Variables,
    and Documentation from each file. This fits within the AI model's context
    window even for 10+ .robot files.
    """
    base = Path(candidate["path"])
    content_parts = []
    all_library_lines = []

    for rel in candidate["files"]:
        if not rel.endswith(".robot"):
            continue
        fp = base / rel if candidate["is_dir"] else Path(candidate["path"])
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # Extract Settings, Variables, and Test Case headers
        sections = []
        current_section = None
        current_lines = []
        test_case_names = []
        for line in text.splitlines():
            if line.startswith("*** "):
                if current_section and current_lines:
                    sections.append((current_section, current_lines))
                current_section = line.strip().strip("*").strip()
                current_lines = []
            elif current_section in ("Settings", "Variables"):
                current_lines.append(line)
            elif current_section == "Test Cases":
                # Capture test case names (non-indented lines)
                if line and not line[0].isspace():
                    test_case_names.append(line.strip())
        if current_section in ("Settings", "Variables") and current_lines:
            sections.append((current_section, current_lines))

        is_resource = "resource" in rel.lower()
        label = "RESOURCE FILE (shared, NOT the test — ignore its Libraries for classification)" if is_resource else ">>> TEST FILE (use THIS for run_local / uses_browser / pip_packages classification) <<<"
        file_summary = [f"── {rel} [{label}] ──"]
        for sec_name, sec_lines in sections:
            file_summary.append(f"*** {sec_name} ***")
            if is_resource and sec_name == "Settings":
                file_summary.append("    # NOTE: Libraries below belong to this RESOURCE file, NOT the test.")
                file_summary.append("    # Do NOT use these to decide run_local or uses_browser.")
            file_summary.extend(sec_lines)
        if test_case_names:
            file_summary.append("*** Test Cases (names only) ***")
            for tc in test_case_names:
                file_summary.append(f"  {tc}")

        content_parts.append("\n".join(file_summary))

        # Collect all Library lines + read actual imports from .py files
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("Library") and "../" in stripped:
                lib_path = stripped.split(None, 1)[1].strip() if len(stripped.split(None, 1)) > 1 else ""
                if lib_path:
                    entry = f"  - {lib_path} (from {rel})"
                    # Read actual imports from the .py file
                    py_file = base / lib_path if candidate["is_dir"] else base.parent / lib_path
                    if py_file.exists():
                        try:
                            py_code = py_file.read_text(errors="replace")
                            py_imports = []
                            for m in re.findall(r'^(?:from|import)\s+([\w.]+)', py_code, re.MULTILINE):
                                top = m.split(".")[0]
                                if top not in ("os", "sys", "json", "time", "re", "pathlib",
                                               "subprocess", "datetime", "logging", "uuid",
                                               "base64", "argparse", "textwrap", "shutil",
                                               "importlib", "collections", "typing", "copy",
                                               "concurrent", "threading", "tempfile", "csv",
                                               "io", "math", "glob", "traceback", "platform",
                                               "contextlib", "configparser", "hashlib",
                                               "config", "utils", "page_objects", "data",
                                               "collaborator_utils"):
                                    py_imports.append(top)
                            if py_imports:
                                entry += f"\n      ACTUAL imports: {', '.join(sorted(set(py_imports)))}"
                        except Exception:
                            pass
                    all_library_lines.append(entry)

    # Also read Requirements.txt if present (case-sensitive lookup)
    if candidate["is_dir"]:
        rpath = _find_requirements(base) or _find_requirements(base.parent)
        if rpath:
            try:
                content_parts.append(f"── {rpath.name} ──\n{rpath.read_text(errors='replace')}")
            except Exception:
                pass

    # Append explicit checklist of all local .py Library imports
    if all_library_lines:
        content_parts.append(
            "═══ LOCAL .py LIBRARY IMPORTS (determine packages for EACH) ═══\n"
            + "\n".join(all_library_lines)
        )

    return "\n\n".join(content_parts)


def ai_analyze(candidate, static_info):
    """Ask Bob a short focused question about this test suite."""
    name = candidate["name"]
    entry = static_info.get("entry_point", "unknown")
    imports = ", ".join(static_info["imports"][:15])
    sel = "YES" if static_info["uses_selenium"] else "NO"
    cpdctl = "YES" if static_info["uses_cpdctl"] else "NO"
    pytest_used = "YES" if static_info["uses_pytest"] else "NO"
    base = Path(candidate["path"])

    # Check if this candidate has .robot files
    has_robot = any(f.endswith(".robot") for f in candidate["files"])

    if has_robot:
        # Robot Framework suite — send full .robot content to Bob
        robot_content = _read_robot_content(candidate)
        file_list = ", ".join(candidate["files"][:15])

        # Pre-scan: does the TEST file (not resource) import SeleniumLibrary directly?
        test_has_selenium = False
        test_has_cpdctl = False
        for rel in candidate["files"]:
            if not rel.endswith(".robot") or "resource" in rel.lower():
                continue
            fp = base / rel if candidate["is_dir"] else Path(candidate["path"])
            try:
                txt = fp.read_text(errors="replace")
                if re.search(r'Library\s+SeleniumLibrary', txt):
                    test_has_selenium = True
                if re.search(r'(?i)cpdctl|Run Cpdctl|Configure Cpdctl', txt):
                    test_has_cpdctl = True
            except Exception:
                pass

        _sensitive = {"PASSWORD", "TOKEN", "SECRET", "KEY"}
        def _mask(k, v):
            if any(s in k.upper() for s in _sensitive):
                return "(set)"
            return v
        env_var_lines = "\n".join(f"  ${k} = {_mask(k, v)}" for k, v in _ENV_VARS.items()
                                    if not k.startswith("HERMES_") and not k.startswith("OLLAMA_"))
        prompt = f"""You are a JSON-only responder. Analyze this Robot Framework test suite for a CI pipeline.

Name: {name}
Files: {file_list}
Pre-scan facts (from reading the actual files):
  - TEST FILE (.robot) directly imports SeleniumLibrary: {"YES" if test_has_selenium else "NO"}
  - TEST FILE (.robot) references cpdctl: {"YES — found " + str(sum(1 for m in re.finditer(r'(?i)cpdctl', open(base / [r for r in candidate['files'] if r.endswith('.robot') and 'resource' not in r.lower()][0]).read()))) + " cpdctl references" if test_has_cpdctl else "NO"}

Available environment variables from .env (use these in -v flags):
{env_var_lines}
  $CPD_URL_HOST = hostname extracted from CPD_URL (no https:// prefix)

Full .robot file contents:
{robot_content}

Respond with exactly one JSON object:
{{"description": "short description", "run_command": "python3 -m robot -v url:$CPD_URL_HOST -v username:$CPD_USERNAME -v password:$CPD_PASSWORD file.robot", "timeout_sec": 3600, "critical": false, "run_first": false, "run_local": true, "env_vars": ["CPD_URL"], "pip_packages": ["robotframework", "robotframework-seleniumlibrary", "selenium"], "uses_browser": true, "capabilities": {{"needs_selenium": true, "needs_cpdctl": false, "needs_chrome": true, "needs_ssh_vm": false}}, "retry_strategy": {{"max_retries": 2, "retry_on": ["ElementNotVisible", "TimeoutException", "ElementClickIntercepted"], "pre_retry_actions": ["dismiss_walkme_overlay", "increase_wait_timeout"]}}, "self_heal_hints": [{{"error_pattern": "element click intercepted.*walkme-popup", "fix": "Execute JavaScript document.querySelectorAll('[id*=walkme]').forEach(e => e.remove())", "category": "ui"}}, {{"error_pattern": "not visible after", "fix": "Increase implicit wait to 120s and retry with --variable TIMEOUT:120", "category": "ui"}}], "pre_actions": ["dismiss_overlays"]}}

MANDATORY RULES (violating these causes build failures):
1. run_command FILENAME: Use the EXACT filename from the "Files:" list above. Do NOT modify, rename, or add/remove underscores. Copy the filename character-for-character.
2. run_local classification:
   - true ONLY if the test file's OWN Settings imports SeleniumLibrary/Selenium2Library/Browser directly.
   - false if SeleniumLibrary is only in a Resource file but the test file only uses cpdctl/Process/CLI keywords.
   - Look at the Test Cases names: "browser", "login page", "click", "UI" → true. "cpdctl", "import", "compile", "run flow", "create project" → false.
3. pip_packages for VM suites (run_local=false): These run on a remote s390x VM with NO C compiler and NO browser.
   - NEVER include packages that need native compilation (they fail to build: no gcc/cc/clang).
   - NEVER include browser/UI packages (no display, no Chrome).
   - ONLY include pure-Python packages: robotframework, requests, pyyaml, hvac, urllib3, etc.
   - Ask yourself: "Does this package install with just pip on a machine with NO C compiler?" If unsure, exclude it.
4. pip_packages for LOCAL suites (run_local=true): No restrictions, include everything needed.
5. VARIABLE MAPPING: For EVERY variable declared in the *** Variables *** section of the .robot file that has an EMPTY value or ${{EMPTY}} default, you MUST add a -v flag in run_command mapping it to the best matching environment variable from the list above. Match by meaning, not exact name:
   - Robot ${{git_token}} or ${{github_token}} → -v git_token:$GIT_TOKEN or -v github_token:$GIT_TOKEN (use the EXACT robot variable name)
   - Robot ${{git_url}} or ${{github_url}} → -v git_url:$GIT_URL or -v github_url:$GIT_URL
   - Robot ${{repo_username}} → -v repo_username:$GIT_REPO_USERNAME
   - Robot ${{DB2_HOST}} → -v DB2_HOST:$DB2_HOST
   - CRITICAL: copy the variable name EXACTLY as the .robot file declares it. The names above are
     illustrative only — a suite declaring ${{BRANCH_NAME}} needs "-v BRANCH_NAME:$GIT_BRANCH", NOT
     "-v branch:...". A -v flag whose name does not match a declared variable is silently ignored by
     Robot: no error, the variable keeps its empty default, and the suite fails much later for a
     reason that looks unrelated.

6. CREDENTIALS THE SUITE FETCHES ITSELF: if the suite setup calls a keyword that fetches a password
   or builds a URL (e.g. "Fetch CPD Password Over SSH", "Build CPD URL"), check whether it does so
   UNCONDITIONALLY. If there is no "IF '${{url}}' == ''" style guard, then -v url:/-v password: are
   OVERWRITTEN and useless — you must instead pass whatever that keyword consumes, typically
   -v CLUSTER_NAME:$CLUSTER_NAME -v CLUSTER_PASS:$CLUSTER_PASS -v CLUSTER_USER:$REMOTE_USER.

7. WRONG HARDCODED DEFAULTS: if a variable's default value contains a hostname or URL (for example a
   ${{test_url}} pointing at some other cluster), it is stale and MUST be overridden in run_command
   using $CPD_URL_HOST. Never let a generated command reach a cluster other than $CPD_URL_HOST.

8. VALUES THAT MUST BE UNIQUE PER RUN: CPD refuses to create a git token, or upload a file, whose
   name already exists — the dialog stays open, its Continue button is disabled, and every later
   click is intercepted, so the suite fails minutes later for a reason that looks unrelated. Reusing
   a name makes the SECOND run fail while the first passes, which is a silent trap.
   The environment provides $HERMES_RUN_ID: one value per run, identical across a suite's steps.
   Suffix every such name with it, e.g. -v TOKEN_NAME:tok$HERMES_RUN_ID, -v file_name:f$HERMES_RUN_ID.txt
   Apply this to ${{TOKEN_NAME}}, ${{token_name}}, ${{file_name}}, ${{python_file_name}} and any other
   variable naming an artefact the product creates.
   Do NOT do this for PROJECT NAMES: the suites search for an existing project and create one only
   when absent, so a project name must stay the SAME across runs or a new project is created every
   time and the reuse path is never exercised. Give project names a short fixed value.

9. Do NOT invent values for cluster facts (VERSION, RUNTIME, NAMESPACE). Pass them only from the
   environment variable list above. If no matching env var exists, omit the flag rather than guessing.

Field rules:
- run_command: Output EXACTLY ONE command. Never repeat or duplicate the command.
- run_command: python3 -m robot <variables> <EXACT_FILENAME_FROM_FILES_LIST>. Add -v flags for variables with empty defaults.
- IMPORTANT: Read the resource.robot or code to see if it prepends "https://" to the url variable. If it does (e.g. Evaluate "https://" + "${{url}}"), pass the URL WITHOUT the protocol: -v url:${{CPD_URL_HOST}} where CPD_URL_HOST strips the scheme. Otherwise use -v url:$CPD_URL.
- For standard WSL/WML robot suites that use resource.robot with "https://" + url pattern, use: -v url:$CPD_URL_HOST -v username:$CPD_USERNAME -v password:$CPD_PASSWORD
- uses_browser: true ONLY if the test file's OWN Settings imports SeleniumLibrary/Selenium2Library/Browser directly, OR test cases clearly use browser keywords. false if SeleniumLibrary is only inherited from Resource.
- pip_packages: Always include "robotframework". Map Robot libraries to pip names:
  SeleniumLibrary -> robotframework-seleniumlibrary
  Selenium2Library -> robotframework-selenium2library (NOT seleniumlibrary — different package)
  YamlLibrary -> robotframework-yamllibrary
  RequestsLibrary -> robotframework-requests
  SSHLibrary -> robotframework-sshlibrary
  Only include browser/UI packages if run_local=true AND the TEST FILE directly imports a browser library.
  CRITICAL: Look at the "LOCAL .py LIBRARY IMPORTS" section at the bottom. Each .py file shows "ACTUAL imports: ..." — these are the REAL Python imports read from the source code. Convert EVERY listed import to its pip package name:
    selenium -> selenium
    webdriver_manager -> webdriver-manager
    plotly -> plotly
    pandas -> pandas
    requests -> requests
    urllib3 -> urllib3
    hvac -> hvac
    pyyaml -> pyyaml
    pytest -> pytest
    paramiko -> paramiko
    fabric -> fabric
  If a .py file has no "ACTUAL imports" line, it uses only stdlib — no extra packages needed for that file.
  Include packages from Requirements.txt if shown.
  Built-in Robot libraries (BuiltIn, Collections, DateTime, OperatingSystem, Process, String, XML) need NO pip install.
- capabilities:
  needs_selenium: true ONLY if the test file's OWN Settings imports SeleniumLibrary directly. false if only inherited from Resource.
  needs_cpdctl: true if "cpdctl" appears anywhere or test uses Run Cpdctl, Configure Cpdctl keywords
  needs_chrome: true if needs_selenium is true
  needs_ssh_vm: true if run_local is false
- timeout_sec: default 3600. Use 7200 for performance tests.
- retry_strategy: Analyze the .robot file for UI interactions that commonly fail. Generate:
  max_retries: 2 for UI/Selenium tests, 1 for API/CLI tests
  retry_on: list error class names from the test keywords (e.g. ElementNotVisible, TimeoutException, ElementClickIntercepted, ConnectionError, WebDriverException)
  pre_retry_actions: list remediation steps the framework should try before retrying. Common ones:
    "dismiss_walkme_overlay" — if the test clicks UI elements (WalkMe popups block clicks on CPD)
    "increase_wait_timeout" — if test uses Wait Until Element Is Visible with short timeouts
    "refresh_page" — if the test navigates to a page that may not fully load
    "clear_browser_cache" — if the test has session/cookie issues
    "re_authenticate" — if the test does CPD login
- self_heal_hints: Analyze the .robot file's Test Cases and Keywords. For EACH interaction that could fail, generate a hint object:
  error_pattern: regex matching the likely Selenium/Robot error message
  fix: the exact Robot Framework or JavaScript command to fix it
  category: "ui" for browser issues, "auth" for login issues, "infra" for connectivity, "config" for env/setup issues
  Examples of good hints:
    - Test clicks a button → hint for walkme overlay interception: {{"error_pattern": "element click intercepted.*walkme", "fix": "Execute JavaScript document.querySelectorAll('[id*=walkme]').forEach(e => e.remove())", "category": "ui"}}
    - Test waits for element → hint for slow page load: {{"error_pattern": "not visible after", "fix": "Increase implicit wait and add Sleep 5s before retry", "category": "ui"}}
    - Test does login → hint for auth failure: {{"error_pattern": "Login failed|Invalid credentials|401", "fix": "Re-authenticate via API and refresh session token", "category": "auth"}}
    - Test accesses Git → hint for token expiry: {{"error_pattern": "Bad credentials|403|token", "fix": "Refresh GIT_TOKEN from environment", "category": "auth"}}
  Generate at least 2 hints per suite. Be specific to what the .robot file actually does.
- pre_actions: list of actions to run BEFORE the test starts. Common ones:
  "dismiss_overlays" — for ANY Selenium test (always include for UI tests)
  "verify_cpd_login" — if the test expects to be logged into CPD
  "check_git_connectivity" — if the test uses GitHub integration

No text before or after the JSON."""
    else:
        # Python suite — existing prompt
        snippet = ""
        if entry:
            ep = base / entry if candidate["is_dir"] else base
            try:
                lines = ep.read_text(errors="replace").splitlines()[:20]
                snippet = "\n".join(lines)
            except Exception:
                pass

        file_list = ", ".join(candidate["files"][:15])

        prompt = f"""You are a JSON-only responder. Analyze this test suite for a CI pipeline.

Name: {name}
Entry: {entry}
Files: {file_list}
Selenium: {sel}
cpdctl: {cpdctl}
pytest: {pytest_used}
Imports: {imports}

Code snippet:
{snippet}

Respond with exactly one JSON object:
{{"description": "short description", "is_test": true, "run_command": "python3 main.py", "timeout_sec": 3600, "critical": false, "run_first": false, "run_local": true, "env_vars": ["CPD_URL"], "pip_packages": ["selenium", "requests"], "uses_browser": true, "capabilities": {{"needs_selenium": true, "needs_cpdctl": false, "needs_chrome": true, "needs_ssh_vm": false}}, "retry_strategy": {{"max_retries": 1, "retry_on": ["ConnectionError", "TimeoutError"], "pre_retry_actions": []}}, "self_heal_hints": [{{"error_pattern": "Connection refused|ConnectionError", "fix": "Check service connectivity and retry", "category": "infra"}}], "pre_actions": []}}

Field rules:
- is_test: true if this is an actual test suite, test script, or operational validation script (e.g. SRE scripts that test node-pinning, shutdown/restart, monitoring, uninstall). Shell (.sh) scripts should almost always be is_test: true — they are runnable validation scripts. Only set false for pure helper/library files that cannot run standalone (e.g. shared functions sourced by other scripts).
- run_command: the exact shell command to run this suite from its directory. For .py files: "python3 <actual_filename>.py" — use the REAL filename of the main script, NEVER guess "main.py" unless a file literally named main.py exists. For .sh files: "bash <actual_filename>.sh". For .robot files: "python3 -m robot -v url:$CPD_URL -v username:$CPD_USERNAME -v password:$CPD_PASSWORD testfile.robot". Pick the right runner based on the file type. If a HERMES HINTS block contains run_command, use that value exactly.
- HERMES HINTS: In Python files these appear in docstrings. In shell (.sh) files they appear as comments like:
  # HERMES HINTS:
  #   run_first: false
  #   timeout_sec: 3600
  #   run_local: false
  If present, respect the values specified.
- run_first: true ONLY for gate/prerequisite tests that must pass before anything else runs (e.g. job0, connectivity checks), OR long-running tests (8h+) that must start early. If the code contains "HERMES HINTS" with "run_first: true", respect it.
- critical: true ONLY for run_first gate tests. false for everything else.
- run_local: true if the suite needs a browser (Selenium/Chrome/WebDriver/Playwright). false if it runs via cpdctl on a remote VM. Suites using cpdctl are NEVER run_local.
- uses_browser: true if the suite uses Selenium, Playwright, Chrome, WebDriver, or any browser UI automation — even indirectly. false otherwise. cpdctl-based suites are NEVER browser suites.
- pip_packages: list pip package names needed. Use correct pip names (e.g. "pyyaml" for yaml, "python-dotenv" for dotenv, "Pillow" for PIL, "robotframework" for robot). Omit standard-library modules.
- capabilities: detect what the test environment needs:
  needs_selenium: true if it uses Selenium/WebDriver/Playwright/browser automation
  needs_cpdctl: true if it uses cpdctl CLI commands
  needs_chrome: true if needs_selenium is true
  needs_ssh_vm: true if run_local is false (suite runs on remote VM)
- timeout_sec: estimated run time in seconds. Default 3600. Use 7200+ for long performance tests. If the code has "HERMES HINTS" with a specific timeout_sec, use that value.
- retry_strategy: Generate retry configuration based on the test type:
  max_retries: 2 for UI/browser tests, 1 for API/CLI/cpdctl tests
  retry_on: list the error types this test could encounter (e.g. ConnectionError, TimeoutError, WebDriverException, ElementClickIntercepted)
  pre_retry_actions: list remediation steps before retrying (e.g. "dismiss_walkme_overlay", "increase_wait_timeout", "re_authenticate", "refresh_page")
- self_heal_hints: Generate at least 1 hint per suite with:
  error_pattern: regex matching the likely error
  fix: actionable fix command
  category: "ui", "auth", "infra", or "config"
- pre_actions: list of actions to run BEFORE the test starts (e.g. "dismiss_overlays" for Selenium tests, "check_git_connectivity" for git tests)

No text before or after the JSON."""

    print(f"  [Agent:{name}] asking Bob...", flush=True)
    resp = bob_generate(prompt, timeout=300, job_name=name)
    result = extract_json(resp)

    if result:
        # Ensure robotframework is in pip_packages for .robot suites
        if has_robot:
            pkgs = result.get("pip_packages", [])
            if "robotframework" not in pkgs:
                pkgs.insert(0, "robotframework")
                result["pip_packages"] = pkgs
        print(f"  [Agent:{name}] OK — {result.get('description', '')[:50]}", flush=True)
        return result
    else:
        print(f"  [Agent:{name}] fallback (no JSON). Raw: {resp[:120]}", flush=True)
        if has_robot:
            return {"pip_packages": ["robotframework"]}
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# Build suite entry
# ═════════════════════════════════════════════════════════════════════════════

def build_suite(candidate, static_info, ai_result):
    """Combine static + AI analysis into a suite entry."""
    name = re.sub(r'_+', '_', re.sub(r'[^a-z0-9_]', '_', candidate["name"].lower())).strip("_")
    entry = static_info.get("entry_point") or (candidate["files"][0] if candidate["files"] else candidate["name"].split("/")[-1])
    is_dir = candidate["is_dir"]
    cli_args = static_info.get("cli_args", [])

    needs_browser = static_info["uses_selenium"] or ai_result.get("uses_browser", False) or name in BROWSER_OVERRIDES
    uses_zip = static_info.get("uses_zip_arg", False)
    zip_names = static_info.get("zip_names", [])

    # Build command
    parts = []
    if is_dir:
        parts.append(f'cd "$HERMES_EXTRA_DIR/{candidate.get("suite_dir", candidate["name"])}"')
        if candidate.get("has_reqs"):
            req_path = static_info.get("requirements_path")
            if req_path:
                try:
                    rel_req = str(Path(req_path).relative_to(TEST_ROOT))
                except ValueError:
                    rel_req = None
            else:
                rel_req = None
            if rel_req:
                req_ref = f'"$HERMES_EXTRA_DIR/{rel_req}"'
            else:
                req_ref = "requirements.txt"
            pip_base = "python3 -m pip install -q"
            parts.append(f'{pip_base} -r {req_ref}')
    else:
        parent_dir = "/".join(candidate["name"].split("/")[:-1])
        if parent_dir:
            parts.append(f'cd "$HERMES_EXTRA_DIR/{parent_dir}"')
        else:
            parts.append('cd "$HERMES_EXTRA_DIR"')

    # Install pip packages identified by the AI agent (skip if Requirements.txt already covers them)
    has_req_install = any("pip install" in p and "-r " in p for p in parts)
    pip_pkgs = ai_result.get("pip_packages", [])
    if pip_pkgs and not has_req_install:
        pkg_list = " ".join(pip_pkgs)
        parts.append(f"python3 -m pip install --quiet {pkg_list}")

    # Use AI-provided run_command if available
    ai_cmd = ai_result.get("run_command", "")
    if ai_cmd:
        # Strip BOM, non-ASCII garbage, and stray quotes from hallucinated output
        ai_cmd = ai_cmd.replace("﻿", "").replace("§", "").replace("​", "")
        ai_cmd = re.sub(r'python3\s+-m\s+(["\']?)(\S+\.py)\1', r'python3 \2', ai_cmd)
        # Strip pip install from ai_cmd when Requirements.txt already handles deps
        if has_req_install:
            ai_cmd = re.sub(r'python3\s+-m\s+pip\s+install\s+[^&]+&&\s*', '', ai_cmd)
            ai_cmd = re.sub(r'pip3?\s+install\s+[^&]+&&\s*', '', ai_cmd)
        # Fix duplicate commands — model sometimes outputs the command twice
        for marker in ("python3 -m robot", "python3 -m pytest", "robot "):
            idx = ai_cmd.find(marker)
            if idx >= 0:
                idx2 = ai_cmd.find(marker, idx + len(marker))
                if idx2 > idx:
                    ai_cmd = ai_cmd[:idx] + ai_cmd[idx2:]
                break
        # Validate robot/sh/py filenames — model can hallucinate names
        if candidate["is_dir"]:
            base = Path(candidate["path"])
            for ext in ("*.robot", "*.sh", "*.py"):
                actual_files = {f.name for f in base.glob(ext)}
                suffix = ext.replace("*", "")
                for wrong in re.findall(r'(\S+' + re.escape(suffix) + r')', ai_cmd):
                    if wrong not in actual_files:
                        def _norm(s): return s.lower().replace("_", "").replace("-", "")
                        close = [f for f in actual_files if _norm(f) == _norm(wrong)]
                        if close:
                            ai_cmd = ai_cmd.replace(wrong, close[0])
                        elif entry and entry.endswith(suffix):
                            ai_cmd = ai_cmd.replace(wrong, entry)
        else:
            real_name = candidate["files"][0] if candidate.get("files") else ""
            if real_name:
                for ext in ("*.robot", "*.sh", "*.py"):
                    suffix = ext.replace("*", "")
                    for wrong in re.findall(r'(\S+' + re.escape(suffix) + r')', ai_cmd):
                        if wrong != real_name:
                            ai_cmd = ai_cmd.replace(wrong, real_name)
        # Append CLI args detected by static analysis if missing from ai_cmd
        if "--cpd-url" in cli_args and "--cpd-url" not in ai_cmd:
            ai_cmd += ' --cpd-url "$CPD_URL"'
        if "--username" in cli_args and "--username" not in ai_cmd:
            ai_cmd += ' --username "$CPD_USERNAME"'
        if "--password" in cli_args and "--password" not in ai_cmd:
            ai_cmd += ' --password "$CPD_PASSWORD"'
        if "--zip" in cli_args and "--zip" not in ai_cmd and zip_names:
            ai_cmd += f' --zip "${{_zip:-{zip_names[0]}}}"'
        # Fix Robot -v flags: read the .robot file's actual variable names
        if ".robot" in ai_cmd:
            robot_match = re.search(r'(\S+\.robot)', ai_cmd)
            if robot_match:
                robot_name = robot_match.group(1)
                base_dir = Path(candidate["path"]) if candidate["is_dir"] else Path(candidate["path"]).parent
                robot_path = base_dir / robot_name
                if robot_path.exists():
                    rtxt = robot_path.read_text(errors="replace")
                    # Check what variables the .robot file declares
                    has_cpd_host = bool(re.search(r'\$\{CPD_HOST\}', rtxt))
                    # Check if resource.robot prepends https://
                    res_path = base_dir / "resource.robot"
                    url_needs_host = False
                    if res_path.exists():
                        restxt = res_path.read_text(errors="replace")
                        if '"https://"' in restxt and "url" in restxt:
                            url_needs_host = True
                    if has_cpd_host:
                        ai_cmd = re.sub(r'-v\s+(?:url|CPD_URL|cpd_url):\$CPD_URL(?:_HOST)?\b', '-v CPD_HOST:$CPD_URL_HOST', ai_cmd)
                        if '-v CPD_HOST:' not in ai_cmd:
                            ai_cmd = re.sub(r'(\S+\.robot)', r'-v CPD_HOST:$CPD_URL_HOST \1', ai_cmd, count=1)
                        ai_cmd = re.sub(r'-v\s+username:\$CPD_USERNAME\b', '-v CPD_USERNAME:$CPD_USERNAME', ai_cmd)
                        ai_cmd = re.sub(r'-v\s+password:\$CPD_PASSWORD\b', '-v CPD_PASSWORD:$CPD_PASSWORD', ai_cmd)
                    elif url_needs_host:
                        ai_cmd = re.sub(r'-v\s+url:\$CPD_URL(?:_HOST)?\b', '-v url:$CPD_URL_HOST', ai_cmd)
        # Sanitize -v flags: dedup, reject garbage/hallucinated values, validate env vars
        if "python3 -m robot" in ai_cmd:
            robot_file = re.search(r'(\S+\.robot)', ai_cmd)
            if not robot_file:
                rn = candidate["files"][0] if candidate.get("files") else ""
                if rn:
                    ai_cmd = ai_cmd.rstrip().rstrip("'\"") + " " + rn
                    robot_file = re.search(r'(\S+\.robot)', ai_cmd)
            runner_match = re.search(r'(.*?python3\s+-m\s+robot)\s', ai_cmd)
            if robot_file and runner_match:
                rf = robot_file.group(1)
                runner = runner_match.group(1)
                flags = re.findall(r'-v\s+(\S+)', ai_cmd)
                seen, clean = set(), []
                for flag in flags:
                    if ":" not in flag:
                        continue
                    key, val = flag.split(":", 1)
                    if not val.startswith("$") or not re.match(r'^\$[A-Z][A-Z0-9_]*$', val):
                        continue
                    env_name = val[1:]
                    if env_name not in _ENV_VARS and env_name not in _DERIVED_ENV:
                        continue
                    knorm = key.lower().replace("_", "")
                    if knorm in seen:
                        continue
                    seen.add(knorm)
                    clean.append(f"-v {key}:{val}")
                ai_cmd = runner + " " + " ".join(clean) + " " + rf
                # Inject missing vars: read .robot file for empty-default variables
                base_dir = Path(candidate["path"]) if candidate["is_dir"] else Path(candidate["path"]).parent
                robot_path = base_dir / rf
                if robot_path.exists():
                    rtxt = robot_path.read_text(errors="replace")
                    # Includes variables inherited from Resource imports, so
                    # credentials declared in resource.robot are not missed.
                    for rv in sorted(_empty_default_variables(str(base_dir), rf)):
                        if re.search(r'-v\s+' + re.escape(rv) + r':', ai_cmd, re.IGNORECASE):
                            continue
                        matched = _match_env_var(rv)
                        if matched:
                            ai_cmd = re.sub(r'(\S+\.robot)', f'-v {rv}:${matched} \\1', ai_cmd, count=1)
                    var_section = ""
                    if "*** Variables ***" in rtxt:
                        var_section = rtxt.split("*** Variables ***", 1)[1].split("***")[0]
                    # Matches a Variables-table entry with no real value:
                    #   ${FOO}                        (bare)
                    #   ${FOO}    ${EMPTY}    # note  (explicit empty)
                    #   ${FOO}    # note              (comment only)
                    # The previous pattern required \s+ before the line end, so a
                    # bare "${FOO}\n" never matched and almost nothing was injected.
                    empty_var_re = r'^\$\{(\w+)\}[ \t]*(?:\$\{EMPTY\})?[ \t]*(?:#.*)?$'
                    for m in re.finditer(empty_var_re, var_section, re.MULTILINE):
                        rv = m.group(1)
                        if re.search(r'-v\s+' + re.escape(rv) + r':', ai_cmd, re.IGNORECASE):
                            continue
                        matched = _match_env_var(rv)
                        if matched:
                            ai_cmd = re.sub(r'(\S+\.robot)', f'-v {rv}:${matched} \\1', ai_cmd, count=1)
        parts.append(ai_cmd)
    else:
        # Fallback: auto-detect command from static analysis
        # Zip file discovery at runtime
        if uses_zip and not is_dir:
            if zip_names:
                zip_pattern = zip_names[0]
            else:
                keyword = name.replace("_automation", "").replace("_test", "").replace("_", "")
                zip_pattern = f"*{keyword}*.zip"
            parts.append(f'_zip=$(find "$HERMES_EXTRA_DIR" -maxdepth 2 -name "{zip_pattern}" | head -1)')

        run_parts = []
        if needs_browser and not static_info["uses_pytest"]:
            run_parts.append("HEADLESS=1")

        # A shell entry point wraps its own runner (often pytest) — never hand
        # the .sh itself to pytest ("found no collectors for run_full_test.sh").
        if entry.endswith(".sh"):
            run_parts.append(f"bash {entry}")
        elif static_info["uses_pytest"]:
            run_parts.append(f"pytest {entry} -v -s")
        else:
            run_parts.append(f"python3 {entry}")

        if "--zip" in cli_args and zip_names:
            run_parts.append(f'--zip "${{_zip:-{zip_names[0]}}}"')
        elif uses_zip:
            run_parts.append('${_zip:+"$_zip"}')

        if "--cpd-url" in cli_args:
            run_parts.append('--cpd-url "$CPD_URL"')
        if "--username" in cli_args:
            run_parts.append('--username "$CPD_USERNAME"')
        if "--password" in cli_args:
            run_parts.append('--password "$CPD_PASSWORD"')

        parts.append(" ".join(run_parts))

    cmd = "bash -c '" + " && ".join(parts) + "'"

    # Fix .robot commands: ensure python3 -m robot is used and -v flags are present
    if any(f.endswith(".robot") for f in candidate.get("files", [])):
        cmd = re.sub(r'\bpython3\s+(\S+\.robot)\b', r'python3 -m robot \1', cmd)
        robot_match = re.search(r'(\S+\.robot)', cmd)
        if robot_match and "-v " not in cmd:
            robot_name = robot_match.group(1)
            cmd = cmd.replace(
                f"python3 -m robot {robot_name}",
                f"python3 -m robot -v url:$CPD_URL_HOST -v username:$CPD_USERNAME -v password:$CPD_PASSWORD {robot_name}"
            )
        if robot_match:
            rf = robot_match.group(1)
            base_dir = Path(candidate["path"]) if candidate["is_dir"] else Path(candidate["path"]).parent
            robot_path = base_dir / rf
            if robot_path.exists():
                rtxt = robot_path.read_text(errors="replace")
                var_section = ""
                if "*** Variables ***" in rtxt:
                    var_section = rtxt.split("*** Variables ***", 1)[1].split("***")[0]
                for m in re.finditer(r'\$\{(\w+)\}\s+(?:\$\{EMPTY\}|#|$)', var_section, re.MULTILINE):
                    rv = m.group(1)
                    if re.search(r'-v\s+' + re.escape(rv) + r':', cmd, re.IGNORECASE):
                        continue
                    matched = _match_env_var(rv)
                    if matched:
                        cmd = re.sub(r'(\S+\.robot)', f'-v {rv}:${matched} \\1', cmd, count=1)

    # Tags
    tags = []
    if needs_browser:
        tags.extend(["selenium", "mac", "ui"])
    if static_info["uses_cpdctl"]:
        tags.extend(["cpdctl", "vm"])
    if "perf" in name or "scenario" in name:
        tags.append("perf")
    if not tags:
        tags.append("core")

    suite = {
        "name": name,
        "cmd": cmd,
        "timeout_sec": ai_result.get("timeout_sec", 3600),
        "critical": ai_result.get("critical", False),
        "tags": tags,
    }

    hints = static_info.get("hermes_hints", {})

    if hints.get("run_first", ai_result.get("run_first", False)):
        suite["run_first"] = True
        suite["critical"] = hints.get("critical", True)
    if "timeout_sec" in hints:
        suite["timeout_sec"] = hints["timeout_sec"]
    is_robot = any(f.endswith(".robot") for f in candidate.get("files", []))
    if static_info["uses_cpdctl"]:
        suite["run_local"] = False
    elif "run_local" in hints:
        suite["run_local"] = hints["run_local"]
    elif needs_browser or ai_result.get("run_local", False):
        suite["run_local"] = True
    elif is_robot and not static_info["uses_cpdctl"]:
        suite["run_local"] = True

    # cpd_project from code
    cpd_proj = static_info.get("cpd_project")
    if cpd_proj:
        suite["cpd_project"] = cpd_proj

    desc = ai_result.get("description")
    if desc:
        suite["description"] = desc

    pip_packages = ai_result.get("pip_packages", [])
    if pip_packages:
        suite["pip_packages"] = pip_packages

    caps = ai_result.get("capabilities", {})
    if caps:
        suite["capabilities"] = caps

    return suite


# ═════════════════════════════════════════════════════════════════════════════
# Requirements aggregator
# ═════════════════════════════════════════════════════════════════════════════

def collect_requirements(suites, all_static, all_ai_results):
    """Split pip requirements by target: VM suites → s390x, LOCAL suites → x86.

    Uses pip_packages stored on each suite object (set by build_suite)
    so ordering is correct even after the assembler reorders suites.
    """
    vm_pkgs = set()
    local_pkgs = set()
    for suite in suites:
        is_local = suite.get("run_local", False)
        target = local_pkgs if is_local else vm_pkgs
        for pkg in suite.get("pip_packages", []):
            target.add(pkg)
    # selenium/webdriver belong on x86 (browser), never on s390x
    _browser_only = {"selenium", "webdriver-manager", "webdriver_manager"}
    moved = vm_pkgs & _browser_only
    vm_pkgs -= _browser_only
    local_pkgs |= moved
    # DS_resource.robot imports SeleniumLibrary — Robot Framework must parse it
    # even on VM, so robotframework-seleniumlibrary is a parse-time dependency
    if vm_pkgs and "robotframework" in vm_pkgs:
        vm_pkgs.add("robotframework-seleniumlibrary")
    return sorted(vm_pkgs), sorted(local_pkgs)


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70, flush=True)
    print("  Hermes Suite Generator — Bob AI agents", flush=True)
    print("=" * 70, flush=True)
    print(f"  AI backend: Bob (IBM)", flush=True)
    print(f"  Test root:  {TEST_ROOT}", flush=True)
    print(flush=True)

    # Check Bob API key
    if not BOB_API_KEY:
        print("  ERROR: BOB_API_KEY not set in .env", flush=True)
        sys.exit(1)
    print("  Bob: API key configured", flush=True)
    if not _ensure_bob_ready():
        print("  WARNING: Bob Shell CLI setup issue — will retry on first call", flush=True)

    # Phase 1 — Discovery
    print(f"\n{'='*70}", flush=True)
    print("  PHASE 1 — Discovery Agent (filesystem scan)", flush=True)
    print("=" * 70, flush=True)

    candidates = discover(TEST_ROOT)
    print(f"  Found {len(candidates)} test suite(s):", flush=True)
    for c in candidates:
        tag = "DIR " if c["is_dir"] else "FILE"
        robot_cnt = sum(1 for f in c["files"] if f.endswith(".robot"))
        py_cnt = sum(1 for f in c["files"] if f.endswith(".py"))
        sh_cnt = sum(1 for f in c["files"] if f.endswith(".sh"))
        counts = f"{robot_cnt} .robot, {py_cnt} .py, {sh_cnt} .sh"
        print(f"    [{tag}] {c['name']}  ({counts})", flush=True)

    jenkins_suites = _discover_jenkins_jobs(TEST_ROOT)
    if jenkins_suites:
        print(f"  Found {len(jenkins_suites)} Jenkins job suite(s):", flush=True)
        for js in jenkins_suites:
            src = js.get("ai_source", "defaults")
            print(f"    [JENKINS] {js['name']}  ({js['jenkins_job_name']})  [AI: {src}]", flush=True)

    suites = []
    all_static = []
    all_ai_results = []
    suite_dirs = {}
    # Gate test: auto-detect from HERMES_JOB0_DIR (any product can set this)
    job0_dir = os.getenv("HERMES_JOB0_DIR")
    job0_script = None
    if job0_dir and Path(job0_dir).exists():
        sh_files = list(Path(job0_dir).glob("*.sh"))
        run_sh = [f for f in sh_files if f.name.startswith("run")]
        job0_sh = run_sh[0] if run_sh else (sh_files[0] if sh_files else None)
        if job0_sh:
            job0_script = job0_sh.name
    if job0_dir and job0_script:
        job0_suite = {
            "name": "job0_test",
            "cmd": f'bash "$HERMES_JOB0_DIR/{job0_script}"',
            "timeout_sec": 7200,
            "critical": True,
            "run_first": True,
            "tags": ["core", "job0", "gate"],
            "capabilities": {
                "needs_cpdctl": True,
                "needs_selenium": False,
                "needs_chrome": False,
                "needs_ssh_vm": True,
            },
        }
        suites.append(job0_suite)
        all_static.append({})
        all_ai_results.append({})
        print(f"  + job0_test (gate test from {job0_dir})", flush=True)

    # Phase 2 — Static + AI analysis
    print(f"\n{'='*70}", flush=True)
    print("  PHASE 2 — Analysis Agents (static scan + Bob)", flush=True)
    print("=" * 70, flush=True)

    for c in candidates:
        info = static_scan(c)
        all_static.append(info)

        ai_result = ai_analyze(c, info)
        all_ai_results.append(ai_result)

        hints = info.get("hermes_hints", {})
        is_test = hints.get("is_test", ai_result.get("is_test", True))
        if not is_test:
            print(f"  [Skip] {c['name']} — not a test suite", flush=True)
            continue
        suite = build_suite(c, info, ai_result)
        # Remember where this suite's .robot files live so validation can read
        # them. Stripped again before the config is written.
        suite_dirs[suite["name"]] = c["path"] if c.get("is_dir") else str(Path(c["path"]).parent)
        suites.append(suite)

    if jenkins_suites:
        suites.extend(jenkins_suites)
        print(f"  Added {len(jenkins_suites)} Jenkins job suite(s)", flush=True)

    # Phase 3 — Assemble
    print(f"\n{'='*70}", flush=True)
    print("  PHASE 3 — Assembler Agent", flush=True)
    print("=" * 70, flush=True)

    # Ask Bob to review the final ordering
    suite_names = [s["name"] for s in suites]
    order_prompt = f"""You are a JSON-only responder. Reorder these CI test suite names.
Put critical/gate tests first, then core tests, then performance tests, then UI tests.

Suite names: {json.dumps(suite_names)}

Respond with exactly one JSON object like: {{"ordered": ["name1", "name2"]}}
No text before or after the JSON."""

    print("  [Assembler] asking Bob for ordering...", flush=True)
    resp = bob_generate(order_prompt, timeout=60, job_name="assembler_ordering")

    # Try to parse ordered list
    try:
        parsed = extract_json(resp)
        ordered = None
        if isinstance(parsed, dict) and "ordered" in parsed:
            ordered = parsed["ordered"]
        elif isinstance(parsed, list):
            ordered = parsed
        else:
            m = re.search(r'\[.*\]', resp, re.DOTALL)
            if m:
                ordered = json.loads(m.group())
        if ordered:
            name_to_suite = {s["name"]: s for s in suites}
            reordered = []
            for n in ordered:
                if n in name_to_suite:
                    reordered.append(name_to_suite.pop(n))
            for s in name_to_suite.values():
                reordered.append(s)
            suites = reordered
            print(f"  [Assembler] ordered {len(suites)} suites", flush=True)
    except Exception:
        print("  [Assembler] using discovery order", flush=True)

    # Collect requirements — split by target
    vm_deps, local_deps = collect_requirements(suites, all_static, all_ai_results)

    # Write outputs
    print(f"\n{'='*70}", flush=True)
    print("  OUTPUT", flush=True)
    print("=" * 70, flush=True)

    # Determine test type from discovered suites (robot vs python)
    has_robot = any(any(f.endswith(".robot") for f in c["files"]) for c in candidates)
    has_python = any(any(f.endswith(".py") for f in c["files"]) for c in candidates)
    test_type = "robot" if has_robot else ("python" if has_python else "none")
    print(f"\n  Test type detected: {test_type.upper()}", flush=True)

    suites_file = CONFIGS_DIR / "suites_generated.json"

    # Validate, and where a suite's freshly generated command is broken, fall back
    # to that suite's entry from the previous config if that one was sound. The
    # config is regenerated on every supervisor run, so without this a single bad
    # model response replaces a working command with an unrunnable one.
    for s in suites:
        if suite_dirs.get(s["name"]):
            s["path"] = suite_dirs[s["name"]]
    bad_cmds = validate_suite_cmds(suites)
    if bad_cmds:
        previous = {}
        if suites_file.exists():
            try:
                prev = json.loads(suites_file.read_text(encoding="utf-8"))
                previous = {p["name"]: p for p in prev.get("suites", []) if "cmd" in p}
            except Exception:
                previous = {}
        by_name = {s["name"]: s for s in suites}
        recovered, still_bad = [], []
        for name, reason in bad_cmds:
            old = previous.get(name)
            if old and suite_dirs.get(name):
                old["path"] = suite_dirs[name]
            if old and not validate_suite_cmds([old]):
                by_name[name]["cmd"] = old["cmd"]
                recovered.append(name)
            else:
                still_bad.append((name, reason))

        print("\n  ⚠️  VALIDATION PROBLEMS IN GENERATED COMMANDS:", flush=True)
        for name, reason in bad_cmds:
            print(f"      {name}: {reason}", flush=True)
        if recovered:
            print(f"\n  ↩  Kept the previous working command for: {', '.join(recovered)}", flush=True)
        if still_bad:
            print(f"\n  ✗  Still broken with no good previous version: "
                  f"{', '.join(n for n, _ in still_bad)}", flush=True)
            print("     Re-run generation before trusting this config.", flush=True)

    for s_ in suites:
        s_.pop("path", None)
        s_.pop("ai_source", None)
    suites_file.write_text(json.dumps({
        "test_type": test_type,
        "suites": suites,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"  Wrote: {suites_file}", flush=True)

    def _write_reqs(filepath, label, pkgs):
        lines = [
            f"# {label} — auto-generated by generate_suites.py",
            f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Install: python3 -m pip install -r {filepath.name}",
            "",
        ]
        for pkg in pkgs:
            lines.append(pkg)
        lines.append("")
        filepath.write_text("\n".join(lines), encoding="utf-8")
        print(f"  Wrote: {filepath}  ({len(pkgs)} packages)", flush=True)

    _write_reqs(CONFIGS_DIR / "requirements_s390x.txt",
                "s390x VM Requirements (VM-lane suites)", vm_deps)
    _write_reqs(CONFIGS_DIR / "requirements_x86.txt",
                "x86 VM Requirements (LOCAL-lane suites)", local_deps)

    # Summary
    print(f"\n{'='*70}", flush=True)
    print("  SUMMARY", flush=True)
    print("=" * 70, flush=True)
    print(f"  Suites: {len(suites)}", flush=True)
    print(f"  s390x packages: {len(vm_deps)}", flush=True)
    print(f"  x86 packages:   {len(local_deps)}", flush=True)
    print(flush=True)

    for s in suites:
        flags = ""
        if s.get("run_local"):
            flags += " [LOCAL→x86]"
        else:
            flags += " [VM→s390x]"
        if s.get("critical"):
            flags += " [CRITICAL]"
        desc = s.get("description", "")
        print(f"  {s['name']:<30} {str(s.get('tags', [])):<40}{flags}", flush=True)
        if desc:
            print(f"    -> {desc}", flush=True)

    print(f"\n  s390x VM packages:", flush=True)
    for p in vm_deps:
        print(f"    - {p}", flush=True)
    print(f"\n  x86 VM packages:", flush=True)
    for p in local_deps:
        print(f"    - {p}", flush=True)

    # Bob statistics (only if Bob was used)
    if _bob_stats["calls"] > 0:
        print(f"\n  {'─'*40}", flush=True)
        print(f"  IBM Bob Statistics", flush=True)
        print(f"  {'─'*40}", flush=True)
        print(f"  Total calls:    {_bob_stats['calls']}", flush=True)
        print(f"  Successful:     {_bob_stats['success']}", flush=True)
        print(f"  Errors:         {_bob_stats['errors']}", flush=True)
        print(f"  Timeouts:       {_bob_stats['timeouts']}", flush=True)
        print(f"  Setup time:     {_bob_stats['setup_time_sec']}s", flush=True)
        print(f"  Total API time: {round(_bob_stats['total_time_sec'], 2)}s", flush=True)
        if _bob_stats["jobs_analyzed"]:
            avg = round(_bob_stats["total_time_sec"] / len(_bob_stats["jobs_analyzed"]), 2)
            print(f"  Avg per job:    {avg}s", flush=True)
            for j in _bob_stats["jobs_analyzed"]:
                print(f"    ✓ {j['job']} ({j['time_sec']}s)", flush=True)
        if _bob_stats["error_details"]:
            for e in _bob_stats["error_details"]:
                print(f"    ✗ {e.get('job', '?')}: {e.get('error', '?')}", flush=True)

        # Write stats file
        bob_stats_file = CONFIGS_DIR / "bob_stats.json"
        bob_stats_file.write_text(json.dumps(_bob_stats, indent=2) + "\n")
        print(f"  Stats file:     {bob_stats_file}", flush=True)

    print(f"\n  Done.", flush=True)


if __name__ == "__main__":
    main()
