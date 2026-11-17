#!/usr/bin/env python3
"""Write a jenkins_jobs.json parameter block, instead of typing one.

THE MANUAL STEP THIS REMOVES

Onboarding a Jenkins job today means opening it in a browser, reading its
parameter list, and hand-writing twenty blocks like this:

    "CLUSTER_HOST": {
      "value": "auto",
      "source": "env:CLUSTER_NAME",
      "template": "cpd-zen.apps.{CLUSTER_NAME}.example.com",
      "description": "CP4D cluster hostname"
    }

HOW IT WORKS

Jenkins publishes its own parameter definitions, so the parameter names, types
and defaults are FACTS we fetch rather than guesses:

    GET <job>/api/json?tree=property[parameterDefinitions[...]]

The only genuinely ambiguous part is the MAPPING - that Jenkins' CLUSTER_HOST
should come from our CLUSTER_NAME via a template, and that its IBM_USER is our
CPD_USERNAME. Matching names across two vocabularies is what a model is good at,
so that one decision is the model's, and everything around it is checked.

WHAT THE MODEL IS AND IS NOT TRUSTED WITH

  - It sees parameter NAMES, descriptions and defaults, plus the NAMES of
    available environment variables. It never sees a single value, so no
    credential can reach it.
  - It may not invent a parameter: the output is checked against the list
    Jenkins gave us, and anything extra is dropped.
  - It does NOT decide `sensitive`. That is a safety property, determined by
    rule from the parameter name, because a model that forgets one flag leaks a
    password into a console log.
  - Its proposal is written for you to REVIEW as a diff. Nothing is committed and
    nothing is triggered.

Without a model it still works: a deterministic matcher handles exact and
near-exact name matches, which covers most parameters. The model earns its place
on the rest.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK_DIR = SCRIPT_DIR.parent

OLLAMA = os.getenv("OLLAMA_ENDPOINT", "http://127.0.0.1:11434")
MODEL = os.getenv("HERMES_MODEL", "gemma4:12b")

# A {TOKEN} in a job path or template.
TOKEN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Parameter names whose value is a credential. Rule-based on purpose - see the
# module docstring. Mirrors aicp4d.security.SENSITIVE_KEY_TOKENS.
SECRET_TOKENS = ("PASSWORD", "PASSWD", "TOKEN", "SECRET", "APIKEY", "API_KEY",
                 "KEY", "CREDENTIAL", "PASS", "ENTITLEMENT")
# ...except these, which merely contain a secret-looking word.
NOT_SECRET = ("KEY_PATH", "KEYFILE", "KEY_FILE", "PASSWORD_FILE", "TOKEN_PATH",
              "PUBLIC_KEY", "KEYSTORE_PATH")


def is_secret(name: str) -> bool:
    upper = name.upper()
    if any(exempt in upper for exempt in NOT_SECRET):
        return False
    return any(token in upper for token in SECRET_TOKENS)


# ---------------------------------------------------------------------------
# .env — names only
# ---------------------------------------------------------------------------

def env_var_names(env_file: Path | None = None) -> list[str]:
    """The NAMES of variables available for mapping. Values are never read.

    Deliberate: this list goes into a model prompt, and a value never does.
    """
    path = env_file or (FRAMEWORK_DIR / ".env")
    names: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                names.append(line.partition("=")[0].strip())
    return sorted(set(names))


def env_values(env_file: Path | None = None) -> dict[str, str]:
    """Values, used ONLY to expand a job path locally before fetching it."""
    path = env_file or (FRAMEWORK_DIR / ".env")
    values = dict(os.environ)
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip().strip("'\"")
    return values


# ---------------------------------------------------------------------------
# Jenkins introspection
# ---------------------------------------------------------------------------

@dataclass
class ParamDef:
    """One parameter, exactly as Jenkins describes it. Ground truth."""
    name: str
    type: str = "StringParameterDefinition"
    default: str = ""
    description: str = ""
    choices: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        if self.choices:
            return "choice"
        low = self.type.lower()
        if "boolean" in low:
            return "bool"
        if "password" in low:
            return "password"
        return "string"


def _opener(user: str, token: str):
    handler = urllib.request.HTTPSHandler()
    opener = urllib.request.build_opener(handler)
    if user and token:
        cred = base64.b64encode(f"{user}:{token}".encode()).decode()
        opener.addheaders = [("Authorization", f"Basic {cred}")]
    return opener


def fetch_parameters(jenkins_url: str, job_path: str, *, user: str = "",
                     token: str = "", timeout: int = 30) -> list[ParamDef]:
    """Ask Jenkins what parameters this job takes."""
    base = f"{jenkins_url.rstrip('/')}/{job_path.strip('/')}"
    tree = ("property[parameterDefinitions[name,type,description,"
            "defaultParameterValue[value],choices]]")
    url = f"{base}/api/json?tree={urllib.parse.quote(tree, safe='[],')}"
    opener = _opener(user, token)
    try:
        with opener.open(url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise SystemExit(
                f"Jenkins refused the request ({exc.code}). Set JENKINS_USER and "
                f"JENKINS_API_TOKEN in {FRAMEWORK_DIR / '.env'}.\n"
                f"  Token: Jenkins → your user → Configure → API Token")
        if exc.code == 404:
            raise SystemExit(
                f"no such job: {base}\n"
                f"  Check --job-path. It is the URL path, e.g.\n"
                f"  job/test/job/Suite/job/My_Job")
        raise SystemExit(f"Jenkins returned HTTP {exc.code} for {base}")
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        raise SystemExit(f"cannot reach {jenkins_url}: {exc}")

    if "<html" in body[:400].lower():
        raise SystemExit(
            "Jenkins returned a login page instead of JSON - authentication "
            "failed. Set JENKINS_USER and JENKINS_API_TOKEN in .env.")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        raise SystemExit(f"Jenkins returned non-JSON for {base}")

    params: list[ParamDef] = []
    for prop in data.get("property", []):
        for definition in prop.get("parameterDefinitions", []) or []:
            name = definition.get("name", "")
            if not name:
                continue
            default = (definition.get("defaultParameterValue") or {}).get("value", "")
            params.append(ParamDef(
                name=name,
                type=definition.get("type", "StringParameterDefinition"),
                default="" if default is None else str(default),
                description=definition.get("description", "") or "",
                choices=tuple(definition.get("choices", []) or ()),
            ))
    if not params:
        raise SystemExit(
            f"{base} declares no build parameters.\n"
            f"  Either it is not parameterised, or the token lacks read access.")
    return params


# ---------------------------------------------------------------------------
# Mapping: deterministic first, model for the remainder
# ---------------------------------------------------------------------------

def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


# Vocabulary differences seen across the five existing configs. Checked before
# the model is asked, because a known answer should not cost a model call.
KNOWN_ALIASES = {
    "clusteruser": "CPD_USERNAME", "ibmuser": "CPD_USERNAME",
    "iamuser": "CPD_USERNAME", "cpduser": "CPD_USERNAME",
    "clusterpassword": "CPD_PASSWORD", "ibmpassword": "CPD_PASSWORD",
    "adminpasswd": "CPD_PASSWORD", "cpdpassword": "CPD_PASSWORD",
    "openshiftpassword": "OPENSHIFT_PASSWORD",
    "openshiftuser": "OPENSHIFT_USER",
    "clusternamespace": "CPD_NAMESPACE", "namespace": "CPD_NAMESPACE",
    "cp4dapikey": "CP4D_APIKEY", "clusterapikey": "CP4D_APIKEY",
    "db2host": "DB2_HOST", "db2username": "DB2_USERNAME",
    "db2password": "DB2_PASSWORD",
}

# Parameters that are built from the cluster name rather than read directly.
# The pattern is recognisable: a hostname or URL containing the cluster name.
HOST_HINTS = ("host", "url", "endpoint", "route", "server")


def deterministic_mapping(params: list[ParamDef],
                          env_names: list[str]) -> dict[str, dict]:
    """Everything that can be decided without a model. Usually most of it."""
    by_norm = {_normalise(n): n for n in env_names}
    mapping: dict[str, dict] = {}

    for param in params:
        norm = _normalise(param.name)
        entry: dict = {}

        if param.kind == "choice":
            entry = {"value": param.default or param.choices[0],
                     "source": "default",
                     "enum": list(param.choices)}
        elif norm in by_norm:
            entry = {"value": "auto", "source": f"env:{by_norm[norm]}"}
        elif norm in KNOWN_ALIASES and KNOWN_ALIASES[norm] in env_names:
            entry = {"value": "auto", "source": f"env:{KNOWN_ALIASES[norm]}"}
        elif param.default:
            entry = {"value": param.default, "source": "default"}

        if entry:
            entry["description"] = param.description[:160]
            if is_secret(param.name):
                entry["sensitive"] = True
                entry["value"] = "auto"
            mapping[param.name] = entry
    return mapping


_SYSTEM = """\
You map Jenkins build parameters onto available environment variable names.

Return ONE JSON object and nothing else. For each parameter you are asked about:

  "PARAM_NAME": {
    "source": "env:VARNAME"   (read a variable directly)
              | "default"     (use a fixed value)
    "value":  "auto"          (with env:)  |  the literal (with default)
    "template": "..."         (optional; build the value, e.g.
                               "cpd-zen.apps.{CLUSTER_NAME}.example.com")
  }

Rules:
- Use ONLY variable names from the provided list. Never invent one.
- Only include parameters you were asked about. Do not add others.
- A hostname or URL parameter is usually built with a "template" from a cluster
  name variable, not read directly.
- Every {TOKEN} in a template must be a variable from the list.
- If you cannot map a parameter confidently, omit it. Omission is safe; a wrong
  mapping points a build at the wrong system.
"""


def model_mapping(unmapped: list[ParamDef], env_names: list[str],
                  timeout: int = 120) -> dict[str, dict]:
    """Ask the local model about whatever the rules could not decide.

    Only names and descriptions are sent. Returns {} on any failure, so the
    command still produces a reviewable block with a model that is down.
    """
    if not unmapped:
        return {}
    described = "\n".join(
        f"  {p.name}  (type={p.kind}"
        + (f", default={p.default!r}" if p.default else "")
        + ")" + (f"  -- {p.description[:100]}" if p.description else "")
        for p in unmapped)
    prompt = (f"Available environment variable names:\n"
              f"  {', '.join(env_names)}\n\n"
              f"Jenkins parameters to map:\n{described}\n")

    body = {"model": MODEL,
            "messages": [{"role": "system", "content": _SYSTEM},
                         {"role": "user", "content": prompt}],
            "stream": False, "format": "json",
            "options": {"temperature": 0}}
    request = urllib.request.Request(
        f"{OLLAMA}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, OSError, TimeoutError,
            json.JSONDecodeError) as exc:
        print(f"  [model] unavailable ({exc}); rules-only mapping", file=sys.stderr)
        return {}

    text = data.get("message", {}).get("content", "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            parsed = json.loads(match.group()) if match else {}
        except (json.JSONDecodeError, AttributeError):
            parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def harden(proposal: dict, params: list[ParamDef],
           env_names: list[str]) -> tuple[dict[str, dict], list[str]]:
    """Check the model's proposal. Drops anything unsafe; reports every drop.

    This is the function that makes a model-authored mapping trustworthy. It
    enforces what the prompt merely requests.
    """
    valid_names = {p.name for p in params}
    by_name = {p.name: p for p in params}
    clean: dict[str, dict] = {}
    notes: list[str] = []

    for name, entry in proposal.items():
        if name not in valid_names:
            notes.append(f"dropped {name!r}: Jenkins does not declare it")
            continue
        if not isinstance(entry, dict):
            notes.append(f"dropped {name!r}: not an object")
            continue

        source = str(entry.get("source", ""))
        template = str(entry.get("template", "") or "")
        out: dict = {}

        if source.startswith("env:"):
            var = source.split(":", 1)[1]
            if var not in env_names:
                notes.append(
                    f"dropped {name!r}: source names {var!r}, which is not in .env")
                continue
            out = {"value": "auto", "source": source}
        elif source == "default":
            out = {"value": str(entry.get("value", "")), "source": "default"}
        else:
            notes.append(f"dropped {name!r}: source {source!r} is not recognised")
            continue

        if template:
            unknown = [t for t in TOKEN.findall(template) if t not in env_names]
            if unknown:
                notes.append(
                    f"dropped template on {name!r}: unknown token(s) "
                    f"{', '.join(unknown)}")
            else:
                out["template"] = template

        description = by_name[name].description[:160]
        if description:
            out["description"] = description
        # The model does not get a vote on this.
        if is_secret(name):
            out["sensitive"] = True
            out["value"] = "auto"
        clean[name] = out
    return clean, notes


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def build_block(job_name: str, job_path: str, mapping: dict[str, dict],
                *, description: str = "", timeout_sec: int = 7200) -> dict:
    return {"name": job_name,
            "job_path": job_path,
            "description": description or f"Jenkins job: {job_name}",
            "timeout_sec": timeout_sec,
            "parameters": mapping,
            "wait_for_result": True}


def merge_into(config_path: Path, block: dict, jenkins_url: str) -> str:
    """Add or replace this job in a jenkins_jobs.json. Returns what changed."""
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        verb = "updated"
    else:
        config = {"jenkins_url": jenkins_url,
                  "env_file": "source.env",
                  "instructions": ("Parameter mappings are generated by "
                                   "jenkins_introspect.py and reviewed by hand. "
                                   "Sensitive values must never be logged."),
                  "jobs": []}
        verb = "created"
    config.setdefault("jenkins_url", jenkins_url)
    jobs = config.setdefault("jobs", [])
    for index, existing in enumerate(jobs):
        if existing.get("name") == block["name"]:
            jobs[index] = block
            break
    else:
        jobs.append(block)
        verb = "added to" if verb == "updated" else verb
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
    return verb


def report(params: list[ParamDef], mapping: dict[str, dict],
           notes: list[str]) -> None:
    width = max((len(p.name) for p in params), default=0)
    print(f"\nJenkins declares {len(params)} parameter(s); "
          f"mapped {len(mapping)}.\n")
    for param in params:
        entry = mapping.get(param.name)
        if not entry:
            print(f"  {'UNMAPPED':<10} {param.name:<{width}}  "
                  f"-- {param.description[:60] or 'no description'}")
            continue
        source = entry.get("source", "")
        detail = entry.get("template") or source
        flag = " [secret]" if entry.get("sensitive") else ""
        print(f"  {'ok':<10} {param.name:<{width}}  {detail}{flag}")
    if notes:
        print("\nrejected by validation:")
        for note in notes:
            print(f"  - {note}")
    unmapped = [p.name for p in params if p.name not in mapping]
    if unmapped:
        print(f"\n{len(unmapped)} parameter(s) need a decision from you: "
              f"{', '.join(unmapped)}")
        print("  Add them by hand, or re-run once the matching variable exists "
              "in .env.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="jenkins_introspect.py",
        description="Generate a jenkins_jobs.json parameter block from a live "
                    "Jenkins job.")
    parser.add_argument("--jenkins-url", default="",
                        help="e.g. https://jenkins.example.com")
    parser.add_argument("--job-path", required=True,
                        help="URL path, e.g. job/test/job/Suite/job/My_Job")
    parser.add_argument("--name", default="",
                        help="name for this job in the config "
                             "(default: last path segment, lowercased)")
    parser.add_argument("--config", default="",
                        help="jenkins_jobs.json to write (default: print only)")
    parser.add_argument("--env-file", default="",
                        help="where variable NAMES come from (default: ../.env)")
    parser.add_argument("--no-model", action="store_true",
                        help="deterministic matching only")
    parser.add_argument("--write", action="store_true",
                        help="actually write --config (default: preview)")
    args = parser.parse_args(argv)

    env_file = Path(args.env_file) if args.env_file else None
    values = env_values(env_file)
    names = env_var_names(env_file)
    if not names:
        print(f"warning: no variable names found in "
              f"{env_file or FRAMEWORK_DIR / '.env'}", file=sys.stderr)

    jenkins_url = args.jenkins_url or values.get("JENKINS_URL", "")
    if not jenkins_url:
        print("error: --jenkins-url is required (or set JENKINS_URL in .env)",
              file=sys.stderr)
        return 2

    # A job path may carry {CPD_VERSION}-style tokens. Expand before fetching.
    job_path, missing = args.job_path, []
    for token in TOKEN.findall(args.job_path):
        if values.get(token):
            job_path = job_path.replace(f"{{{token}}}", values[token])
        else:
            missing.append(token)
    if missing:
        print(f"error: --job-path needs {', '.join(missing)}, which is not set "
              f"in the environment or .env", file=sys.stderr)
        return 2

    job_name = args.name or re.sub(
        r"[^a-z0-9-]+", "-", job_path.rstrip("/").split("/")[-1].lower()).strip("-")

    print(f"job    : {job_name}")
    print(f"path   : {job_path}")
    print(f"jenkins: {jenkins_url}")

    params = fetch_parameters(
        jenkins_url, job_path,
        user=values.get("JENKINS_USER", ""),
        token=values.get("JENKINS_API_TOKEN", ""))

    mapping = deterministic_mapping(params, names)
    notes: list[str] = []
    unmapped = [p for p in params if p.name not in mapping]

    if unmapped and not args.no_model:
        print(f"\n{len(mapping)} mapped by rule; asking {MODEL} about "
              f"{len(unmapped)} remaining (names only, no values sent)")
        proposal = model_mapping(unmapped, names)
        accepted, notes = harden(proposal, params, names)
        for name, entry in accepted.items():
            mapping.setdefault(name, entry)

    report(params, mapping, notes)

    block = build_block(job_name, args.job_path, mapping)
    print("\n" + "-" * 68)
    print(json.dumps(block, indent=4))
    print("-" * 68)

    if args.config:
        target = Path(args.config)
        if args.write:
            verb = merge_into(target, block, jenkins_url)
            print(f"\n{verb} {target}")
            print("Review the diff before committing - the mapping is a "
                  "proposal, not a verified fact.")
        else:
            print(f"\npreview only. Re-run with --write to update {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
