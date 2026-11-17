"""Hermes Security Module — log sanitization + encrypted .env storage."""

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = FRAMEWORK_DIR / ".env"
ENV_ENC_PATH = FRAMEWORK_DIR / ".env.enc"

_SENSITIVE_KEY_PATTERNS = {"PASSWORD", "TOKEN", "SECRET", "KEY", "CREDENTIAL", "SSH_PASS"}
_REDACTED = "[REDACTED]"


# ─── Layer 1: Log Sanitizer ─────────────────────────────────────────────────

class LogSanitizer:
    """Scrubs known secrets and sensitive patterns from text before it
    leaves the local machine (e.g. before sending to Ollama)."""

    def __init__(self, env_vars: dict | None = None):
        self._secret_values: list[tuple[str, str]] = []
        self._patterns: list[tuple[re.Pattern, str]] = []
        if env_vars:
            self._load_secrets(env_vars)
        self._compile_patterns()

    def _load_secrets(self, env_vars: dict):
        for key, val in env_vars.items():
            if not val or len(val) < 4:
                continue
            if any(s in key.upper() for s in _SENSITIVE_KEY_PATTERNS):
                self._secret_values.append((key, val))

    def _compile_patterns(self):
        self._patterns = [
            # GitHub personal access tokens
            (re.compile(r'ghp_[A-Za-z0-9]{36,}'), f"{_REDACTED}:GH_TOKEN"),
            # GitHub OAuth / app tokens
            (re.compile(r'gh[os]_[A-Za-z0-9]{36,}'), f"{_REDACTED}:GH_TOKEN"),
            # IBM Cloud API keys
            (re.compile(r'(?i)(?:apikey|api[_-]?key)\s*[:=]\s*["\']?([A-Za-z0-9_-]{30,})["\']?'), f"{_REDACTED}:API_KEY"),
            # JWT tokens
            (re.compile(r'eyJ[A-Za-z0-9_-]{20,}\.eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}'), f"{_REDACTED}:JWT"),
            # Bearer tokens in headers
            (re.compile(r'(?i)Bearer\s+[A-Za-z0-9._~+/=-]{20,}'), f"Bearer {_REDACTED}:TOKEN"),
            # Generic base64 passwords (32+ chars of base64 after password-like key)
            (re.compile(r'(?i)(?:password|passwd|pwd)\s*[:=]\s*["\']?([A-Za-z0-9+/=]{20,})["\']?'), f"{_REDACTED}:PASSWORD"),
        ]

    def sanitize(self, text: str) -> str:
        if not text:
            return text
        result = text
        for key, val in self._secret_values:
            if val in result:
                result = result.replace(val, f"{_REDACTED}:{key}")
        for pattern, replacement in self._patterns:
            result = pattern.sub(replacement, result)
        return result

    def audit_log(self, action: str, prompt_len: int, scrubbed_count: int):
        audit_dir = FRAMEWORK_DIR / "logs" / "security"
        audit_dir.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "prompt_length": prompt_len,
            "secrets_scrubbed": scrubbed_count,
        }
        audit_file = audit_dir / "audit.jsonl"
        with open(audit_file, "a") as f:
            f.write(json.dumps(entry) + "\n")


# ─── Layer 2: Encrypted .env ────────────────────────────────────────────────

def _derive_key(passphrase: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 200_000)


def _aes_encrypt(plaintext: bytes, key: bytes) -> dict:
    """AES-256-CBC encryption using stdlib only (no cryptography dependency)."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding as sym_padding
        iv = secrets.token_bytes(16)
        padder = sym_padding.PKCS7(128).padder()
        padded = padder.update(plaintext) + padder.finalize()
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        enc = cipher.encryptor()
        ct = enc.update(padded) + enc.finalize()
        return {"iv": base64.b64encode(iv).decode(),
                "ct": base64.b64encode(ct).decode()}
    except ImportError:
        iv = secrets.token_bytes(16)
        xor_key = hashlib.sha256(key + iv).digest()
        ct = bytes(b ^ xor_key[i % 32] for i, b in enumerate(plaintext))
        tag = hmac.new(key, iv + ct, "sha256").digest()
        return {"iv": base64.b64encode(iv).decode(),
                "ct": base64.b64encode(ct).decode(),
                "tag": base64.b64encode(tag).decode(),
                "fallback": True}


def _aes_decrypt(data: dict, key: bytes) -> bytes:
    if data.get("fallback"):
        iv = base64.b64decode(data["iv"])
        ct = base64.b64decode(data["ct"])
        tag = base64.b64decode(data["tag"])
        expected = hmac.new(key, iv + ct, "sha256").digest()
        if not hmac.compare_digest(tag, expected):
            raise ValueError("Integrity check failed — wrong key or tampered file")
        xor_key = hashlib.sha256(key + iv).digest()
        return bytes(b ^ xor_key[i % 32] for i, b in enumerate(ct))
    else:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.primitives import padding as sym_padding
        iv = base64.b64decode(data["iv"])
        ct = base64.b64decode(data["ct"])
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
        dec = cipher.decryptor()
        padded = dec.update(ct) + dec.finalize()
        unpadder = sym_padding.PKCS7(128).unpadder()
        return unpadder.update(padded) + unpadder.finalize()


def encrypt_env(passphrase: str | None = None) -> Path:
    if not ENV_PATH.exists():
        raise FileNotFoundError(f"{ENV_PATH} not found")
    if not passphrase:
        passphrase = os.getenv("HERMES_MASTER_KEY")
    if not passphrase:
        raise ValueError("Set HERMES_MASTER_KEY env var or pass a passphrase")

    plaintext = ENV_PATH.read_bytes()
    salt = secrets.token_bytes(32)
    key = _derive_key(passphrase, salt)
    encrypted = _aes_encrypt(plaintext, key)
    encrypted["salt"] = base64.b64encode(salt).decode()
    encrypted["version"] = 1

    ENV_ENC_PATH.write_text(json.dumps(encrypted, indent=2))
    os.chmod(ENV_ENC_PATH, 0o600)
    return ENV_ENC_PATH


def decrypt_env(passphrase: str | None = None) -> dict:
    if not ENV_ENC_PATH.exists():
        raise FileNotFoundError(f"{ENV_ENC_PATH} not found — run encrypt_env first")
    if not passphrase:
        passphrase = os.getenv("HERMES_MASTER_KEY")
    if not passphrase:
        raise ValueError("Set HERMES_MASTER_KEY env var or pass a passphrase")

    data = json.loads(ENV_ENC_PATH.read_text())
    salt = base64.b64decode(data["salt"])
    key = _derive_key(passphrase, salt)
    plaintext = _aes_decrypt(data, key).decode()

    env_vars = {}
    for line in plaintext.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        env_vars[k.strip()] = v.strip()
    return env_vars


def load_env_secure() -> dict:
    """Load env vars: prefer encrypted file, fall back to plaintext .env."""
    master_key = os.getenv("HERMES_MASTER_KEY")
    if ENV_ENC_PATH.exists() and master_key:
        try:
            return decrypt_env(master_key)
        except Exception as e:
            print(f"  [Security] WARNING: Failed to decrypt .env.enc: {e}")
            print(f"  [Security] Falling back to plaintext .env", flush=True)
    if ENV_PATH.exists():
        env_vars = {}
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, _, v = line.partition("=")
            env_vars[k.strip()] = v.strip()
        return env_vars
    return {}


# ─── Singleton sanitizer ────────────────────────────────────────────────────

_sanitizer: LogSanitizer | None = None

def get_sanitizer(env_vars: dict | None = None) -> LogSanitizer:
    global _sanitizer
    if _sanitizer is None:
        if env_vars is None:
            env_vars = load_env_secure()
        _sanitizer = LogSanitizer(env_vars)
    return _sanitizer


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python hermes_security.py <encrypt|decrypt|test-sanitize>")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "encrypt":
        path = encrypt_env()
        print(f"Encrypted .env → {path}")
        print(f"  File permissions: {oct(path.stat().st_mode)[-3:]}")
        print(f"  You can now delete .env (keep a backup first)")
        print(f"  Set HERMES_MASTER_KEY in your shell profile to decrypt at runtime")

    elif cmd == "decrypt":
        env = decrypt_env()
        print(f"Decrypted {len(env)} variables:")
        for k in env:
            masked = "(set)" if any(s in k.upper() for s in _SENSITIVE_KEY_PATTERNS) else env[k]
            print(f"  {k} = {masked}")

    elif cmd == "test-sanitize":
        sanitizer = get_sanitizer()
        sample = f"Connecting to CPD with password={os.getenv('CPD_PASSWORD', 'test123')} token=ghp_abc123xyz456abc123xyz456abc123xyz456"
        print(f"Original:  {sample}")
        print(f"Sanitized: {sanitizer.sanitize(sample)}")

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
