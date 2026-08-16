"""Run repository verification without access to platform credentials."""

from __future__ import annotations

import json
import os
import re
import signal
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

MAX_PROJECTS = 6
MAX_OUTPUT_CHARS = 30_000
COMMAND_TIMEOUT = int(os.getenv("IMPLEMENTATION_COMMAND_TIMEOUT_SECONDS", "300"))
VERIFICATION_DEADLINE = time.monotonic() + int(
    os.getenv("IMPLEMENTATION_VERIFICATION_DEADLINE_SECONDS", "840")
)


def main() -> int:
    try:
        source = Path("/workspace/source")
        evidence = Path("/workspace/evidence")
        work = Path("/workspace/work/project")
        evidence.mkdir(parents=True, exist_ok=True)
        if not source.is_dir():
            raise RuntimeError("Immutable source snapshot is missing")
        shutil.copytree(source, work, symlinks=True, dirs_exist_ok=True)

        results: list[dict[str, Any]] = []
        projects = _discover_projects(work)
        recognized_project = False
        for project in projects[:MAX_PROJECTS]:
            if (project / "package.json").is_file():
                recognized_project = True
                results.extend(_verify_node(project, work))
            if _is_python_project(project):
                recognized_project = True
                results.extend(_verify_python(project, work))
        if not recognized_project:
            static_web_result = _verify_static_web(work)
            if static_web_result:
                results.append(static_web_result)
        results.append(_secret_scan(work))

        command_failures = [item for item in results if item["status"] == "failed"]
        attempted = [item for item in results if item["status"] in {"passed", "failed"}]
        categories = {
            category: any(
                item.get("category") == category and item["status"] in {"passed", "failed"}
                for item in results
            )
            for category in ["install", "test", "build", "lint", "security"]
        }
        report = {
            "schemaVersion": 1,
            "complete": True,
            "projectsDiscovered": [path.relative_to(work).as_posix() or "." for path in projects],
            "commandsAttempted": len(attempted),
            "commandsFailed": len(command_failures),
            "coverage": categories,
            "results": results,
        }
        (evidence / "verification.json").write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        print(
            f"Verification completed: {len(attempted)} checks, "
            f"{len(command_failures)} failures"
        )
        # Verification failures are evidence for the evaluator, not an init
        # failure. Exit zero so the final evaluator can issue a revision verdict.
        return 0
    except Exception as exc:
        Path("/workspace/evidence").mkdir(parents=True, exist_ok=True)
        (Path("/workspace/evidence") / "verification.json").write_text(
            json.dumps(
                {
                    "schemaVersion": 1,
                    "complete": False,
                    "error": str(exc)[:1000],
                    "coverage": {},
                    "results": [],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        print(f"Verification infrastructure failed: {exc}", file=sys.stderr)
        return 0


def _discover_projects(root: Path) -> list[Path]:
    ignored = {".git", "node_modules", ".venv", "venv", "dist", "build", ".next"}
    found: set[Path] = set()
    for marker in ["package.json", "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"]:
        for path in root.rglob(marker):
            relative_parts = path.relative_to(root).parts
            if any(part in ignored for part in relative_parts) or len(relative_parts) > 4:
                continue
            found.add(path.parent)
    return sorted(found, key=lambda path: (len(path.relative_to(root).parts), str(path))) or [root]


def _is_python_project(path: Path) -> bool:
    return any((path / marker).is_file() for marker in ["pyproject.toml", "requirements.txt", "setup.py", "setup.cfg"])


def _verify_static_web(root: Path) -> dict[str, Any] | None:
    """Give dependency-free static sites an objective, proportionate check."""
    ignored = {".git", "node_modules", "dist", "build", ".next"}
    html_files = [
        path
        for path in root.rglob("*.html")
        if path.is_file()
        and not path.is_symlink()
        and not any(part in ignored for part in path.relative_to(root).parts)
    ][:100]
    if not html_files:
        return None

    failures: list[str] = []
    checked_refs = 0
    reference_pattern = re.compile(
        r"\b(src|href)\s*=\s*(['\"])(.*?)\2", re.IGNORECASE
    )
    for html_file in html_files:
        try:
            if html_file.stat().st_size > 2_000_000:
                failures.append(f"{html_file.relative_to(root)} is too large to inspect")
                continue
            content = html_file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            failures.append(f"{html_file.relative_to(root)} is unreadable: {exc}")
            continue
        if not content.strip():
            failures.append(f"{html_file.relative_to(root)} is empty")
            continue
        for match in reference_pattern.finditer(content):
            attribute = match.group(1).lower()
            reference = match.group(3).strip()
            if not reference or reference.startswith(("#", "data:", "http:", "https:", "mailto:", "tel:", "javascript:")):
                continue
            clean = reference.split("#", 1)[0].split("?", 1)[0]
            if not clean:
                continue
            # Extensionless href values are often client-side routes, not files.
            if attribute == "href" and not Path(clean).suffix:
                continue
            target = root / clean.lstrip("/") if clean.startswith("/") else html_file.parent / clean
            try:
                target.resolve().relative_to(root.resolve())
            except ValueError:
                failures.append(
                    f"{html_file.relative_to(root)} references a path outside the repository: {reference}"
                )
                continue
            checked_refs += 1
            if not target.is_file():
                failures.append(
                    f"{html_file.relative_to(root)} has a missing local asset: {reference}"
                )
        if len(failures) >= 100:
            break

    if failures:
        return _static_result(
            ".",
            "build",
            "static-web-sanity",
            False,
            "; ".join(failures[:100]),
        )
    return _static_result(
        ".",
        "build",
        "static-web-sanity",
        True,
        f"Validated {len(html_files)} HTML file(s) and {checked_refs} local asset reference(s).",
    )


def _verify_node(project: Path, root: Path) -> list[dict[str, Any]]:
    relative = project.relative_to(root).as_posix() or "."
    try:
        package = json.loads((project / "package.json").read_text(encoding="utf-8"))
    except Exception as exc:
        return [_static_result(relative, "install", "node-package-manifest", False, f"Invalid package.json: {exc}")]
    scripts = package.get("scripts") if isinstance(package, dict) else {}
    if not isinstance(scripts, dict):
        scripts = {}
    lock = project / "package-lock.json"
    install = ["npm", "ci" if lock.is_file() else "install", "--ignore-scripts", "--no-audit", "--no-fund"]
    results = [_run(relative, project, "install", "node-dependencies", install, COMMAND_TIMEOUT)]
    if results[-1]["status"] == "failed":
        return results + [
            _skipped(relative, category, f"node-{category}", "Dependency installation failed")
            for category in ["lint", "test", "build", "security"]
        ]
    if "lint" in scripts:
        results.append(
            _run(
                relative,
                project,
                "lint",
                "node-lint",
                ["npm", "run", "lint"],
                COMMAND_TIMEOUT,
            )
        )
    else:
        results.append(_skipped(relative, "lint", "node-lint", "No lint script is declared"))
    if "test" in scripts:
        package_text = json.dumps(package).lower()
        if "jest" in package_text:
            test_command = ["npm", "test", "--", "--runInBand"]
        elif "vitest" in package_text:
            test_command = ["npm", "test", "--", "--run"]
        else:
            test_command = ["npm", "test"]
        results.append(_run(relative, project, "test", "node-tests", test_command, COMMAND_TIMEOUT))
    else:
        results.append(_skipped(relative, "test", "node-tests", "No test script is declared"))
    extra_test_scripts = sorted(
        name
        for name in scripts
        if name != "test" and re.search(r"(?:^|:)(?:integration|e2e|contract)(?::|$)", name)
    )
    for script_name in extra_test_scripts[:5]:
        results.append(
            _run(
                relative,
                project,
                "test",
                f"node-{script_name}",
                ["npm", "run", script_name],
                COMMAND_TIMEOUT,
            )
        )
    if "build" in scripts:
        results.append(_run(relative, project, "build", "node-build", ["npm", "run", "build"], COMMAND_TIMEOUT))
    else:
        results.append(_skipped(relative, "build", "node-build", "No build script is declared"))
    results.append(
        _run(
            relative,
            project,
            "security",
            "node-audit",
            ["npm", "audit", "--omit=dev", "--audit-level=high", "--json"],
            COMMAND_TIMEOUT,
        )
    )
    return results


def _verify_python(project: Path, root: Path) -> list[dict[str, Any]]:
    relative = project.relative_to(root).as_posix() or "."
    venv = project / ".nexus-evaluation-venv"
    results = [_run(relative, project, "install", "python-venv", [sys.executable, "-m", "venv", str(venv)], 90)]
    python = venv / "bin" / "python"
    if results[-1]["status"] == "failed":
        return results + [
            _skipped(relative, category, f"python-{category}", "Virtual environment creation failed")
            for category in ["install", "lint", "test", "build", "security"]
        ]
    install_target: list[str]
    if (project / "requirements.txt").is_file():
        install_target = ["-r", "requirements.txt"]
    else:
        install_target = ["."]
    results.append(
        _run(relative, project, "install", "python-dependencies", [str(python), "-m", "pip", "install", "--disable-pip-version-check", *install_target], COMMAND_TIMEOUT)
    )
    if results[-1]["status"] == "failed":
        return results + [
            _skipped(relative, category, f"python-{category}", "Dependency installation failed")
            for category in ["lint", "test", "build", "security"]
        ]
    if _module_available(python, project, "ruff"):
        results.append(_run(relative, project, "lint", "python-ruff", [str(python), "-m", "ruff", "check", "."], COMMAND_TIMEOUT))
    else:
        results.append(_skipped(relative, "lint", "python-ruff", "Ruff is not installed by the project"))
    if _module_available(python, project, "pytest"):
        results.append(_run(relative, project, "test", "python-pytest", [str(python), "-m", "pytest", "-q"], COMMAND_TIMEOUT))
    else:
        results.append(_run(relative, project, "test", "python-unittest", [str(python), "-m", "unittest", "discover", "-v"], COMMAND_TIMEOUT))
    results.append(_run(relative, project, "security", "python-package-check", [str(python), "-m", "pip", "check"], 90))
    return results


def _module_available(python: Path, cwd: Path, module: str) -> bool:
    remaining = _remaining_timeout(20)
    if remaining <= 0:
        return False
    process = subprocess.Popen(
        [str(python), "-m", "pip", "show", module],
        cwd=cwd,
        env=_safe_env(cwd),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        return process.wait(timeout=remaining) == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        _kill_process_group(process.pid)


def _run(project: str, cwd: Path, category: str, name: str, argv: list[str], timeout: int) -> dict[str, Any]:
    started = time.monotonic()
    timeout = _remaining_timeout(timeout)
    if timeout <= 0:
        return _static_result(
            project,
            category,
            name,
            False,
            "The global verification time budget was exhausted before this check ran.",
        )
    try:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=_safe_env(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        captured = bytearray()
        captured_total = [0]

        def drain_output() -> None:
            if process.stdout is None:
                return
            for chunk in iter(lambda: process.stdout.read(64 * 1024), b""):
                captured_total[0] += len(chunk)
                captured.extend(chunk)
                maximum = MAX_OUTPUT_CHARS * 4
                if len(captured) > maximum:
                    del captured[: len(captured) - maximum]

        reader = threading.Thread(target=drain_output, daemon=True)
        reader.start()
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_process_group(process.pid)
            process.wait(timeout=10)
        finally:
            _kill_process_group(process.pid)
            reader.join(timeout=5)
            if process.stdout is not None:
                process.stdout.close()
        stdout = captured.decode("utf-8", errors="replace")
        if timed_out:
            return {
                "project": project,
                "category": category,
                "name": name,
                "command": argv,
                "status": "failed",
                "exitCode": None,
                "durationSeconds": round(time.monotonic() - started, 3),
                "output": _sanitize(stdout)
                + f"\nTimed out after {timeout} seconds.",
                "outputTruncated": True,
            }
        output = _sanitize(stdout)
        semantic_failure = name == "python-unittest" and bool(
            re.search(r"\bRan 0 tests?\b", stdout)
        )
        passed = process.returncode == 0 and not semantic_failure
        if semantic_failure:
            output = _sanitize(
                f"{stdout}\nNo automated tests were discovered by unittest."
            )
        return {
            "project": project,
            "category": category,
            "name": name,
            "command": argv,
            "status": "passed" if passed else "failed",
            "exitCode": process.returncode if not semantic_failure else 1,
            "durationSeconds": round(time.monotonic() - started, 3),
            "output": output,
            "outputTruncated": captured_total[0] > len(captured)
            or len(stdout) > len(output),
        }
    except Exception as exc:
        return _static_result(project, category, name, False, str(exc))


def _kill_process_group(process_group_id: int) -> None:
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except OSError:
        pass


def _remaining_timeout(requested: int) -> int:
    return min(requested, max(0, int(VERIFICATION_DEADLINE - time.monotonic())))


def _safe_env(cwd: Path) -> dict[str, str]:
    allowed = {"PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    home = cwd / ".nexus-evaluation-home"
    home.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "CI": "true",
            "NO_COLOR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "NPM_CONFIG_UPDATE_NOTIFIER": "false",
        }
    )
    return env


def _secret_scan(root: Path) -> dict[str, Any]:
    patterns = [
        re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
        re.compile(r"\bgh[ps]_[A-Za-z0-9]{30,}\b"),
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
        re.compile(r"(?i)(?:api[_-]?key|secret|password)\s*[=:]\s*['\"][^'\"\s]{12,}['\"]"),
    ]
    findings: list[str] = []
    ignored = {"node_modules", ".git", ".venv", ".nexus-evaluation-venv", "dist", "build", ".next"}
    for path in root.rglob("*"):
        if not path.is_file() or path.is_symlink() or any(part in ignored for part in path.relative_to(root).parts):
            continue
        try:
            if path.stat().st_size > 1_000_000:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if any(pattern.search(text) for pattern in patterns):
            findings.append(path.relative_to(root).as_posix())
            if len(findings) >= 100:
                break
    return {
        "project": ".",
        "category": "security",
        "name": "credential-pattern-scan",
        "command": [],
        "status": "failed" if findings else "passed",
        "exitCode": 1 if findings else 0,
        "durationSeconds": 0,
        "output": "Potential credential material found in: " + ", ".join(findings) if findings else "No high-confidence committed credential patterns found.",
        "outputTruncated": len(findings) >= 100,
    }


def _sanitize(value: str) -> str:
    sanitized = re.sub(r"Bearer\s+[A-Za-z0-9._~-]+", "Bearer [redacted]", value, flags=re.IGNORECASE)
    sanitized = re.sub(
        r"(?i)(api[_-]?key|token|secret|password)(\s*[=:]\s*)[^\s,;]+",
        r"\1\2[redacted]",
        sanitized,
    )
    return sanitized[-MAX_OUTPUT_CHARS:]


def _static_result(project: str, category: str, name: str, passed: bool, output: str) -> dict[str, Any]:
    return {
        "project": project,
        "category": category,
        "name": name,
        "command": [],
        "status": "passed" if passed else "failed",
        "exitCode": 0 if passed else 1,
        "durationSeconds": 0,
        "output": _sanitize(output),
        "outputTruncated": False,
    }


def _skipped(project: str, category: str, name: str, reason: str) -> dict[str, Any]:
    return {
        "project": project,
        "category": category,
        "name": name,
        "command": [],
        "status": "skipped",
        "exitCode": None,
        "durationSeconds": 0,
        "output": reason,
        "outputTruncated": False,
    }


if __name__ == "__main__":
    raise SystemExit(main())
