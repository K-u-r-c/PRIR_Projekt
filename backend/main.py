"""FastAPI service that exposes the PRIR test scenarios and can run them on demand."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = Path(__file__).resolve().parent / "test_data" / "test_cases.json"
COMMAND_TIMEOUT = int(os.environ.get("PRIR_TEST_TIMEOUT_SECONDS", "600"))


def load_test_data() -> Dict[str, Any]:
  try:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))
  except FileNotFoundError as exc:
    raise RuntimeError(f"Test description file not found at {DATA_PATH}") from exc
  except json.JSONDecodeError as exc:
    raise RuntimeError(f"Invalid JSON in {DATA_PATH}: {exc}") from exc


test_data = load_test_data()


class RunResult(BaseModel):
  testId: str
  scenarioId: str
  command: str
  exitCode: int
  durationMs: float
  stdout: str
  stderr: str
  startedAt: str
  finishedAt: str
  success: bool
  errorMessage: Optional[str] = None


class RunOverrides(BaseModel):
  threads: Optional[int] = None
  useCuda: Optional[bool] = None
  extraArgs: Optional[str] = None


def find_test(test_id: str) -> Optional[Dict[str, Any]]:
  return next((item for item in test_data.get("tests", []) if item.get("id") == test_id), None)


def find_scenario(test: Dict[str, Any], scenario_id: str) -> Optional[Dict[str, Any]]:
  return next((item for item in test.get("scenarios", []) if item.get("id") == scenario_id), None)


def run_command(command: str) -> subprocess.CompletedProcess[str]:
  """Run a command through bash for compatibility with quoted strings."""
  cmd = ["bash", "-lc", command]
  return subprocess.run(
      cmd,
      cwd=str(BASE_DIR),
      capture_output=True,
      text=True,
      timeout=COMMAND_TIMEOUT,
      check=False,
  )


app = FastAPI(title="PRIR test runner", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/tests")
def list_tests() -> Dict[str, Any]:
  return test_data


@app.post("/api/tests/reload")
def reload_tests() -> Dict[str, str]:
  global test_data
  test_data = load_test_data()
  return {"status": "ok"}


def _sanitize_threads(args: List[str]) -> List[str]:
  cleaned: List[str] = []
  skip_next = False
  for token in args:
    if skip_next:
      skip_next = False
      continue
    if token == "--threads":
      skip_next = True
      continue
    if token.startswith("--threads="):
      continue
    cleaned.append(token)
  return cleaned


def _strip_flag(args: List[str], flag: str) -> List[str]:
  return [token for token in args if token != flag]


def _apply_overrides(base_command: str, overrides: RunOverrides) -> str:
  tokens = shlex.split(base_command)
  if overrides.threads is not None:
    tokens = _sanitize_threads(tokens)
    tokens.extend(["--threads", str(overrides.threads)])
  if overrides.useCuda is not None:
    tokens = _strip_flag(tokens, "--use-cuda")
    tokens = _strip_flag(tokens, "--cpu-only")
    if overrides.useCuda:
      tokens.append("--use-cuda")
    else:
      tokens.append("--cpu-only")
  if overrides.extraArgs:
    extra = shlex.split(overrides.extraArgs)
    tokens.extend(extra)
  return shlex.join(tokens)


@app.post("/api/tests/{test_id}/scenarios/{scenario_id}/run", response_model=RunResult)
def execute_scenario(test_id: str, scenario_id: str, overrides: RunOverrides | None = None) -> RunResult:
  test = find_test(test_id)
  if not test:
    raise HTTPException(status_code=404, detail=f"Unknown test id '{test_id}'")
  scenario = find_scenario(test, scenario_id)
  if not scenario:
    raise HTTPException(status_code=404, detail=f"Unknown scenario id '{scenario_id}'")

  command = scenario.get("command")
  if not command:
    raise HTTPException(status_code=400, detail="Scenario is missing a command")
  overrides = overrides or RunOverrides()
  if overrides.threads is not None and overrides.threads <= 0:
    raise HTTPException(status_code=400, detail="threads override must be positive")
  final_command = _apply_overrides(command, overrides)

  started_at = datetime.now(tz=timezone.utc)
  started_ts = time.perf_counter()
  error_message = None

  try:
    completed = run_command(final_command)
  except subprocess.TimeoutExpired as exc:
    finished_at = datetime.now(tz=timezone.utc)
    duration_ms = (time.perf_counter() - started_ts) * 1000
    return RunResult(
        testId=test_id,
        scenarioId=scenario_id,
        command=final_command,
        exitCode=-1,
        durationMs=duration_ms,
        stdout=exc.stdout or "",
        stderr=exc.stderr or "",
        startedAt=started_at.isoformat(),
        finishedAt=finished_at.isoformat(),
        success=False,
        errorMessage=f"Command exceeded timeout of {COMMAND_TIMEOUT} seconds.",
    )
  except OSError as exc:
    finished_at = datetime.now(tz=timezone.utc)
    duration_ms = (time.perf_counter() - started_ts) * 1000
    return RunResult(
        testId=test_id,
        scenarioId=scenario_id,
        command=final_command,
        exitCode=-1,
        durationMs=duration_ms,
        stdout="",
        stderr=str(exc),
        startedAt=started_at.isoformat(),
        finishedAt=finished_at.isoformat(),
        success=False,
        errorMessage="Failed to spawn the command.",
    )

  finished_at = datetime.now(tz=timezone.utc)
  duration_ms = (time.perf_counter() - started_ts) * 1000
  success = completed.returncode == 0
  if not success:
    error_message = "Command finished with a non-zero exit code."

  return RunResult(
      testId=test_id,
      scenarioId=scenario_id,
      command=final_command,
      exitCode=completed.returncode,
      durationMs=duration_ms,
      stdout=completed.stdout or "",
      stderr=completed.stderr or "",
      startedAt=started_at.isoformat(),
      finishedAt=finished_at.isoformat(),
      success=success,
      errorMessage=error_message,
  )
