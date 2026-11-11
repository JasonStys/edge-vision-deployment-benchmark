"""File: tests/test_runtime_privacy.py
Purpose: Verify real fresh-process import order and the official event-disable API invocation.
Symbols: the authored child observes actual imports; it never opens a user profile or dataset.
This proves opt-out configuration, not an OS-wide network-traffic certification.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("inherited", [None, "0"])
def test_actual_vendor_initialization_is_preceded_by_opt_out_and_api_disable(inherited):
    """Clear inherited opt-out and observe actual locked vendor imports in a fresh interpreter."""
    source = Path(__file__).resolve().parents[1] / "src"
    program = r"""
import builtins,json,os,sys
sys.path.insert(0,sys.argv[1])
original_import=builtins.__import__
events=[]
calls=[]
wrapped=False
def observed_import(name,*args,**kwargs):
    global wrapped
    if name.startswith('onnxruntime'):
        events.append(os.environ.get('ORT_DISABLE_TELEMETRY') == '1')
    value=original_import(name,*args,**kwargs)
    if name == 'onnxruntime' and not wrapped:
        wrapped=True
        original_disable=value.disable_telemetry_events
        def disabled():
            calls.append(True)
            return original_disable()
        value.disable_telemetry_events=disabled
    return value
builtins.__import__=observed_import
import edgevision.pipeline
print(json.dumps({'imports':len(events),'early':bool(events) and all(events),
                  'apiCalls':len(calls)}))
"""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() not in {"ORT_DISABLE_TELEMETRY", "PYTHONPATH"}
    }
    if inherited is not None:
        environment["ORT_DISABLE_TELEMETRY"] = inherited
    # Only the fixed interpreter and literal authored test program execute; no supplied code is run.
    result = subprocess.run(  # noqa: S603 - fixed interpreter and authored program
        [sys.executable, "-I", "-c", program, str(source)],
        env=environment,
        check=True,
        capture_output=True,
        timeout=30,
    )
    assert len(result.stdout) <= 4096
    observation = json.loads(result.stdout)
    assert observation["imports"] > 0 and observation["early"] is True
    assert observation["apiCalls"] >= 1
