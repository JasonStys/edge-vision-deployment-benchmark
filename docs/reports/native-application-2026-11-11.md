# Native workbench validation â€” 2026-11-11

This report describes observed Windows development evidence, not a claim that a public release or
clean-main hosted run already passed. The earlier reviewed Word/workbook editions retain their
historical source baseline. The new native application is version 1.1.0; portable model format 1 and
the original C++/Rust inference algorithms are unchanged.

## Results

- Full original and new Python suite: **76 passed, zero skips**, CPython 3.14.10, 8.03 s in the observed
  run. Coverage was **97.59% combined statements/branches for `edgevision`**, above the unchanged 94%
  floor. This denominator excludes new desktop widgets and native C++/Rust coverage. Actual widget
  tests and compiled-application journeys are reported separately rather than called 100% coverage.
- Ruff format/lint of new native source/tests and strict original mypy passed. Strict mypy remains
  scoped to the original eleven scientific source files; no claim of strict GUI typing is made.
- Original C++ CTest: **1/1 passed**. Original Rust tests: **3/3 passed**. Actual custom-vector
  NumPy/ONNX float/int8/C++/Rust probabilities agreed within the unchanged `1e-5` bound; the largest
  observed six-row reference difference was approximately `1.073e-6` for dynamic int8.
- **14 actual ZIP guard tests** passed: bytes/relocation/no-overwrite, unsafe Windows names and path
  traversal, case collisions, declared links/devices, expansion budgets, bounded metadata and exact
  test marker. These are distinct from successful app journeys and do not represent fuzzing.
- Frozen Windows candidate `pilot-005` completed full ZIP relocation and **two actual native
  launches**, both exit 0. It restored exact captured generation/report/vector identities and reran
  all five real engines. The package included patched Python 3.14.10 and actual Tcl/Tk 9.0.4.
- Four actually loaded Microsoft runtime DLLs were verified inside the extracted application, with
  hashes matching the signature-checked app-local SDK inputs. The fifth retained runtime input was
  not claimed to be loaded. OS libraries remain Windows dependencies.

## What the actual user journeys verify

Editable held-out CSV is saved/imported through real controls. A malformed CSV cannot replace valid
editor data. Five actual engines produce all thirty sample/runtime rows. Review renders all twelve
model/condition rows; strict gates expose real failures, and NaN is rejected before work starts.
Export matches complete captured JSON and refuses an existing destination. Cancelled rebuild leaves
the active generation intact; accepted rebuild runs original training/export/checks and publishes a
new verified generation while retaining the old one. The second process restores exact identities
and imports the saved CSV to execute the real engines again.

Repeated Quit is requested while a genuine local file-write operation is deliberately held. Actual
window/profile ownership remains until that operation is released. The parent requires normal
process exit; this event proof is not a promise that every possible computation finishes within a
universal shutdown deadline. Source-only widget tests also verify a failing result renderer retains
captured evidence and leaves the close pump usable.

## Genuine issues found and fixed

The first frozen windowed probe reached its unchanged 120-second parent bound without an acceptance
receipt. A separate console diagnostic exposed `ModuleNotFoundError: fusion_utils` in the exact
ONNX Runtime 1.31.0 quantization import chain. That vendor version uses bare transformer-helper
imports. The builder now supplies the locked installed vendor transformer path during freezing;
no third-party source, scientific algorithm or user import path is modified. The subsequent normal
windowed candidate passed both full extracted-app journeys at the same deadline.

An inherited PowerShell 7 module path initially prevented the fixed Windows PowerShell verifier
from loading its own security module. The builder omits that inherited variable only in the verifier
child environment, retaining Microsoft signature checks. The FlatBuffers wheel lacked a license
file; its exact tagged upstream Apache license is now included. Both failures and failed staging
remain private development evidence; they are not relabelled as passes.

An old shared default pytest directory was inaccessible, preventing fixture setup. The full suite
was rerun in a new owned directory and passed; the setup-error run is not counted as a test pass.

## Limits and remaining gates

This candidate is unsigned Windows x64 and uses synthetic 8Ã—8 shape data. It does not certify
real-camera, GPU/NPU/ARM/mobile behavior, physical energy or clean-machine installation. One trusted
OS owner holds plaintext bounded state; no user accounts, external telemetry, live camera, general
model import or automatic deployment are added. GUI/native timings are explicitly different scopes.
The local package was built while source was dirty and records that fact. Exact clean-source native
acceptance, all hosted original/security/native checks, review and delivery remain separate gates.

A later pre-acceptance build correctly refused source-index regeneration during freezing. That
source-guard failure is preserved separately, not counted as a compiled-application pass. Final
post-commit native evidence and hosted results are recorded in the review handoff.
