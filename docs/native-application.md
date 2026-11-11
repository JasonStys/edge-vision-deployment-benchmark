# Native Edge Vision Workbench

By Jason Stys · application 1.1.0.

## Use the full application

Extract the complete unsigned Windows x64 package, retain every adjacent resource, and open
`EdgeVisionWorkbench.exe`. No Python, compiler, IDE, administrator service or Docker setup is needed
for the packaged application. Windows and its supported OS libraries remain prerequisites; unsigned
distribution is not a signing, clean-machine or SmartScreen certification. Source/contact:
[Jason Stys on GitHub](https://github.com/JasonStys).

1. Choose **Prepare held-out sample data**, import a vector CSV, or toggle cells in the 8×8 drawing.
   Select the asserted class and choose **Use drawn vector**. **Clear drawing** affects only those
   cells. Prepared examples are data inside this complete application.
2. CSV has the exact `id,label,f0,…,f63` header, 1–256 rows, unique safe IDs, one of `vertical`,
   `horizontal`, `diagonal`, and 64 finite pixel values in [0,1]. Maximum input is 1 MiB. Labels are
   user assertions, not model-verified ground truth. Scroll the CSV horizontally to inspect every
   column. Save edited vectors explicitly with **Save vector CSV**; editor contents are not autosaved.
3. Run the three scientific variants or **Compare all five actual engines**. The prediction table
   scrolls vertically. Each result retains class probabilities, actual identity/order checks, batch
   timings and maximum reference error. C++/Rust invoke only the fixed packaged executables, with
   verified hashes and fresh generated input/output paths. Native timings include startup and I/O;
   Python timings cover prediction calls, so these scopes are not interchangeable benchmarks.
4. Under **Deployment review gates**, set finite 0–1 thresholds for minimum accuracy/recall and
   maximum calibration/reference error. The original review runs three variants under clean,
   Gaussian, occlusion and dimming conditions, each on the same 48 held-out samples. All twelve rows
   and failed gates remain visible. Changing advisory gates does not change original release checks.
5. **Export captured evidence** saves exactly the captured full JSON to a new file. The screen preview
   can truncate at 60,000 characters; export does not. Existing files are refused, not overwritten.
6. **Inspect complete candidate** verifies the fixed artifact fingerprints; it does not rerun quality.
   **Rebuild reproducible reference** requires confirmation and runs the original deterministic
   training, calibration, ONNX export, quantization and release checks. It publishes a new generation
   only after successful capture; earlier generations remain.
7. Use **Quit application** or the window close control. Active real work finishes before the window
   and owner handle close. A failing test deadline is not a universal shutdown-time promise.

## Data, recovery and privacy

Default Windows data is below `%LOCALAPPDATA%/JasonStysPortfolio/edge-vision-deployment-benchmark/workspace`.
Explicit `--profile` selects an empty or already identified directory. Frozen executable/resource
trees, directory aliases and pre-existing hardlinked marker/owner files are refused. Existing model
and state corruption is preserved and reported rather than replaced. Another process running as the
same OS owner can still alter its namespace; this is not race-proof or cryptographically authenticated
storage. There are no application accounts or authenticated reviewer roles.

The profile is plaintext: saved model generations, up to 100 captured reports, up to 100 native
analyses, and up to 100 candidates. Capacity stops further work with a visible error. Close the app,
back up the entire profile together, and choose a new empty profile for a fresh workspace; exports
are result snapshots, not a full backup. No automatic retention deletion is added. User-selected
exports stay where the user chooses them. No telemetry, camera, external model import or automatic
deployment is added.

The process sets the vendor's `ORT_DISABLE_TELEMETRY=1` before ONNX Runtime initializes and invokes
its event-disable API. A frozen early hook applies the same policy before application imports.
This follows the [vendor privacy documentation](https://github.com/microsoft/onnxruntime/blob/main/docs/Privacy.md).
The test proves configured import order, not an OS-wide network audit. An external embedding caller
that already initialized the SDK remains responsible for that earlier initialization.

## Build and verify

Build-time tools are separate from end-user requirements. Use an isolated CPython 3.14.10 environment,
the exact `requirements-native.lock`, PyInstaller 6.22.3, a licensed Visual Studio x64 toolchain with
signed app-local CRT source, CMake and Rust 1.98.1. The original Linux/native verification remains.

```powershell
python -m pip install -r requirements-native.lock
python -m pip install --no-build-isolation --no-deps .
pwsh -NoProfile -File scripts/build-kernels.ps1 -Output .runtime/build-001
python scripts/build-native.py --output release/build-001 --kernels .runtime/build-001/kernels --crt-sdk "ABSOLUTE_LICENSED_VS_CRT_DIRECTORY"
python scripts/archive_native.py release/build-001/EdgeVisionWorkbench artifacts/native/build-001
```

The CRT directory is the unmodified licensed Visual Studio `VC/Redist/MSVC/VERSION/x64/Microsoft.VCxxx.CRT`
directory. `scripts/check-crt.ps1` requires Microsoft Authenticode/company identity for the five fixed
DLLs. Actual used DLL origins and hashes are checked in the frozen process. Build output retains
actual upstream notices; the FlatBuffers wheel omits its license, so the exact upstream
[25.12.19 license](https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE) is retained separately.
Microsoft runtime terms and the distributable list remain separate from this repository's MIT source.

Use fresh output paths; no earlier archive, evidence, profile or installed app is removed. Hosted CI
requires a clean checkout equal to `GITHUB_SHA`. Local development candidates record `sourceDirty`.
The normal builder is windowed. `--console-diagnostics` is an explicit local debugging aid and is
refused in CI; it cannot satisfy the normal hosted application gate.

## Actual acceptance contract

The relocated executable starts twice with a system-only executable PATH and an exact marked
synthetic acceptance root selected before default OS-profile lookup. It uses actual widget bindings,
real science/native engines, invalid import/gate cases, model rebuild, complete captured export and
restart state. A real file-write operation is held by an event while repeated Quit is requested:
the window and exclusive profile lock remain until release; the parent then requires exit 0. Both
launches must restore the same retained generation/report/vector identities. Screenshots capture
only the app's own window through its owned handle, never the desktop or generic user input.
