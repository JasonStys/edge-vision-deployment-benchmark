# Retained native dependency notices

`flatbuffers-25.12.19.LICENSE` is the unmodified Apache 2.0 license from the exact
[upstream tag](https://github.com/google/flatbuffers/blob/v25.12.19/LICENSE), retrieved because that
version's installed Python wheel declares Apache 2.0 but omits the license file. Its SHA-256 is
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
The native builder requires this exact vendor version before applying that narrow notice fallback.

Other Python/scientific notices are read from their installed distribution files. Actual Tcl/Tk
terms are read from the loaded SDK's resources, including Tcl 9 DLL zipfs. Unmodified Microsoft
app-local runtime libraries retain their separate vendor licensing and source references. Native
packages include the collected notices beside the application; no vendor executable or DLL is
committed to this source repository.
