# File: build-kernels.ps1
# Purpose: Compile and test the original actual C++/Rust inference engines for the Windows workbench.
# State: fresh ignored output; static CRT; per-process build environment only, never installed services.
[CmdletBinding()]
param([Parameter(Mandatory)][string]$Output)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$repository = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$target = [IO.Path]::GetFullPath((Join-Path $repository $Output))
$boundary = [IO.Path]::GetFullPath((Join-Path $repository '.runtime')) + [IO.Path]::DirectorySeparatorChar
if (-not $IsWindows -or -not $target.StartsWith($boundary,[StringComparison]::OrdinalIgnoreCase) -or (Test-Path -LiteralPath $target)) {
    throw 'Choose a fresh Windows kernel staging folder below .runtime.'
}
# A build tool argument vector contains only fixed source and fresh owned output paths.
function Invoke-Checked([string]$Tool,[string[]]$Arguments) {
    & $Tool @Arguments
    if ($LASTEXITCODE -ne 0) { throw 'The actual native inference build/test command failed.' }
}
$cmake = @(Get-Command cmake -CommandType Application)[0].Source
$cargo = @(Get-Command cargo -CommandType Application)[0].Source
$ctest = Join-Path (Split-Path $cmake -Parent) 'ctest.exe'
$oldFlags,$oldToolchain = $env:RUSTFLAGS,$env:RUSTUP_TOOLCHAIN
Push-Location $repository
try {
    [void][IO.Directory]::CreateDirectory($target)
    $cpp = Join-Path $target 'cpp-build'
    $rust = Join-Path $target 'rust-build'
    Invoke-Checked $cmake @('-S','native/cpp','-B',$cpp,'-A','x64','-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded')
    Invoke-Checked $cmake @('--build',$cpp,'--config','Release','--parallel','2')
    Invoke-Checked $ctest @('--test-dir',$cpp,'-C','Release','--output-on-failure')
    $env:RUSTFLAGS = '-C target-feature=+crt-static'
    $env:RUSTUP_TOOLCHAIN = '1.98.1'
    Invoke-Checked $cargo @('test','--manifest-path','native/rust/Cargo.toml','--locked','--target-dir',$rust)
    Invoke-Checked $cargo @('build','--manifest-path','native/rust/Cargo.toml','--release','--locked','--target-dir',$rust)
    $kernels = Join-Path $target 'kernels'
    [void][IO.Directory]::CreateDirectory($kernels)
    Copy-Item -LiteralPath (Join-Path $cpp 'Release/edgevision_cpp.exe') -Destination $kernels
    Copy-Item -LiteralPath (Join-Path $rust 'release/edgevision-rust.exe') -Destination $kernels
    $inventory = [ordered]@{}
    foreach ($name in @('edgevision_cpp.exe','edgevision-rust.exe')) {
        $inventory[$name]=(Get-FileHash -LiteralPath (Join-Path $kernels $name) -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    $inventory | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $kernels 'KERNELS.json') -Encoding utf8NoBOM
    Write-Output 'Original actual inference kernels and tests passed; no global toolchain changes.'
}
finally {
    $env:RUSTFLAGS,$env:RUSTUP_TOOLCHAIN=$oldFlags,$oldToolchain
    Pop-Location
}
