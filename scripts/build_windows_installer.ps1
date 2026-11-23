<#
.SYNOPSIS
    Stage and build the Second Brain Windows Desktop installer (`SecondBrain-Setup-x64.exe`).

.DESCRIPTION
    Executes the deterministic Option 1 Slim Installer staging pipeline:
      1. Builds the React 19 SPA (`frontend/dist`).
      2. Downloads (cached in `build/downloads/`) and verifies SHA-256 checksums
         for pinned native `qdrant.exe` and Astral `uv.exe`.
      3. Provisions Astral managed CPython 3.13 (`--managed-python`) into
         `build/staging/python` and installs locked CPU + `desktop` wheels
         into `build/staging/venv` via `uv sync --frozen --no-dev --extra desktop`.
      4. Strips bytecode/test bloat, enforces zero-model-weights staging, and
         copies `backend/src`, configs, `frontend/dist`, `launcher_entry.py`,
         and `THIRD_PARTY_NOTICES.txt` into `build/staging/app`.
      5. Runs staged Python/desktop/API import smoke checks and compiles
         `installer/second-brain.iss` via Inno Setup 6 (`ISCC.exe`) unless
         `-StageOnly` is specified.
#>
[CmdletBinding()]
param(
    [switch]$StageOnly,
    [switch]$SkipFrontendBuild,
    [switch]$UseExistingStaging,
    [string]$QdrantVersion = "v1.20.2",
    [string]$QdrantSha256 = "7a5290b9ef1b204b250a1626ddafec580647ba3cc62ba1630b3a8efdc2718b32",
    [string]$UvVersion = "0.13.9",
    [string]$UvSha256 = "edd61f7f5728a56cc62455386661860bdfc7bab65872f39b0c25bdcc1d8e1beb",
    [string]$PythonVersion = "3.13"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Assert-LastExitCode {
    param([Parameter(Mandatory = $true)][string]$StepName)
    if ($LASTEXITCODE -ne 0) {
        throw "$StepName failed with exit code $LASTEXITCODE."
    }
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$DestPath,
        [Parameter(Mandatory = $true)][string]$ExpectedSha256
    )

    $expected = $ExpectedSha256.Trim().ToLowerInvariant()
    if (Test-Path -LiteralPath $DestPath) {
        $cachedHash = (Get-FileHash -LiteralPath $DestPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($cachedHash -eq $expected) {
            Write-Host "  [cache hit] $(Split-Path -Leaf $DestPath) (SHA-256 verified)"
            return
        }
        Write-Warning "Cached file '$DestPath' failed SHA-256 verification ($cachedHash != $expected); re-downloading."
        Remove-Item -LiteralPath $DestPath -Force
    }

    $tmpPath = "$DestPath.tmp"
    if (Test-Path -LiteralPath $tmpPath) {
        Remove-Item -LiteralPath $tmpPath -Force
    }

    Write-Host "  [download] $Url"
    & curl.exe -fL --retry 3 --retry-delay 2 -o $tmpPath $Url
    Assert-LastExitCode "Download $Url"

    $actualHash = (Get-FileHash -LiteralPath $tmpPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $expected) {
        Remove-Item -LiteralPath $tmpPath -Force -ErrorAction SilentlyContinue
        throw "SHA-256 mismatch for '$Url'. Expected: $expected, Actual: $actualHash"
    }

    Move-Item -LiteralPath $tmpPath -Destination $DestPath -Force
    Write-Host "  [verified] $(Split-Path -Leaf $DestPath) -> $actualHash"
}

function Expand-SingleExecutable {
    param(
        [Parameter(Mandatory = $true)][string]$ZipPath,
        [Parameter(Mandatory = $true)][string]$ExeName,
        [Parameter(Mandatory = $true)][string]$DestExePath,
        [Parameter(Mandatory = $true)][string]$TempRoot
    )

    $extractDir = Join-Path $TempRoot ([System.IO.Path]::GetFileNameWithoutExtension($ZipPath))
    if (Test-Path -LiteralPath $extractDir) {
        Remove-Item -LiteralPath $extractDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $extractDir -Force | Out-Null

    Expand-Archive -LiteralPath $ZipPath -DestinationPath $extractDir -Force
    $found = Get-ChildItem -LiteralPath $extractDir -Recurse -Filter $ExeName -File | Select-Object -First 1
    if ($null -eq $found) {
        throw "Could not find '$ExeName' inside archive '$ZipPath'."
    }

    Copy-Item -LiteralPath $found.FullName -Destination $DestExePath -Force
    Remove-Item -LiteralPath $extractDir -Recurse -Force -ErrorAction SilentlyContinue
}

function Remove-BytecodeAndTests {
    param(
        [Parameter(Mandatory = $true)][string]$RootPath,
        [switch]$StripTests
    )

    if (-not (Test-Path -LiteralPath $RootPath)) {
        return
    }

    Get-ChildItem -LiteralPath $RootPath -Recurse -Force -File -ErrorAction SilentlyContinue |
        Where-Object {
            $ext = $_.Extension.ToLowerInvariant()
            $ext -eq ".pyc" -or $ext -eq ".pyo"
        } |
        Remove-Item -Force -ErrorAction SilentlyContinue

    Get-ChildItem -LiteralPath $RootPath -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -ceq "__pycache__" } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

    if ($StripTests) {
        $sitePackages = Join-Path $RootPath "Lib\site-packages"
        if (Test-Path -LiteralPath $sitePackages) {
            # Remove only explicit standalone test suites so runtime `test`/`testing` subpackages stay intact.
            $explicitTestPatterns = @(
                (Join-Path $sitePackages "torch\test"),
                (Join-Path $sitePackages "pandas\tests"),
                (Join-Path $sitePackages "numpy\testing\tests"),
                (Join-Path $sitePackages "scipy\*\tests")
            )
            foreach ($pattern in $explicitTestPatterns) {
                Get-Item -Path $pattern -ErrorAction SilentlyContinue |
                    Where-Object { $_.PSIsContainer } |
                    ForEach-Object {
                        Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
                    }
            }
        }
    }
}

function Assert-ZeroModelWeights {
    param(
        [Parameter(Mandatory = $true)][string]$StagingRoot,
        [Parameter(Mandatory = $true)][string]$VenvRoot,
        [Parameter(Mandatory = $true)][string]$AppRoot
    )

    $allStagedFiles = Get-ChildItem -LiteralPath $StagingRoot -Recurse -Force -File -ErrorAction SilentlyContinue
    $sitePackagesPrefix = (Join-Path $VenvRoot "Lib\site-packages").ToLowerInvariant() + "\"
    $allowedWheelResourcePrefixes = @(
        (Join-Path $VenvRoot "Lib\site-packages\torch\").ToLowerInvariant(),
        (Join-Path $VenvRoot "Lib\site-packages\onnxruntime\datasets\").ToLowerInvariant(),
        (Join-Path $VenvRoot "Lib\site-packages\pymupdf\").ToLowerInvariant(),
        (Join-Path $VenvRoot "Lib\site-packages\pymupdf4llm\").ToLowerInvariant(),
        (Join-Path $VenvRoot "Lib\site-packages\rapidocr\models\").ToLowerInvariant()
    )

    $forbiddenAlways = $allStagedFiles | Where-Object {
        $_.Extension.ToLowerInvariant() -in @(".safetensors", ".gguf", ".ckpt", ".h5")
    }
    if ($forbiddenAlways) {
        $sample = ($forbiddenAlways | Select-Object -First 5 | ForEach-Object { $_.FullName }) -join "; "
        throw "Forbidden AI model weights detected in staging payload: $sample"
    }

    $unexpectedWeightFiles = $allStagedFiles | Where-Object {
        $ext = $_.Extension.ToLowerInvariant()
        if ($ext -notin @(".onnx", ".bin", ".pt", ".pth")) {
            return $false
        }
        $full = $_.FullName.ToLowerInvariant()
        $isSitePthConfig = ($ext -eq ".pth") -and $full.StartsWith($sitePackagesPrefix) -and ($_.Length -lt 65536)
        if ($isSitePthConfig) {
            return $false
        }
        foreach ($allowedPrefix in $allowedWheelResourcePrefixes) {
            if ($full.StartsWith($allowedPrefix)) {
                return $false
            }
        }
        return $true
    }
    if ($unexpectedWeightFiles) {
        $sample = ($unexpectedWeightFiles | Select-Object -First 5 | ForEach-Object { $_.FullName }) -join "; "
        throw "Unexpected AI model weight or binary files staged: $sample"
    }

    $forbiddenModelCacheDirs = Get-ChildItem -LiteralPath $StagingRoot -Recurse -Force -Directory -ErrorAction SilentlyContinue |
        Where-Object {
            $name = $_.Name.ToLowerInvariant()
            $full = $_.FullName.ToLowerInvariant()
            $name.StartsWith("models--") -or
            ($name -in @("huggingface", "fastembed", "flashrank", "docling-models") -and -not $full.StartsWith($sitePackagesPrefix))
        }
    if ($forbiddenModelCacheDirs) {
        $sample = ($forbiddenModelCacheDirs | Select-Object -First 5 | ForEach-Object { $_.FullName }) -join "; "
        throw "Forbidden downloaded model cache directory staged: $sample"
    }

    $forbiddenDirs = @(
        (Join-Path $AppRoot "data\models"),
        (Join-Path $AppRoot "backend\data\models"),
        (Join-Path $StagingRoot "models")
    )
    foreach ($dir in $forbiddenDirs) {
        if (Test-Path -LiteralPath $dir) {
            throw "Forbidden staged model cache directory exists: $dir"
        }
    }
}

function Get-DirectorySizeMB {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0.0
    }
    $sum = (Get-ChildItem -LiteralPath $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $sum) {
        $sum = 0
    }
    return [Math]::Round($sum / 1MB, 2)
}

function Find-IsccExecutable {
    $cmd = Get-Command "ISCC.exe" -ErrorAction SilentlyContinue
    if ($null -ne $cmd) {
        return $cmd.Source
    }

    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "${env:ProgramFiles}\Inno Setup 6\ISCC.exe",
        "${env:LOCALAPPDATA}\Programs\Inno Setup 6\ISCC.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    return $null
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$frontendDir = Join-Path $repoRoot "frontend"
$backendDir = Join-Path $repoRoot "backend"
$installerDir = Join-Path $repoRoot "installer"
$buildRoot = Join-Path $repoRoot "build"
$downloadsDir = Join-Path $buildRoot "downloads"
$tempExtractDir = Join-Path $buildRoot "tmp_extract"
$stagingDir = Join-Path $buildRoot "staging"
$stagingApp = Join-Path $stagingDir "app"
$stagingBin = Join-Path $stagingDir "bin"
$stagingPython = Join-Path $stagingDir "python"
$stagingVenv = Join-Path $stagingDir "venv"
$distDir = Join-Path $repoRoot "dist"

Write-Host "=== Second Brain Windows Desktop Staging & Build Pipeline ==="
Write-Host "Repo root : $repoRoot"
Write-Host "StageOnly : $($StageOnly.IsPresent)"

New-Item -ItemType Directory -Path $downloadsDir -Force | Out-Null
New-Item -ItemType Directory -Path $distDir -Force | Out-Null

$frontendDistDir = Join-Path $frontendDir "dist"
$frontendIndex = Join-Path $frontendDistDir "index.html"
$stagedQdrantExe = Join-Path $stagingBin "qdrant.exe"
$stagedUvExe = Join-Path $stagingBin "uv.exe"
$stagedPythonExe = Join-Path $stagingPython "python.exe"
$stagedVenvPython = Join-Path $stagingVenv "Scripts\python.exe"
$stagedVenvPythonw = Join-Path $stagingVenv "Scripts\pythonw.exe"
$stagedPyvenvCfg = Join-Path $stagingVenv "pyvenv.cfg"

$reuseBaseStaging = $false
if ($UseExistingStaging) {
    $hasReusableStaging = (Test-Path -LiteralPath $frontendIndex) -and
        (Test-Path -LiteralPath $stagedQdrantExe) -and
        (Test-Path -LiteralPath $stagedUvExe) -and
        (Test-Path -LiteralPath $stagedPythonExe) -and
        (Test-Path -LiteralPath $stagedVenvPythonw) -and
        (Test-Path -LiteralPath $stagedPyvenvCfg)
    if ($hasReusableStaging) {
        $reuseBaseStaging = $true
    } else {
        Write-Host "  -UseExistingStaging requested, but base staging payload is incomplete; performing full stage..."
    }
}

if ($reuseBaseStaging) {
    Write-Host "`n[1-3/5] Reusing existing staged frontend, native binaries, Python, and venv (-UseExistingStaging)..."
    if (Test-Path -LiteralPath $stagingApp) {
        Remove-Item -LiteralPath $stagingApp -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagingApp -Force | Out-Null
} else {
    if (Test-Path -LiteralPath $stagingDir) {
        Write-Host "Cleaning previous staging directory: $stagingDir"
        Remove-Item -LiteralPath $stagingDir -Recurse -Force
    }
    if (Test-Path -LiteralPath $tempExtractDir) {
        Remove-Item -LiteralPath $tempExtractDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $stagingApp, $stagingBin, $stagingPython, $tempExtractDir -Force | Out-Null

    # --------------------------------------------------------------------------
    # Step 1: Build React SPA
    # --------------------------------------------------------------------------
    Write-Host "`n[1/5] Building React SPA (frontend/dist)..."
    if ($SkipFrontendBuild -and (Test-Path -LiteralPath $frontendIndex)) {
        Write-Host "  Skipping npm build (-SkipFrontendBuild); existing frontend/dist/index.html found."
    } else {
        Push-Location $frontendDir
        try {
            & npm.cmd run build
            Assert-LastExitCode "Frontend build (npm.cmd run build)"
        } finally {
            Pop-Location
        }
    }
    if (-not (Test-Path -LiteralPath $frontendIndex)) {
        throw "Expected compiled SPA entry at '$frontendIndex' was not produced."
    }

    # --------------------------------------------------------------------------
    # Step 2: Fetch & Verify Pinned Native Binaries (Qdrant + Astral uv)
    # --------------------------------------------------------------------------
    Write-Host "`n[2/5] Downloading & verifying SHA-256 for Qdrant ($QdrantVersion) and uv ($UvVersion)..."
    $qdrantZipName = "qdrant-x86_64-pc-windows-msvc.zip"
    $qdrantUrl = "https://github.com/qdrant/qdrant/releases/download/$QdrantVersion/$qdrantZipName"
    $qdrantZipPath = Join-Path $downloadsDir "$QdrantVersion-$qdrantZipName"

    $uvZipName = "uv-x86_64-pc-windows-msvc.zip"
    $uvUrl = "https://github.com/astral-sh/uv/releases/download/$UvVersion/$uvZipName"
    $uvZipPath = Join-Path $downloadsDir "$UvVersion-$uvZipName"

    Get-VerifiedDownload -Url $qdrantUrl -DestPath $qdrantZipPath -ExpectedSha256 $QdrantSha256
    Get-VerifiedDownload -Url $uvUrl -DestPath $uvZipPath -ExpectedSha256 $UvSha256

    Expand-SingleExecutable -ZipPath $qdrantZipPath -ExeName "qdrant.exe" -DestExePath $stagedQdrantExe -TempRoot $tempExtractDir
    Expand-SingleExecutable -ZipPath $uvZipPath -ExeName "uv.exe" -DestExePath $stagedUvExe -TempRoot $tempExtractDir
    Remove-Item -LiteralPath $tempExtractDir -Recurse -Force -ErrorAction SilentlyContinue

    if (-not (Test-Path -LiteralPath $stagedQdrantExe)) { throw "Missing staged binary: $stagedQdrantExe" }
    if (-not (Test-Path -LiteralPath $stagedUvExe)) { throw "Missing staged binary: $stagedUvExe" }
    Write-Host "  Staged qdrant.exe & uv.exe successfully."

    # --------------------------------------------------------------------------
    # Step 3: Provision Managed CPython 3.13 & Build CPU + Desktop Virtualenv
    # --------------------------------------------------------------------------
    Write-Host "`n[3/5] Provisioning standalone CPython $PythonVersion & staging CPU + desktop venv..."
    & $stagedUvExe python install --managed-python $PythonVersion
    Assert-LastExitCode "uv python install --managed-python $PythonVersion"

    $managedPyOut = & $stagedUvExe python find --managed-python $PythonVersion
    Assert-LastExitCode "uv python find --managed-python $PythonVersion"
    $managedPyExe = (@($managedPyOut)[0]).Trim()
    if (-not (Test-Path -LiteralPath $managedPyExe)) {
        throw "Managed Python executable not found at '$managedPyExe'."
    }

    $managedPyHomeOut = & $managedPyExe -c "import os, sys; print(os.path.realpath(sys.base_prefix))"
    Assert-LastExitCode "Resolve managed Python realpath(sys.base_prefix)"
    $managedPyHome = (@($managedPyHomeOut)[0]).Trim()
    if (-not (Test-Path -LiteralPath $managedPyHome)) {
        throw "Resolved managed Python home directory does not exist: '$managedPyHome'."
    }
    Write-Host "  Copying managed CPython from $managedPyHome -> $stagingPython"
    Copy-Item -Path "$managedPyHome\*" -Destination $stagingPython -Recurse -Force

    # Strip stdlib test suite & bytecode inside staged Python to save installer size
    $pyLibTest = Join-Path $stagingPython "Lib\test"
    if (Test-Path -LiteralPath $pyLibTest) {
        Remove-Item -LiteralPath $pyLibTest -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-BytecodeAndTests -RootPath $stagingPython

    $stagedPythonwExe = Join-Path $stagingPython "pythonw.exe"
    if (-not (Test-Path -LiteralPath $stagedPythonExe)) {
        throw "Staged Python executable missing at '$stagedPythonExe'."
    }
    if (-not (Test-Path -LiteralPath $stagedPythonwExe)) {
        throw "Staged pythonw.exe missing at '$stagedPythonwExe'."
    }
    & $stagedPythonExe --version
    Assert-LastExitCode "Staged Python --version"

    $prevUvProjectEnv = $env:UV_PROJECT_ENVIRONMENT
    $prevUvLinkMode = $env:UV_LINK_MODE
    $prevUvNativeTls = $env:UV_NATIVE_TLS
    try {
        $env:UV_PROJECT_ENVIRONMENT = $stagingVenv
        $env:UV_LINK_MODE = "copy"
        $env:UV_NATIVE_TLS = "true"

        Push-Location $backendDir
        try {
            & $stagedUvExe venv --relocatable $stagingVenv --python $stagedPythonExe --clear
            Assert-LastExitCode "uv venv $stagingVenv"

            & $stagedUvExe sync --frozen --no-dev --extra desktop --no-install-project --python $stagedPythonExe
            Assert-LastExitCode "uv sync --frozen --no-dev --extra desktop"
        } finally {
            Pop-Location
        }
    } finally {
        $env:UV_PROJECT_ENVIRONMENT = $prevUvProjectEnv
        $env:UV_LINK_MODE = $prevUvLinkMode
        $env:UV_NATIVE_TLS = $prevUvNativeTls
    }

    if (-not (Test-Path -LiteralPath $stagedVenvPython)) { throw "Missing staged venv python.exe at '$stagedVenvPython'." }
    if (-not (Test-Path -LiteralPath $stagedVenvPythonw)) { throw "Missing staged venv pythonw.exe at '$stagedVenvPythonw'." }
    if (-not (Test-Path -LiteralPath $stagedPyvenvCfg)) { throw "Missing staged pyvenv.cfg at '$stagedPyvenvCfg'." }

    Write-Host "  Stripping __pycache__, *.pyc, *.pyo, and test directories from staged venv..."
    Remove-BytecodeAndTests -RootPath $stagingVenv -StripTests

    Write-Host "  Verifying staged venv package imports..."
    $smokeImportOutput = & $stagedVenvPython -c "import fastapi, langgraph, docling.document_converter, torch, webview, pystray, pymupdf, rapidocr, fastembed, laya, flashrank, scipy, numpy; print('SMOKE_OK')"
    Assert-LastExitCode "Staged venv smoke import check"
    if (($smokeImportOutput | Out-String) -notmatch "SMOKE_OK") {
        throw "Staged venv smoke import check did not emit SMOKE_OK."
    }
    Write-Host "  $($smokeImportOutput.Trim())"

    $pywProc = Start-Process -FilePath $stagedVenvPythonw -ArgumentList @("-c", '"import fastapi, webview, pystray"') -Wait -PassThru -WindowStyle Hidden
    if ($pywProc.ExitCode -ne 0) {
        throw "Staged pythonw.exe smoke execution failed with exit code $($pywProc.ExitCode)."
    }
    Write-Host "  pythonw.exe execution verified (exit code 0)."
}

# ------------------------------------------------------------------------------
# Step 4: Copy Application Payload into build/staging/app
# ------------------------------------------------------------------------------
Write-Host "`n[4/5] Copying application payload into $stagingApp..."
$stagedAppBackend = Join-Path $stagingApp "backend"
$stagedAppFrontendDist = Join-Path $stagingApp "frontend\dist"
$stagedAppInstaller = Join-Path $stagingApp "installer"

New-Item -ItemType Directory -Path $stagedAppBackend, $stagedAppFrontendDist, $stagedAppInstaller -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $backendDir "src") -Destination (Join-Path $stagedAppBackend "src") -Recurse -Force
Remove-BytecodeAndTests -RootPath (Join-Path $stagedAppBackend "src")

foreach ($cfgFile in @("pyproject.toml", "uv.lock", "config.yaml")) {
    $srcPath = Join-Path $backendDir $cfgFile
    if (-not (Test-Path -LiteralPath $srcPath)) {
        throw "Required backend file missing: $srcPath"
    }
    Copy-Item -LiteralPath $srcPath -Destination (Join-Path $stagedAppBackend $cfgFile) -Force
}

Copy-Item -Path "$frontendDistDir\*" -Destination $stagedAppFrontendDist -Recurse -Force
Copy-Item -LiteralPath (Join-Path $installerDir "launcher_entry.py") -Destination (Join-Path $stagedAppInstaller "launcher_entry.py") -Force
Copy-Item -LiteralPath (Join-Path $installerDir "THIRD_PARTY_NOTICES.txt") -Destination (Join-Path $stagedAppInstaller "THIRD_PARTY_NOTICES.txt") -Force
Copy-Item -LiteralPath (Join-Path $installerDir "second-brain.ico") -Destination (Join-Path $stagedAppInstaller "second-brain.ico") -Force
Copy-Item -LiteralPath (Join-Path $installerDir "second-brain.svg") -Destination (Join-Path $stagedAppInstaller "second-brain.svg") -Force

Assert-ZeroModelWeights -StagingRoot $stagingDir -VenvRoot $stagingVenv -AppRoot $stagingApp
Write-Host "  Zero staged AI model weights check passed."

Write-Host "  Verifying launcher_entry.py + staged DesktopPaths + FastAPI app import..."
$stagedLauncherEntry = Join-Path $stagedAppInstaller "launcher_entry.py"
$entryCheckCmd = @"
import importlib.util, os, sys
spec = importlib.util.spec_from_file_location('launcher_entry', r'$stagedLauncherEntry')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod._configure_runtime_paths()
from src.desktop.config import DesktopPaths
from src.desktop.app import DesktopApp
from src.api.main import app as fastapi_app
paths = DesktopPaths()
assert paths.is_staged_install, f'Expected staged install, got app_dir={paths.app_dir}'
assert paths.qdrant_exe.is_file(), f'Missing qdrant.exe at {paths.qdrant_exe}'
assert paths.uv_exe.is_file(), f'Missing uv.exe at {paths.uv_exe}'
assert paths.frontend_dist_dir.is_dir(), f'Missing frontend/dist at {paths.frontend_dist_dir}'
assert (paths.app_dir / 'app' / 'installer' / 'second-brain.ico').is_file(), 'Missing staged second-brain.ico'
assert fastapi_app is not None
print('LAUNCHER_ENTRY_OK')
"@
$entryOutput = & $stagedVenvPython -c $entryCheckCmd
Assert-LastExitCode "Staged launcher_entry.py & DesktopPaths verification"
if (($entryOutput | Out-String) -notmatch "LAUNCHER_ENTRY_OK") {
    throw "Staged launcher_entry.py check did not emit LAUNCHER_ENTRY_OK."
}
Write-Host "  $($entryOutput.Trim())"

# ------------------------------------------------------------------------------
# Step 5: Staging Metrics & Optional Inno Setup 6 Compile
# ------------------------------------------------------------------------------
$sizeApp = Get-DirectorySizeMB -Path $stagingApp
$sizeBin = Get-DirectorySizeMB -Path $stagingBin
$sizePython = Get-DirectorySizeMB -Path $stagingPython
$sizeVenv = Get-DirectorySizeMB -Path $stagingVenv
$sizeTotal = [Math]::Round($sizeApp + $sizeBin + $sizePython + $sizeVenv, 2)

Write-Host "`n[5/5] Staging Summary:"
Write-Host "  app/    : $sizeApp MB"
Write-Host "  bin/    : $sizeBin MB"
Write-Host "  python/ : $sizePython MB"
Write-Host "  venv/   : $sizeVenv MB"
Write-Host "  TOTAL   : $sizeTotal MB (uncompressed staged payload)"

if ($StageOnly) {
    Write-Host "`n-StageOnly specified; skipping Inno Setup ISCC.exe compilation."
    exit 0
}

$isccExe = Find-IsccExecutable
if ($null -eq $isccExe) {
    throw "Inno Setup 6 compiler (ISCC.exe) was not found. Install Inno Setup 6 or re-run with -StageOnly."
}

$issPath = Join-Path $installerDir "second-brain.iss"
Write-Host "`nCompiling installer with $isccExe..."
& $isccExe $issPath
Assert-LastExitCode "Inno Setup ISCC.exe compilation"

$setupExe = Join-Path $distDir "SecondBrain-Setup-x64.exe"
if (-not (Test-Path -LiteralPath $setupExe)) {
    throw "Expected compiled installer not found at '$setupExe'."
}

$exeInfo = Get-Item -LiteralPath $setupExe
$exeSizeMB = [Math]::Round($exeInfo.Length / 1MB, 2)
$exeSha256 = (Get-FileHash -LiteralPath $setupExe -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "`nInstaller build complete:"
Write-Host "  Artifact : $setupExe"
Write-Host "  Size     : $exeSizeMB MB"
Write-Host "  SHA-256  : $exeSha256"
