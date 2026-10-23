# One-command setup for Windows (PowerShell).
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
# Creates .venv (CPU PyTorch + requirements), downloads Qwen2.5-0.5B-Instruct into models/, and runs npm install.
# The app picks up .venv automatically, so afterwards just run: npm run electron:dev
param(
  [string]$Model = "Qwen/Qwen2.5-0.5B-Instruct",
  [switch]$SkipModel,
  [string]$Venv = ".venv"   # the app looks for .venv; another name is only for testing
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
# Stop on a failing external command ($ErrorActionPreference does not look at exe exit codes)
function Run([string]$what, [scriptblock]$cmd, [int]$retry = 0) {
  for ($i = 0; $i -le $retry; $i++) {
    & $cmd
    if ($LASTEXITCODE -eq 0) { return }
    if ($i -lt $retry) { Write-Host "  retrying ($what)..."; Start-Sleep -Seconds 5 }
  }
  throw "Failed: $what (exit code $LASTEXITCODE)"
}

Step "Checking Python and Node.js"
# The newest Python may not have wheels for every dependency yet (pyyaml failed to build on 3.15),
# so with the py launcher prefer a stable version: 3.12, 3.11, 3.13, 3.10
$py = $null; $pyArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
  foreach ($v in "3.12", "3.11", "3.13", "3.10") {
    $ErrorActionPreference = "Continue"   # a missing version writes to stderr; do not treat that as fatal
    & py "-$v" -c "import sys" 2>$null
    $ok = ($LASTEXITCODE -eq 0)
    $ErrorActionPreference = "Stop"
    if ($ok) { $py = "py"; $pyArgs = @("-$v"); break }
  }
}
if (-not $py) { $py = "python" }
& $py @pyArgs -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13), sys.version; print(sys.version)"
if ($LASTEXITCODE -ne 0) { throw "Python 3.10-3.13 is required: https://www.python.org/downloads/" }
node --version
if ($LASTEXITCODE -ne 0) { throw "Node.js 22 LTS is required: https://nodejs.org/" }

$vpy = "$Venv\Scripts\python.exe"
if (Test-Path $vpy) {
  & $vpy -c "import sys; assert (3,10) <= sys.version_info[:2] <= (3,13)"
  if ($LASTEXITCODE -ne 0) { throw "$Venv was made with an unsupported Python. Delete the $Venv folder and run this script again." }
} else {
  Step "Creating $Venv"
  Run "create $Venv" { & $py @pyArgs -m venv $Venv }
}

Step "Installing PyTorch (CPU) and requirements"
Run "upgrade pip" { & $vpy -m pip install --upgrade pip }
# Antivirus can lock a freshly downloaded wheel (WinError 32), so retry once
Run "install torch" { & $vpy -m pip install torch --index-url https://download.pytorch.org/whl/cpu } 1
Run "install requirements" { & $vpy -m pip install -r requirements.txt huggingface_hub } 1

if (-not $SkipModel) {
  $dir = "models\" + ($Model.Split("/")[-1])
  if (-not (Test-Path "$dir\config.json")) {
    Step "Downloading $Model (about 1 GB)"
    Run "download model" { & $vpy -c "from huggingface_hub import snapshot_download; snapshot_download('$Model', local_dir=r'$dir')" } 1
  }
}

Step "Installing app dependencies (npm install)"
Run "npm install" { npm install }

Step "Done. Start the app with:  npm run electron:dev"
Write-Host "Then click 'Open model folder' and choose the folder under models\" -ForegroundColor Green
