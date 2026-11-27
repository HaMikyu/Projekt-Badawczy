# Buduje dist\DetektorAI\DetektorAI.exe (okno WebView2 + serwer w tym samym procesie, torch CPU).
#   powershell -ExecutionPolicy Bypass -File app\build_exe.ps1
# Wymaga srodowiska .venv-build (Python 3.11, torch CPU, pywebview, pyinstaller) - patrz README.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$py = Join-Path $root ".venv-build\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Brak .venv-build - utworz je wg README (sekcja Aplikacja EXE)." }

& $py -m PyInstaller --noconfirm --clean --distpath dist --workpath build app\DetektorAI.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller nie powiodl sie" }

$out = Join-Path $root "dist\DetektorAI"
foreach ($d in @("models\system-one-final", "models\distilbert-final")) {
    New-Item -ItemType Directory -Force (Join-Path $out $d) | Out-Null
    Copy-Item -Recurse -Force (Join-Path $root "$d\*") (Join-Path $out $d)
}
New-Item -ItemType Directory -Force (Join-Path $out "statistical\runs") | Out-Null
Copy-Item -Force (Join-Path $root "statistical\runs\model.joblib"), (Join-Path $root "statistical\runs\results.json") (Join-Path $out "statistical\runs")

$size = (Get-ChildItem -Recurse $out | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host ("Gotowe: {0}\DetektorAI.exe  (folder {1:N2} GB)" -f $out, $size)
