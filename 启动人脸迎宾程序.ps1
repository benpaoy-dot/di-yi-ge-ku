$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "项目虚拟环境尚未配置。请先运行：powershell -NoProfile -ExecutionPolicy Bypass -File `"$PSScriptRoot\配置项目环境.ps1`""
}
Push-Location $PSScriptRoot
try {
    & $python (Join-Path $PSScriptRoot "face_welcome_app.py")
    if ($LASTEXITCODE -ne 0) { throw "程序退出异常，退出码：$LASTEXITCODE。请查看上方错误。" }
} finally {
    Pop-Location
}
