$ErrorActionPreference = "Stop"
$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$pythonReady = $false
if (Test-Path -LiteralPath $python) {
    & $python -c "import sys, tkinter, cv2, insightface, onnxruntime, PIL; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" 2>$null
    $pythonReady = ($LASTEXITCODE -eq 0)
}
if (-not $pythonReady) {
    $setup = Join-Path $PSScriptRoot "配置项目环境.ps1"
    if (-not (Test-Path -LiteralPath $setup)) {
        throw "找不到项目环境配置脚本：$setup"
    }
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $setup
    if ($LASTEXITCODE -ne 0) { throw "项目环境配置失败，退出码：$LASTEXITCODE。" }
}
Push-Location $PSScriptRoot
try {
    & $python (Join-Path $PSScriptRoot "face_welcome_app.py")
    if ($LASTEXITCODE -ne 0) { throw "程序退出异常，退出码：$LASTEXITCODE。请查看上方错误。" }
} catch {
    Write-Host "启动失败：" -ForegroundColor Red
    Write-Host $_ -ForegroundColor Red
    Read-Host "按 Enter 关闭此窗口"
    exit 1
} finally {
    Pop-Location
}
