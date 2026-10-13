$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
            throw "找不到 Python 启动器，请先安装 Windows x64 Python 3.10。"
        }
        & py -3.10 -m venv (Join-Path $PSScriptRoot ".venv")
        if ($LASTEXITCODE -ne 0) { throw "创建虚拟环境失败，请确认已安装 Python 3.10。" }
    }
    $version = & $python -c "import sys; print('.'.join(map(str, sys.version_info[:2])))"
    if ($LASTEXITCODE -ne 0 -or $version -ne "3.10") {
        throw "现有 .venv 不是 Python 3.10，请先备份并重命名该环境后重新配置。"
    }
    & $python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "更新 pip 失败。" }
    $requirements = Join-Path $PSScriptRoot "requirements-lock.txt"
    if (-not (Test-Path -LiteralPath $requirements)) {
        $requirements = Join-Path $PSScriptRoot "requirements.txt"
    }
    & $python -m pip install --only-binary=:all: -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "安装依赖失败，请检查网络连接及上方错误。" }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { throw "依赖存在冲突。" }
    & $python (Join-Path $PSScriptRoot "verify_environment.py")
    if ($LASTEXITCODE -ne 0) { throw "环境检查未通过，请检查上方具体错误。" }
    Write-Host "配置成功：InsightFace / ArcFace 主识别已验证。可运行 启动人脸迎宾程序.ps1。"
} finally {
    Pop-Location
}
