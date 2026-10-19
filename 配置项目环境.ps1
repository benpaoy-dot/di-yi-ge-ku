$ErrorActionPreference = "Stop"
Push-Location $PSScriptRoot
try {
    $venv = Join-Path $PSScriptRoot ".venv"
    $python = Join-Path $venv "Scripts\python.exe"

    function Get-PythonMajorMinor([string] $Executable) {
        if (-not (Test-Path -LiteralPath $Executable)) {
            return $null
        }
        $output = & $Executable -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $null
        }
        $line = $output | Select-Object -Last 1
        if ($null -eq $line) {
            return $null
        }
        return $line.ToString().Trim()
    }

    function Find-Python310Executable {
        $launcher = Get-Command py -ErrorAction SilentlyContinue
        if ($null -eq $launcher) {
            throw "项目环境不可用，且找不到 Python 启动器。请先安装 Windows x64 Python 3.10。"
        }
        $entries = @(& $launcher.Source -0p 2>&1)
        if ($LASTEXITCODE -ne 0) {
            throw "无法读取已安装的 Python 列表，请修复 Python 启动器后重试。"
        }
        foreach ($entry in $entries) {
            $line = $entry.ToString()
            if ($line -notmatch '(?i)3\.10(?:\.\d+)?') {
                continue
            }
            $match = [regex]::Match($line, '(?i)(?<path>[A-Z]:\\.*\\python(?:w)?\.exe)\s*$')
            if (-not $match.Success) {
                continue
            }
            $candidate = $match.Groups['path'].Value.Trim()
            if ((Get-PythonMajorMinor $candidate) -eq "3.10") {
                return $candidate
            }
        }
        throw "没有找到可运行的 Windows x64 Python 3.10。请安装 Python 3.10 后重新启动。"
    }

    $version = Get-PythonMajorMinor $python
    if ($version -ne "3.10") {
        $bootstrapPython = Find-Python310Executable
        if (Test-Path -LiteralPath $venv) {
            $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
            $backup = Join-Path $PSScriptRoot ".venv.broken-$stamp"
            $suffix = 1
            while (Test-Path -LiteralPath $backup) {
                $backup = Join-Path $PSScriptRoot ".venv.broken-$stamp-$suffix"
                $suffix++
            }
            Move-Item -LiteralPath $venv -Destination $backup
            Write-Host "已备份不可用的项目环境：$backup"
        }

        & $bootstrapPython -m venv $venv
        if ($LASTEXITCODE -ne 0) { throw "创建 Python 3.10 虚拟环境失败。" }
        $python = Join-Path $venv "Scripts\python.exe"
        $version = Get-PythonMajorMinor $python
    }

    if ($version -ne "3.10") {
        throw "新建的项目环境不是 Python 3.10，当前检测结果：$version"
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
