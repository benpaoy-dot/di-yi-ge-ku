$ErrorActionPreference = "Stop"
$setup = Join-Path $PSScriptRoot "配置项目环境.ps1"
if (-not (Test-Path -LiteralPath $setup)) {
    throw "Project setup script was not found: $setup"
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $setup
exit $LASTEXITCODE
