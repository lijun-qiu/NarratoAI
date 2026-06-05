# 启动 NarratoAI 多开实例（Windows PowerShell）
# 用法:
#   .\scripts\start_instance.ps1           # 单实例 8501
#   .\scripts\start_instance.ps1 -Instance 1
#   .\scripts\start_instance.ps1 -Instance 2
#   .\scripts\start_instance.ps1 -Instance dev -Port 8600

param(
    [string]$Instance = "",
    [int]$Port = 0
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ProjectRoot

$argsList = @()
if ($Instance) {
    $argsList += @("-i", $Instance)
}
if ($Port -gt 0) {
    $argsList += @("-p", "$Port")
}

python run_webui.py @argsList
