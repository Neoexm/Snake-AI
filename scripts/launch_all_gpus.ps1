<#
.SYNOPSIS
    Launch multi-GPU training for Snake RL

.DESCRIPTION
    Wrapper script to launch multi-GPU training using tools/launcher_multi_gpu.py

.PARAMETER Gpus
    Comma-separated GPU IDs or 'all' (default: all)

.PARAMETER Config
    Config file path (default: train/configs/base.yaml)

.PARAMETER TotalTimesteps
    Total timesteps per GPU (default: 500000)

.PARAMETER Seed
    Base random seed (default: 42)

.PARAMETER RunName
    Base run name (default: timestamp)

.PARAMETER Logdir
    Log directory (default: runs)

.PARAMETER AutoScale
    Enable auto-scaling: true or false (default: true)

.PARAMETER Wait
    Wait for all processes to complete (switch)

.EXAMPLE
    .\scripts\launch_all_gpus.ps1

.EXAMPLE
    .\scripts\launch_all_gpus.ps1 -Gpus "0,1,2" -TotalTimesteps 5000000

.EXAMPLE
    .\scripts\launch_all_gpus.ps1 -Config "train/configs/large.yaml" -Wait

#>

param(
    [string]$Gpus = "all",
    [string]$Config = "train/configs/base.yaml",
    [int]$TotalTimesteps = 500000,
    [int]$Seed = 42,
    [string]$RunName = "",
    [string]$Logdir = "runs",
    [string]$AutoScale = "true",
    [switch]$Wait
)

# Build command
$cmd = "python tools/launcher_multi_gpu.py"
$cmd += " --gpus $Gpus"
$cmd += " --config $Config"
$cmd += " --total-timesteps $TotalTimesteps"
$cmd += " --seed $Seed"
$cmd += " --logdir $Logdir"
$cmd += " --auto-scale $AutoScale"

if ($RunName) {
    $cmd += " --run-name $RunName"
}

if ($Wait) {
    $cmd += " --wait"
}

# Print command
Write-Host "Launching multi-GPU training..." -ForegroundColor Cyan
Write-Host "Command: $cmd" -ForegroundColor Gray
Write-Host ""

# Execute
Invoke-Expression $cmd

# Check exit code
if ($LASTEXITCODE -ne 0) {
    Write-Host "Training failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}