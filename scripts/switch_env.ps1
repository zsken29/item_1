# conda-env-switch.ps1
# Conda 环境快速切换脚本

param(
    [Parameter(Position=0)]
    [string]$Action = "list"
)

# 定义常用环境
$ENVIRONMENTS = @{
    "vlmc" = @{
        Description = "VLMCompression - LLaVA/Bunny/InternVL"
        Projects = @("VLMCompression")
    }
    "cv_env" = @{
        Description = "计算机视觉环境"
        Projects = @("CSTVR", "CSTVR_ori", "lop_repro")
    }
    "d2l_env" = @{
        Description = "D2L 学习环境"
        Projects = @()
    }
}

# 彩色输出
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    $colorMap = @{
        "Red" = [ConsoleColor]::Red
        "Green" = [ConsoleColor]::Green
        "Yellow" = [ConsoleColor]::Yellow
        "Cyan" = [ConsoleColor]::Cyan
        "Magenta" = [ConsoleColor]::Magenta
        "White" = [ConsoleColor]::White
    }
    Write-Host $Message -ForegroundColor $colorMap[$Color]
}

# 列出所有环境
function List-Environments {
    Write-ColorOutput "`n📦 Conda 环境列表:`n" "Cyan"
    
    # 获取当前环境
    $currentEnv = $env:CONDA_DEFAULT_ENV
    if (-not $currentEnv) {
        $currentEnv = "base"
    }
    
    # 列出所有环境
    $envs = conda env list 2>$null | Select-Object -Skip 2
    
    foreach ($env in $envs) {
        $envName = $env.Trim().Split()[0]
        
        if ($envName -eq $currentEnv) {
            Write-Host "  [$envName] " -NoNewline -ForegroundColor Green
            Write-Host "(当前环境)" -ForegroundColor Gray
        } else {
            Write-Host "  $envName"
        }
        
        # 显示环境描述
        if ($ENVIRONMENTS.ContainsKey($envName)) {
            $desc = $ENVIRONMENTS[$envName].Description
            Write-Host "    → $desc" -ForegroundColor Gray
        }
    }
    
    Write-ColorOutput "`n提示: 使用 `./switch_env.ps1 activate <环境名>` 激活环境`n" "Yellow"
}

# 激活环境
function Activate-Environment {
    param([string]$EnvName)
    
    # 检查环境是否存在
    $exists = conda env list | Select-String -Pattern "^$EnvName "
    if (-not $exists) {
        Write-ColorOutput "❌ 环境 '$EnvName' 不存在!" "Red"
        Write-ColorOutput "使用 `./switch_env.ps1 list` 查看可用环境`n" "Yellow"
        return
    }
    
    Write-ColorOutput "🔄 激活环境: $EnvName..." "Cyan"
    
    # 获取 conda 安装路径
    $condaPath = (conda info --base).Trim()
    
    # 构建激活脚本
    $activateScript = "$condaPath\Scripts\activate.ps1"
    $envScript = "$condaPath\envs\$EnvName\Scripts\activate.ps1"
    
    if (Test-Path $envScript) {
        & $activateScript $EnvName
        
        # 验证激活
        if ($env:CONDA_DEFAULT_ENV -eq $EnvName) {
            Write-ColorOutput "✅ 环境 '$EnvName' 已激活!" "Green"
            
            # 显示环境信息
            $pythonPath = "$condaPath\envs\$EnvName\python.exe"
            if (Test-Path $pythonPath) {
                $version = & $pythonPath --version 2>$null
                Write-ColorOutput "   Python 版本: $version" "Gray"
            }
            
            # 显示相关项目
            if ($ENVIRONMENTS.ContainsKey($EnvName)) {
                $projects = $ENVIRONMENTS[$EnvName].Projects
                if ($projects.Count -gt 0) {
                    Write-ColorOutput "   相关项目: $($projects -join ', ')" "Gray"
                }
            }
        } else {
            Write-ColorOutput "⚠️  请手动激活: conda activate $EnvName" "Yellow"
        }
    } else {
        Write-ColorOutput "❌ 环境激活脚本不存在" "Red"
    }
    
    Write-Host ""
}

# 创建新环境
function New-Environment {
    param(
        [string]$EnvName,
        [string]$PythonVersion = "3.10"
    )
    
    Write-ColorOutput "🔨 创建新环境: $EnvName (Python $PythonVersion)..." "Cyan"
    
    conda create -n $EnvName python=$PythonVersion -y
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput "✅ 环境创建成功!`n" "Green"
        Activate-Environment -EnvName $EnvName
    } else {
        Write-ColorOutput "❌ 环境创建失败`n" "Red"
    }
}

# 删除环境
function Remove-Environment {
    param([string]$EnvName)
    
    if ($env:CONDA_DEFAULT_ENV -eq $EnvName) {
        Write-ColorOutput "⚠️  当前正在使用环境 '$EnvName'，无法删除!" "Yellow"
        Write-ColorOutput "请先切换到其他环境: ./switch_env.ps1 activate <其他环境>`n" "Yellow"
        return
    }
    
    Write-ColorOutput "🗑️  删除环境: $EnvName..." "Cyan"
    Write-ColorOutput "⚠️  此操作不可恢复!" "Yellow"
    
    $confirm = Read-Host "确认删除? (y/n)"
    
    if ($confirm -eq 'y' -or $confirm -eq 'Y') {
        conda env remove -n $EnvName -y
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ 环境 '$EnvName' 已删除!`n" "Green"
        }
    } else {
        Write-ColorOutput "取消删除`n" "Yellow"
    }
}

# 运行脚本
function Run-Script {
    param(
        [string]$EnvName,
        [string]$Script
    )
    
    Write-ColorOutput "🚀 在环境 '$EnvName' 中运行: $Script" "Cyan"
    
    conda run -n $EnvName python $Script
    
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput "❌ 脚本执行失败 (exit code: $LASTEXITCODE)`n" "Red"
    }
}

# 显示帮助
function Show-Help {
    Write-ColorOutput "`n📚 Conda 环境管理脚本`n" "Cyan"
    Write-Host "用法: .\switch_env.ps1 <命令> [参数]`n"
    
    Write-ColorOutput "命令:" "Green"
    Write-Host "  list                       列出所有环境"
    Write-Host "  activate <环境名>          激活环境"
    Write-Host "  new <环境名> [版本]        创建新环境 (默认 Python 3.10)"
    Write-Host "  remove <环境名>            删除环境"
    Write-Host "  run <环境名> <脚本>        在指定环境中运行脚本"
    Write-Host "  help                       显示帮助`n"
    
    Write-ColorOutput "示例:" "Green"
    Write-Host "  .\switch_env.ps1 list"
    Write-Host "  .\switch_env.ps1 activate vlmc"
    Write-Host "  .\switch_env.ps1 new myenv 3.11"
    Write-Host "  .\switch_env.ps1 run vlmc train.py`n"
    
    Write-ColorOutput "常用环境:" "Yellow"
    Write-Host "  vlmc    - VLMCompression (LLaVA/Bunny/InternVL)"
    Write-Host "  cv_env  - 计算机视觉 (CSTVR, lop_repro)"
    Write-Host "  d2l_env - D2L 学习`n"
}

# 主程序
switch ($Action.ToLower()) {
    "list" {
        List-Environments
    }
    "activate" {
        if ($args[0]) {
            Activate-Environment -EnvName $args[0]
        } else {
            Write-ColorOutput "❌ 请指定环境名`n" "Red"
            Show-Help
        }
    }
    "new" {
        if ($args[0]) {
            $pythonVer = if ($args[1]) { $args[1] } else { "3.10" }
            New-Environment -EnvName $args[0] -PythonVersion $pythonVer
        } else {
            Write-ColorOutput "❌ 请指定环境名`n" "Red"
            Show-Help
        }
    }
    "remove" {
        if ($args[0]) {
            Remove-Environment -EnvName $args[0]
        } else {
            Write-ColorOutput "❌ 请指定环境名`n" "Red"
        }
    }
    "run" {
        if ($args[0] -and $args[1]) {
            Run-Script -EnvName $args[0] -Script $args[1]
        } else {
            Write-ColorOutput "❌ 请指定环境名和脚本`n" "Red"
        }
    }
    "help" {
        Show-Help
    }
    default {
        if ($Action -ne "help") {
            Write-ColorOutput "❌ 未知命令: $Action`n" "Red"
        }
        Show-Help
    }
}
