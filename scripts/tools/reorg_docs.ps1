# Reorganize docs into subfolders
$ErrorActionPreference = 'Stop'
$root = 'v:\Code\ProjectCode\EquiLens\docs'
Set-Location -Path $root
Write-Host "Working in: $root"

$folders = @('setup', 'docker', 'architecture', 'analytics', 'auditing', 'misc', 'archived')
foreach ($f in $folders) {
    if (-not (Test-Path $f)) {
        New-Item -ItemType Directory -Path $f | Out-Null
        Write-Host "Created: $f"
    }
    else {
        Write-Host "Exists: $f"
    }
}

# Move mapping
$map = @{
    'setup'        = @('SETUP_COMPLETE.md', 'SETUP_IMPROVEMENTS.md', 'SETUP_SCRIPTS_GUIDE.md', 'SETUP_SCRIPTS_COMPARISON.md', 'SETUP_DOCKER_LOCAL_CHANGES.md', 'SMART_SETUP_GUIDE.md', 'SMART_SETUP_COMPLETE.md', 'TROUBLESHOOTING_SETUP.md', 'QUICKSTART.md')
    'docker'       = @('DOCKER_README.md', 'DOCKER_SETUP.md', 'DOCKER_SETUP_COMPLETE.md', 'DOCKER_HUB_DEPLOYMENT.md', 'DOCKER_DEPLOY_QUICKREF.md', 'DOCKER_CONFIG_GUIDE.md', 'DOCKER_BUILD_FIX.md', 'DOCKER_CHECKLIST.md', 'DOCKER_IMAGE_NAMING_FIX.md', 'DOCKER_FIXES_COMPLETE.md', 'DOCKER_SETUP_COMPARISON.md', 'DOCKER_SIMPLE.md', 'DOCKER_SIMPLIFIED_SETUP.md', 'DOCKER_SIMPLIFIED_SUMMARY.md')
    'architecture' = @('ARCHITECTURE.md', 'ARCHITECTURE_SIMPLE.md', 'PIPELINE.md')
    'analytics'    = @('QUICK_REFERENCE.md', 'ADVANCED_ANALYTICS_GUIDE.md', 'ANALYTICS_READY.md', 'ANALYTICS_IMPROVEMENTS_COMPLETE.md', 'FLEXIBLE_ANALYTICS_COMPLETE.md', 'FLEXIBLE_ANALYTICS_REFERENCE.md', 'INTERACTIVE_ANALYTICS_GUIDE.md', 'MULTI_CATEGORY_QUICK_REFERENCE.md', 'N_CATEGORY_SUPPORT_COMPLETE.md', 'CLI_ANALYTICS_FIX_COMPLETE.md')
    'auditing'     = @('AUDITING_MECHANISM.md', 'ENHANCED_AUDITOR_DEFAULT.md', 'VOLUME_MANAGEMENT.md', 'SMART_IMAGE_MANAGEMENT.md', 'EXISTING_OLLAMA_GUIDE.md', 'OLLAMA_SETUP.md', 'OLLAMA_FLEXIBLE_SETUP.md')
    'misc'         = @('CODE_ANALYSIS_REPORT.md', 'REPORT.md', 'SCRIPT_REORGANIZATION_PLAN.md', 'RECOVERY_NOTES.md')
}

Write-Host "Moving files..."
foreach ($kv in $map.GetEnumerator()) {
    $dest = $kv.Key
    foreach ($file in $kv.Value) {
        if (Test-Path $file) {
            $target = Join-Path $dest $file
            if (Test-Path $target) {
                Write-Host "Target exists, archiving $file"
                Move-Item -Path $file -Destination (Join-Path 'archived' $file) -Force
            }
            else {
                Move-Item -Path $file -Destination $target -Force
                Write-Host "Moved $file -> $dest/"
            }
        }
        else {
            Write-Host "Not found: $file"
        }
    }
}

# Archive candidates
$archiveCandidates = @('ONE_CLICK_SETUP.md', 'DOCKER_HUB_DEPLOYMENT.md')
foreach ($f in $archiveCandidates) {
    if (Test-Path $f) {
        Move-Item -Path $f -Destination (Join-Path 'archived' $f) -Force
        Write-Host "Archived: $f"
    }
}

Write-Host "Final docs tree:";
Get-ChildItem -Path $root -Recurse -Depth 2 | ForEach-Object { Write-Host $_.FullName }

Write-Host "Done."
