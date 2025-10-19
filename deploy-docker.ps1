# EquiLens Docker Deployment Script
# Automates building, tagging, and pushing to Docker Hub

param(
    [Parameter(Mandatory=$true, HelpMessage="Version to deploy (e.g., v2.0.0)")]
    [string]$Version,

    [Parameter(HelpMessage="Docker Hub username")]
    [string]$Username = "vkrishna04",

    [Parameter(HelpMessage="Skip confirmation prompts")]
    [switch]$Force
)

# Validate version format
if ($Version -notmatch '^v\d+\.\d+\.\d+$') {
    Write-Host "❌ Invalid version format!" -ForegroundColor Red
    Write-Host "   Expected format: v2.0.0" -ForegroundColor Yellow
    Write-Host "   Provided: $Version" -ForegroundColor Yellow
    exit 1
}

Write-Host "=== EquiLens Docker Deployment ===" -ForegroundColor Cyan
Write-Host "Version:  $Version" -ForegroundColor White
Write-Host "Username: $Username" -ForegroundColor White
Write-Host "Image:    ${Username}/equilens:$Version" -ForegroundColor White
Write-Host ""

# Confirm deployment
if (-not $Force) {
    $confirm = Read-Host "Deploy to Docker Hub? (y/N)"
    if ($confirm -ne "y" -and $confirm -ne "Y") {
        Write-Host "Deployment cancelled." -ForegroundColor Yellow
        exit 0
    }
    Write-Host ""
}

# Check Docker is running
Write-Host "[0/5] Checking Docker..." -ForegroundColor Yellow
try {
    docker ps | Out-Null
    Write-Host "✅ Docker is running" -ForegroundColor Green
} catch {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host "   Please start Docker Desktop and try again." -ForegroundColor Yellow
    exit 1
}

# Check Docker Hub login
Write-Host "[0/5] Checking Docker Hub login..." -ForegroundColor Yellow
$loginCheck = docker info 2>&1 | Select-String "Username:"
if (-not $loginCheck) {
    Write-Host "❌ Not logged in to Docker Hub!" -ForegroundColor Red
    Write-Host "   Please run: docker login" -ForegroundColor Yellow
    exit 1
}
Write-Host "✅ Logged in to Docker Hub" -ForegroundColor Green
Write-Host ""

# Build image
Write-Host "[1/5] Building Docker image..." -ForegroundColor Yellow
$buildStart = Get-Date
docker compose build
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Build failed!" -ForegroundColor Red
    exit 1
}
$buildDuration = (Get-Date) - $buildStart
Write-Host "✅ Build complete ($([math]::Round($buildDuration.TotalSeconds, 1))s)" -ForegroundColor Green
Write-Host ""

# Tag image with version
Write-Host "[2/5] Tagging image..." -ForegroundColor Yellow
docker tag equilens:latest ${Username}/equilens:$Version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Tagging failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Tagged: ${Username}/equilens:$Version" -ForegroundColor Gray

# Tag image as latest
docker tag equilens:latest ${Username}/equilens:latest
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Tagging failed!" -ForegroundColor Red
    exit 1
}
Write-Host "  Tagged: ${Username}/equilens:latest" -ForegroundColor Gray

# Extract major.minor version for tagging strategy
$versionParts = $Version -replace '^v', '' -split '\.'
$majorMinor = "v$($versionParts[0]).$($versionParts[1])"
$major = "v$($versionParts[0])"

# Tag with major.minor
docker tag equilens:latest ${Username}/equilens:$majorMinor
Write-Host "  Tagged: ${Username}/equilens:$majorMinor" -ForegroundColor Gray

# Tag with major
docker tag equilens:latest ${Username}/equilens:$major
Write-Host "  Tagged: ${Username}/equilens:$major" -ForegroundColor Gray

Write-Host "✅ Tagging complete (4 tags)" -ForegroundColor Green
Write-Host ""

# Push to Docker Hub - Version tag
Write-Host "[3/5] Pushing version tag to Docker Hub..." -ForegroundColor Yellow
$pushStart = Get-Date
docker push ${Username}/equilens:$Version
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed!" -ForegroundColor Red
    exit 1
}
$pushDuration = (Get-Date) - $pushStart
Write-Host "✅ Pushed ${Username}/equilens:$Version ($([math]::Round($pushDuration.TotalSeconds, 1))s)" -ForegroundColor Green
Write-Host ""

# Push to Docker Hub - Latest tag
Write-Host "[4/5] Pushing latest tag to Docker Hub..." -ForegroundColor Yellow
docker push ${Username}/equilens:latest
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Push failed!" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Pushed ${Username}/equilens:latest" -ForegroundColor Green

# Push to Docker Hub - Major.Minor tag
docker push ${Username}/equilens:$majorMinor
Write-Host "✅ Pushed ${Username}/equilens:$majorMinor" -ForegroundColor Green

# Push to Docker Hub - Major tag
docker push ${Username}/equilens:$major
Write-Host "✅ Pushed ${Username}/equilens:$major" -ForegroundColor Green
Write-Host ""

# Update setup script
Write-Host "[5/5] Updating setup script..." -ForegroundColor Yellow
$setupScriptPath = "setup-docker-simple.ps1"
if (Test-Path $setupScriptPath) {
    $content = Get-Content $setupScriptPath -Raw
    $content = $content -replace '\$EQUILENS_IMAGE = ".*?"', "`$EQUILENS_IMAGE = `"${Username}/equilens:$Version`""
    $content | Set-Content $setupScriptPath -NoNewline
    Write-Host "✅ Updated $setupScriptPath" -ForegroundColor Green
} else {
    Write-Host "⚠️  Setup script not found: $setupScriptPath" -ForegroundColor Yellow
}
Write-Host ""

# Summary
$totalDuration = $buildDuration + $pushDuration
Write-Host "=== Deployment Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "📦 Images pushed to Docker Hub:" -ForegroundColor Cyan
Write-Host "  • ${Username}/equilens:$Version     (exact version)" -ForegroundColor White
Write-Host "  • ${Username}/equilens:$majorMinor     (minor version)" -ForegroundColor White
Write-Host "  • ${Username}/equilens:$major         (major version)" -ForegroundColor White
Write-Host "  • ${Username}/equilens:latest       (latest)" -ForegroundColor White
Write-Host ""
Write-Host "⏱️  Total time: $([math]::Round($totalDuration.TotalSeconds, 1))s" -ForegroundColor Cyan
Write-Host ""
Write-Host "🔗 Docker Hub: https://hub.docker.com/r/${Username}/equilens/tags" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "  1. git add setup-docker-simple.ps1" -ForegroundColor White
Write-Host "  2. git commit -m 'Release $Version'" -ForegroundColor White
Write-Host "  3. git tag $Version" -ForegroundColor White
Write-Host "  4. git push origin main --tags" -ForegroundColor White
Write-Host ""
Write-Host "Test user installation:" -ForegroundColor Yellow
Write-Host "  docker pull ${Username}/equilens:$Version" -ForegroundColor White
Write-Host "  # Or run: .\setup-docker-simple.ps1" -ForegroundColor White
Write-Host ""
