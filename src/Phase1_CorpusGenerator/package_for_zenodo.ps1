<#
 Copyright 2025 Krishna GSVV

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#>

<#
PowerShell helper to package CorpusGen artifacts for Zenodo upload.
Creates a zip containing the corpus CSV, word_lists.json, generate_corpus.py,
word_lists_schema.json, README.md, CITATION.cff, and LICENSE.md (if present).
It records the current git commit hash and timestamp in a manifest file inside the archive.
#>

param(
    [string]$OutputDir = ".",
    [string]$CorpusFolder = "corpus",
    [string]$OutFile = "equilens_corpus_release.zip"
)

Push-Location $PSScriptRoot

# Discover files
$files = @()
$files += (Get-ChildItem -Path "$CorpusFolder" -Filter "audit_corpus_*.csv" -File -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
$files += (Join-Path $PSScriptRoot "generate_corpus.py")
$files += (Join-Path $PSScriptRoot "word_lists.json")
$files += (Join-Path $PSScriptRoot "word_lists_schema.json")
$files += (Join-Path $PSScriptRoot "README.md")
$files += (Join-Path $PSScriptRoot "CITATION.cff")
$licensePath = Join-Path $PSScriptRoot "LICENSE.md"
if (Test-Path $licensePath) { $files += $licensePath }

# Filter missing
$files = $files | Where-Object { Test-Path $_ }
if (-not $files) {
    Write-Error "No files found to package. Ensure you ran generate_corpus.py and have corpus CSVs."
    Pop-Location
    exit 1
}

# Create manifest with git commit and timestamp if available
$manifest = @{ }
try {
    $commit = git rev-parse --short HEAD 2>$null
    if ($LASTEXITCODE -eq 0) { $manifest.commit = $commit.Trim() }
} catch {
    # git not available or not a repo
}
$manifest.timestamp = (Get-Date).ToString("o")
$manifestFile = Join-Path $PSScriptRoot "release_manifest.json"
$manifest | ConvertTo-Json -Depth 3 | Out-File -FilePath $manifestFile -Encoding utf8

# Prepare output path
$outPath = Join-Path (Resolve-Path $OutputDir) $OutFile

# Build zip
$zipTemp = Join-Path $env:TEMP "equilens_package_$(Get-Random).zip"
if (Test-Path $zipTemp) { Remove-Item $zipTemp -Force }

# Add files and manifest
$itemsToZip = $files + $manifestFile
Compress-Archive -Path $itemsToZip -DestinationPath $zipTemp -Force
Move-Item -Path $zipTemp -Destination $outPath -Force

Write-Host "Created package: $outPath"

# Clean up
Remove-Item $manifestFile -Force

Pop-Location

Write-Host "Done."
