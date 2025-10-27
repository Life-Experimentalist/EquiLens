# Docker Image Attestation Setup Guide

## What is Attestation?

**Attestation** creates a cryptographic proof that verifies:
- ✅ Your Docker image was built by GitHub Actions
- ✅ The build matches the source code in your repository
- ✅ The build hasn't been tampered with

This adds an extra layer of security and trust for users pulling your image.

## Current Status

✅ **Workflow Updated:** Attestation step is now included
✅ **Permissions Added:** `attestations: write` permission enabled
✅ **Changes Pushed:** Workflow is live on GitHub

## Required Repository Settings

To make attestation work, you need to enable it in your repository settings:

### Method 1: Via GitHub Web UI (Recommended)

1. **Navigate to Actions Settings:**
   - Go to: https://github.com/Life-Experimentalist/EquiLens/settings/actions

2. **Check Workflow Permissions:**
   - Under **"Workflow permissions"**, ensure:
     - 🔘 **"Read and write permissions"** is selected
     - ✅ **"Allow GitHub Actions to create and approve pull requests"** (optional)

3. **Look for Attestation Settings:**
   - Scroll through settings page for:
     - **"Artifact attestations"**
     - **"Build provenance"**
     - **"Supply chain security"**
   - Enable any attestation-related options

### Method 2: Wait for GitHub to Fully Roll Out

> **Note:** GitHub is still rolling out attestation features. If you don't see specific attestation settings yet, the workflow may work automatically with the `attestations: write` permission we added.

## Testing the Attestation

Once enabled, every push to `main` or tagged release will:

1. Build the Docker image ✅
2. Push to GitHub Container Registry ✅
3. Create attestation metadata ✅

### Verify Attestation Works

After the next workflow run, check the Actions log:
- ✅ Success: "Attestation created successfully"
- ❌ Error: Still shows "Resource not accessible" → Settings not enabled yet

### View Attestation

Once working, you can verify attestations:

```powershell
# Install GitHub CLI attestation extension
gh extension install github/gh-attestation

# Verify the image
gh attestation verify oci://ghcr.io/life-experimentalist/equilens:latest --owner Life-Experimentalist
```

## What Changed in the Workflow

```yaml
permissions:
    contents: read
    packages: write
    id-token: write
    attestations: write  # ← Added this permission

steps:
    # ... build steps ...

    - name: Attest Docker image
      if: github.event_name != 'pull_request'
      uses: actions/attest-build-provenance@v1
      with:
          subject-name: ${{env.REGISTRY}}/${{env.IMAGE_NAME}}
          subject-digest: ${{steps.build.outputs.digest}}
          push-to-registry: true
```

## Troubleshooting

### Error: "Resource not accessible by integration"

**Cause:** Attestation not enabled in repository settings

**Solution:**
1. Check repository settings (see Method 1 above)
2. Verify workflow permissions are set to "Read and write"
3. Wait for GitHub to fully roll out attestation features to your org

### Error: "Subject digest not found"

**Cause:** Build step didn't output a digest

**Solution:** Ensure the build step has `id: build` and outputs a digest

## Benefits of Attestation

✅ **Provenance Verification:** Users can verify the image source
✅ **Supply Chain Security:** Prevents tampering
✅ **Compliance:** Meets security standards (SLSA)
✅ **Trust:** Shows professional build practices

## Further Reading

- [GitHub Attestations Documentation](https://docs.github.com/en/actions/security-guides/using-artifact-attestations)
- [SLSA Framework](https://slsa.dev/)
- [Docker Build Provenance](https://docs.docker.com/build/attestations/slsa-provenance/)

---

**Status:** Configuration complete, pending repository settings enablement
**Last Updated:** October 20, 2025
