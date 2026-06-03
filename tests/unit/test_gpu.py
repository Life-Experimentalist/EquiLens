"""Unit tests for equilens.core.gpu.GPUManager"""

import subprocess
from unittest.mock import patch

import pytest

from equilens.core.gpu import GPUManager

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NVIDIA_SMI_OUTPUT = (
    "+-----------------------------------------------------------------------------+\n"
    "| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |\n"
    "|-------------------------------+----------------------+----------------------+\n"
    "| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |\n"
    "| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |\n"
    "|                               |                      |               MIG M. |\n"
    "|===============================+======================+======================|\n"
    "|   0  GeForce RTX 3080   Off  | 00000000:01:00.0 Off |                  N/A |\n"
    "| 33%   54C    P2   120W / 320W |   5000MiB / 10240MiB |     10%      Default |\n"
    "+-----------------------------------------------------------------------------+\n"
)

NVCC_OUTPUT = (
    "nvcc: NVIDIA (R) Cuda compiler driver\n"
    "Copyright (c) 2005-2022 NVIDIA Corporation\n"
    "Built on Mon_Oct_24_19:12:59_PDT_2022\n"
    "Cuda compilation tools, release 12.0, V12.0.76\n"
    "Build cuda_12.0.r12.0/compiler.31968024_0\n"
)


def _make_completed(returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(["cmd"], returncode, stdout, stderr)


# ---------------------------------------------------------------------------
# _parse_nvidia_smi
# ---------------------------------------------------------------------------


def test_parse_nvidia_smi_extracts_driver_and_cuda_version():
    mgr = GPUManager()
    details = mgr._parse_nvidia_smi(NVIDIA_SMI_OUTPUT)
    assert details["driver_version"] == "525.85.12"
    assert details["cuda_version"] == "12.0"


def test_parse_nvidia_smi_extracts_gpu_model():
    mgr = GPUManager()
    details = mgr._parse_nvidia_smi(NVIDIA_SMI_OUTPUT)
    assert "gpu_model" in details
    assert "GeForce" in details["gpu_model"] or "RTX" in details["gpu_model"]


def test_parse_nvidia_smi_no_cuda_version():
    mgr = GPUManager()
    output = (
        "| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    |\n"
        "| No CUDA line here                                      |\n"
    )
    details = mgr._parse_nvidia_smi(output)
    # No NVIDIA-SMI line with CUDA Version → details should not have cuda_version
    # (or it may be absent entirely since the trigger line doesn't match)
    assert details.get("cuda_version", "N/A") == "N/A"


def test_parse_nvidia_smi_cuda_na_when_missing_from_header():
    mgr = GPUManager()
    # Line contains NVIDIA-SMI but no "CUDA Version:" part
    output = "| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12         |\n"
    details = mgr._parse_nvidia_smi(output)
    # Key should not be set (no matching line) → get returns "N/A"
    assert details.get("cuda_version", "N/A") == "N/A"


# ---------------------------------------------------------------------------
# _parse_cuda_version
# ---------------------------------------------------------------------------


def test_parse_cuda_version_extracts_version():
    mgr = GPUManager()
    assert mgr._parse_cuda_version(NVCC_OUTPUT) == "12.0"


def test_parse_cuda_version_unknown_when_no_release_line():
    mgr = GPUManager()
    assert mgr._parse_cuda_version("Some random output\nno useful info") == "Unknown"


# ---------------------------------------------------------------------------
# _run_command
# ---------------------------------------------------------------------------


def test_run_command_returns_failed_process_when_not_found_silent():
    mgr = GPUManager()
    with patch("equilens.core.gpu.subprocess.run", side_effect=FileNotFoundError):
        result = mgr._run_command(["nonexistent_tool"], silent_fail=True)
    assert result.returncode == 1


def test_run_command_raises_when_not_found_and_not_silent():
    mgr = GPUManager()
    with patch("equilens.core.gpu.subprocess.run", side_effect=FileNotFoundError):
        with pytest.raises(FileNotFoundError):
            mgr._run_command(["nonexistent_tool"], silent_fail=False)


# ---------------------------------------------------------------------------
# check_gpu_support
# ---------------------------------------------------------------------------


def test_check_gpu_support_nvidia_driver_false_when_smi_fails():
    mgr = GPUManager()

    def fake_run_command(cmd, silent_fail=True):
        # All commands fail
        return _make_completed(returncode=1)

    mgr._run_command = fake_run_command
    mgr._check_docker_gpu = lambda: False

    result = mgr.check_gpu_support()

    assert result["nvidia_driver"] is False
    assert result["gpu_available"] is False


def test_check_gpu_support_nvidia_driver_true_when_smi_succeeds():
    mgr = GPUManager()

    def fake_run_command(cmd, silent_fail=True):
        if cmd[0] == "nvidia-smi":
            return _make_completed(returncode=0, stdout=NVIDIA_SMI_OUTPUT)
        return _make_completed(returncode=1)

    mgr._run_command = fake_run_command
    mgr._check_docker_gpu = lambda: False

    result = mgr.check_gpu_support()

    assert result["nvidia_driver"] is True
    assert result["cuda_runtime"] is True
    assert result["cuda_version"] == "12.0"


def test_check_gpu_support_gpu_available_true_when_all_pass():
    mgr = GPUManager()

    def fake_run_command(cmd, silent_fail=True):
        if cmd[0] == "nvidia-smi":
            return _make_completed(returncode=0, stdout=NVIDIA_SMI_OUTPUT)
        return _make_completed(returncode=1)

    mgr._run_command = fake_run_command
    mgr._check_docker_gpu = lambda: True

    result = mgr.check_gpu_support()

    assert result["gpu_available"] is True


# ---------------------------------------------------------------------------
# get_performance_recommendation
# ---------------------------------------------------------------------------


def test_get_performance_recommendation_gpu_available():
    mgr = GPUManager()
    mgr.gpu_info = {
        "nvidia_driver": True,
        "cuda_runtime": True,
        "docker_gpu": True,
        "gpu_available": True,
        "gpu_details": {},
    }
    rec = mgr.get_performance_recommendation()
    assert "GPU" in rec


def test_get_performance_recommendation_cpu_mode():
    mgr = GPUManager()
    mgr.gpu_info = {
        "nvidia_driver": False,
        "cuda_runtime": False,
        "docker_gpu": False,
        "gpu_available": False,
        "gpu_details": {},
    }
    rec = mgr.get_performance_recommendation()
    assert "CPU" in rec


def test_get_performance_recommendation_triggers_check_when_no_info():
    mgr = GPUManager()

    def fake_check():
        mgr.gpu_info = {
            "gpu_available": False,
            "nvidia_driver": False,
            "cuda_runtime": False,
            "docker_gpu": False,
            "gpu_details": {},
        }
        return mgr.gpu_info

    mgr.check_gpu_support = fake_check
    rec = mgr.get_performance_recommendation()
    assert "CPU" in rec
