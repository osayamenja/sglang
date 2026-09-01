#!/usr/bin/env bash

set -Eeuo pipefail

readonly PURLIN_VERSION="0.6.0"
readonly ACCELERATE_VERSION="1.12.0"
readonly DATASETS_VERSION="5.0.1"
readonly FSSPEC_VERSION="2026.6.0"
readonly PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly VENV_DIR="${REPO_ROOT}/.venv"

CURRENT_STEP="startup"
CUDA_OVERRIDE=""
SKIP_SYSTEM_DEPS=0

log() {
    printf '\n==> %s\n' "$*"
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    printf '\nERROR: Installation failed during "%s" (exit code %s).\n' \
        "${CURRENT_STEP}" "${exit_code}" >&2
    exit "${exit_code}"
}
trap on_error ERR

usage() {
    cat <<'EOF'
Install this SGLang fork and Purlin into the repository's .venv.

Usage:
  bash scripts/install_purlin.sh [options]

Options:
  --cuda 12|13          Require a specific CUDA toolkit major version.
  --skip-system-deps    Do not install Ubuntu packages with apt-get.
  -h, --help            Show this help message.

CUDA is detected from nvcc. When --cuda is supplied, it must match the nvcc
selected by PATH.
EOF
}

while (($# > 0)); do
    case "$1" in
        --cuda)
            (($# >= 2)) || die "--cuda requires 12 or 13"
            CUDA_OVERRIDE=$2
            shift 2
            ;;
        --skip-system-deps)
            SKIP_SYSTEM_DEPS=1
            shift
            ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help for usage)"
            ;;
    esac
done

case "${CUDA_OVERRIDE}" in
    "" | 12 | 13) ;;
    *) die "--cuda must be 12 or 13" ;;
esac

CURRENT_STEP="platform validation"
[[ -f "${REPO_ROOT}/python/pyproject.toml" ]] || \
    die "could not find python/pyproject.toml under ${REPO_ROOT}"
[[ -r /etc/os-release ]] || die "cannot identify the operating system"
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || \
    die "this installer supports Ubuntu only (detected ${PRETTY_NAME:-unknown})"

install_system_dependencies() {
    local missing=()

    if ! command -v cc >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1; then
        missing+=(build-essential)
    fi
    if ! dpkg-query -W -f='${Status}' ca-certificates 2>/dev/null | \
        grep -q '^install ok installed$'; then
        missing+=(ca-certificates)
    fi
    command -v cmake >/dev/null 2>&1 || missing+=(cmake)
    command -v curl >/dev/null 2>&1 || missing+=(curl)
    command -v ninja >/dev/null 2>&1 || missing+=(ninja-build)
    command -v protoc >/dev/null 2>&1 || missing+=(protobuf-compiler)

    if ((${#missing[@]} == 0)); then
        echo "Ubuntu build dependencies are already installed."
        return
    fi

    local apt=(apt-get)
    if ((EUID != 0)); then
        command -v sudo >/dev/null 2>&1 || \
            die "sudo is required to install: ${missing[*]}"
        apt=(sudo apt-get)
    fi

    echo "Installing Ubuntu packages: ${missing[*]}"
    "${apt[@]}" update
    "${apt[@]}" install -y --no-install-recommends "${missing[@]}"
}

CURRENT_STEP="Ubuntu build dependencies"
log "Checking Ubuntu build dependencies"
if ((SKIP_SYSTEM_DEPS)); then
    echo "Skipping apt-get because --skip-system-deps was supplied."
else
    install_system_dependencies
fi

for command in cmake curl ninja protoc; do
    command -v "${command}" >/dev/null 2>&1 || \
        die "${command} is required; install the system dependencies or omit --skip-system-deps"
done
if ! command -v cc >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1; then
    die "C and C++ compilers are required; install build-essential"
fi

CURRENT_STEP="CUDA toolkit detection"
log "Detecting the CUDA toolkit"
command -v nvcc >/dev/null 2>&1 || \
    die "nvcc was not found in PATH; install or select a CUDA 12 or 13 toolkit"
NVCC_OUTPUT="$(nvcc --version)"
CUDA_RELEASE="$(sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' <<<"${NVCC_OUTPUT}" | head -n1)"
[[ -n "${CUDA_RELEASE}" ]] || die "could not parse the CUDA version from nvcc --version"
CUDA_MAJOR="${CUDA_RELEASE%%.*}"
case "${CUDA_MAJOR}" in
    12 | 13) ;;
    *) die "CUDA ${CUDA_RELEASE} is unsupported; CUDA 12 or 13 is required" ;;
esac
if [[ -n "${CUDA_OVERRIDE}" && "${CUDA_OVERRIDE}" != "${CUDA_MAJOR}" ]]; then
    die "--cuda ${CUDA_OVERRIDE} does not match nvcc (${CUDA_RELEASE}); select the intended nvcc through PATH"
fi
echo "Detected CUDA ${CUDA_RELEASE} from $(command -v nvcc)."

CURRENT_STEP="Rust toolchain"
log "Checking the Rust toolchain"
export PATH="${CARGO_HOME:-${HOME}/.cargo}/bin:${HOME}/.local/bin:${PATH}"
if command -v cargo >/dev/null 2>&1 && command -v rustc >/dev/null 2>&1; then
    echo "Rust is already installed: $(rustc --version), $(cargo --version)"
else
    bash "${REPO_ROOT}/scripts/ci/utils/install_rustup.sh"
fi
export PATH="${CARGO_HOME:-${HOME}/.cargo}/bin:${HOME}/.local/bin:${PATH}"
command -v cargo >/dev/null 2>&1 || die "cargo was not available after installing Rust"
command -v rustc >/dev/null 2>&1 || die "rustc was not available after installing Rust"

CURRENT_STEP="uv installation"
log "Checking uv"
if ! command -v uv >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -LsSf \
        https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${HOME}/.cargo/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || die "uv was not available after installation"
UV_BIN="$(command -v uv)"
echo "Using $(${UV_BIN} --version) from ${UV_BIN}."

CURRENT_STEP="Python virtual environment"
log "Preparing Python ${PYTHON_VERSION} in ${VENV_DIR}"
if [[ -x "${VENV_DIR}/bin/python" ]]; then
    VENV_PYTHON_VERSION="$("${VENV_DIR}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    [[ "${VENV_PYTHON_VERSION}" == "${PYTHON_VERSION}" ]] || \
        die "${VENV_DIR} uses Python ${VENV_PYTHON_VERSION}; remove it or move it aside before rerunning"
    echo "Reusing the existing Python ${VENV_PYTHON_VERSION} environment."
elif [[ -e "${VENV_DIR}" ]]; then
    die "${VENV_DIR} exists but is not a usable virtual environment; remove it or move it aside"
else
    "${UV_BIN}" venv "${VENV_DIR}" --python "${PYTHON_VERSION}" --seed --managed-python
fi

readonly PYTHON_BIN="${VENV_DIR}/bin/python"
readonly SGLANG_BIN="${VENV_DIR}/bin/sglang"
UV_PIP=("${UV_BIN}" pip install --python "${PYTHON_BIN}")

CURRENT_STEP="Purlin installation"
log "Installing Purlin ${PURLIN_VERSION}"
"${UV_PIP[@]}" "purlin==${PURLIN_VERSION}"

install_cuda12_dependencies() {
    "${UV_PIP[@]}" "flashinfer_python[cu12]==0.6.12"
    "${UV_PIP[@]}" --reinstall \
        --default-index https://download.pytorch.org/whl/cu129 \
        "torch==2.11.0" "torchaudio==2.11.0" torchvision
    "${UV_PIP[@]}" --reinstall \
        --default-index https://docs.sglang.ai/whl/cu129/ \
        "sglang-kernel==0.4.4+cu129"
    "${UV_PIP[@]}" --reinstall --no-deps \
        --default-index https://docs.sglang.ai/whl/cu129/ \
        "sgl-deep-gemm==0.1.4+cu129"
    "${UV_PIP[@]}" --editable "${REPO_ROOT}/python[cuda12,diffusion]"
}

CURRENT_STEP="SGLang CUDA ${CUDA_MAJOR} installation"
log "Installing this SGLang checkout for CUDA ${CUDA_MAJOR}"
if [[ "${CUDA_MAJOR}" == "12" ]]; then
    install_cuda12_dependencies
else
    "${UV_PIP[@]}" --editable "${REPO_ROOT}/python[cuda13,diffusion]"
fi

CURRENT_STEP="Hugging Face datasets compatibility"
log "Installing datasets ${DATASETS_VERSION} with compatible fsspec ${FSSPEC_VERSION}"
# Installing SGLang into Purlin's existing environment can otherwise preserve a
# newer fsspec and backtrack to datasets 1.1.1. That release cannot import with
# current PyArrow because it still references the removed PyExtensionType API.
"${UV_PIP[@]}" \
    "datasets==${DATASETS_VERSION}" \
    "fsspec==${FSSPEC_VERSION}"

CURRENT_STEP="installation verification"
log "Verifying the installation"
EXPECTED_ACCELERATE_VERSION="${ACCELERATE_VERSION}" \
EXPECTED_CUDA_MAJOR="${CUDA_MAJOR}" EXPECTED_DATASETS_VERSION="${DATASETS_VERSION}" \
EXPECTED_FSSPEC_VERSION="${FSSPEC_VERSION}" EXPECTED_PURLIN_VERSION="${PURLIN_VERSION}" \
    "${PYTHON_BIN}" <<'PY'
import importlib.metadata
import os

import accelerate
import datasets
import fsspec
import purlin
import pyarrow
import sglang
import torch
from datasets import Dataset
from diffusers.image_processor import VaeImageProcessor
from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig

expected_accelerate = os.environ["EXPECTED_ACCELERATE_VERSION"]
expected_cuda = os.environ["EXPECTED_CUDA_MAJOR"]
expected_datasets = os.environ["EXPECTED_DATASETS_VERSION"]
expected_fsspec = os.environ["EXPECTED_FSSPEC_VERSION"]
expected_purlin = os.environ["EXPECTED_PURLIN_VERSION"]
accelerate_version = importlib.metadata.version("accelerate")
datasets_version = importlib.metadata.version("datasets")
fsspec_version = importlib.metadata.version("fsspec")
purlin_version = importlib.metadata.version("purlin")
torch_cuda = torch.version.cuda

if accelerate_version != expected_accelerate:
    raise SystemExit(
        f"Expected accelerate {expected_accelerate}, but found {accelerate_version}."
    )
if purlin_version != expected_purlin:
    raise SystemExit(
        f"Expected purlin {expected_purlin}, but found {purlin_version}."
    )
if datasets_version != expected_datasets:
    raise SystemExit(
        f"Expected datasets {expected_datasets}, but found {datasets_version}."
    )
if fsspec_version != expected_fsspec:
    raise SystemExit(
        f"Expected fsspec {expected_fsspec}, but found {fsspec_version}."
    )
if not torch_cuda or torch_cuda.split(".", 1)[0] != expected_cuda:
    raise SystemExit(
        f"Expected a CUDA {expected_cuda} PyTorch build, but torch.version.cuda is {torch_cuda!r}."
    )

dataset_probe = Dataset.from_dict({"value": [1]})
if dataset_probe[0]["value"] != 1:
    raise SystemExit("datasets/PyArrow interoperability probe failed.")

print(f"accelerate: {accelerate_version}")
print(
    f"datasets: {datasets_version} "
    f"(fsspec {fsspec_version}, PyArrow {pyarrow.__version__})"
)
print(f"purlin: {purlin_version}")
print(f"torch: {torch.__version__} (CUDA {torch_cuda})")
print(f"sglang: {importlib.metadata.version('sglang')}")
print(f"diffusers: {importlib.metadata.version('diffusers')}")
print(
    "sglang diffusion: "
    f"{FluxPipelineConfig.__name__} ({VaeImageProcessor.__name__})"
)
PY

"${SGLANG_BIN}" version
SGLANG_HELP="$(${SGLANG_BIN} serve --help)"
[[ "${SGLANG_HELP}" == *"--enable-purlin"* ]] || \
    die "sglang serve does not expose --enable-purlin"

log "Installation complete"
printf 'Environment: %s\n' "${VENV_DIR}"
printf 'CUDA toolkit: %s\n' "${CUDA_RELEASE}"
printf 'Purlin: %s\n' "${PURLIN_VERSION}"
printf 'Datasets: %s\n' "${DATASETS_VERSION}"
printf '\nActivate it with:\n  source %q\n' "${VENV_DIR}/bin/activate"
