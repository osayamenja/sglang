#!/usr/bin/env bash

set -Eeuo pipefail

readonly PURLIN_VERSION="0.6.0"
readonly ACCELERATE_VERSION="1.12.0"
readonly DATASETS_VERSION="5.0.1"
readonly FSSPEC_VERSION="2026.6.0"
readonly SGL_DEEP_EP_VERSION="0.1.2"
readonly SGL_DEEP_GEMM_VERSION="0.1.7"
readonly SGL_KERNEL_VERSION="0.4.6.post1"
readonly TORCH_VERSION="2.13.0"
readonly TORCHAUDIO_VERSION="2.11.0"
readonly TORCHVISION_VERSION="0.28.0"
readonly MSCCLPP_VERSION="0.10.0"
readonly MSCCLPP_GIT_TAG="v${MSCCLPP_VERSION}"
readonly TORCHCOMMS_GIT_REPOSITORY="https://github.com/meta-pytorch/torchcomms.git"
readonly TORCHCOMMS_GIT_REVISION="6288fc4d658f2b165623eb649c82149e82d2056b"
readonly PYTHON_VERSION="3.12"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly TORCHCOMMS_NCCLX_PATCH="${SCRIPT_DIR}/patches/torchcomms-ncclx-response-file.patch"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly VENV_DIR="${REPO_ROOT}/.venv"

CURRENT_STEP="startup"
CUDA_OVERRIDE=""
SKIP_SYSTEM_DEPS=0
TORCHCOMMS_CUDA_ARCH_LIST=""
TORCHCOMMS_CMAKE_CUDA_ARCHS=""
TORCHCOMMS_NCCLX_GENCODE=""
TORCHCOMMS_CUDA_IMAGE_LABEL=""
TORCHCOMMS_SOURCE_DIR=""
TORCHCOMMS_CUDA_IMAGES=()

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
Install this SGLang fork and its communication backends into the repository's
.venv.

Usage:
  bash scripts/install_purlin.sh [options]

Options:
  --cuda 12|13          Require CUDA 12.9 or a CUDA 13 toolkit.
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
    command -v git >/dev/null 2>&1 || missing+=(git)
    command -v ninja >/dev/null 2>&1 || missing+=(ninja-build)
    command -v protoc >/dev/null 2>&1 || missing+=(protobuf-compiler)
    if ! dpkg-query -W -f='${Status}' libnuma-dev 2>/dev/null | \
        grep -q '^install ok installed$'; then
        missing+=(libnuma-dev)
    fi

    local package
    for package in \
        autoconf automake binutils bison flex git libibverbs-dev libnl-3-dev \
        libnl-route-3-dev libnuma-dev libtool libunwind-dev make patch perl \
        pkg-config; do
        if ! dpkg-query -W -f='${Status}' "${package}" 2>/dev/null | \
            grep -q '^install ok installed$'; then
            missing+=("${package}")
        fi
    done

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

for command in autoconf automake bison cmake curl flex git libtoolize make \
    ninja patch perl pkg-config protoc readelf; do
    command -v "${command}" >/dev/null 2>&1 || \
        die "${command} is required; install the system dependencies or omit --skip-system-deps"
done
if ! command -v cc >/dev/null 2>&1 || ! command -v c++ >/dev/null 2>&1; then
    die "C and C++ compilers are required; install build-essential"
fi

CURRENT_STEP="CUDA toolkit detection"
log "Detecting the CUDA toolkit"
command -v nvcc >/dev/null 2>&1 || \
    die "nvcc was not found in PATH; install or select CUDA 12.9 or CUDA 13"
NVCC_OUTPUT="$(nvcc --version)"
CUDA_RELEASE="$(sed -n 's/.*release \([0-9][0-9]*\)\.\([0-9][0-9]*\).*/\1.\2/p' <<<"${NVCC_OUTPUT}" | head -n1)"
[[ -n "${CUDA_RELEASE}" ]] || die "could not parse the CUDA version from nvcc --version"
CUDA_MAJOR="${CUDA_RELEASE%%.*}"
case "${CUDA_MAJOR}" in
    12 | 13) ;;
    *) die "CUDA ${CUDA_RELEASE} is unsupported; CUDA 12.9 or CUDA 13 is required" ;;
esac
CUDA_MINOR="${CUDA_RELEASE#*.}"
if [[ "${CUDA_MAJOR}" == "12" ]] && ((10#${CUDA_MINOR} < 9)); then
    die "CUDA ${CUDA_RELEASE} is unsupported; CUDA 12.9 or CUDA 13 is required"
fi
if [[ -n "${CUDA_OVERRIDE}" && "${CUDA_OVERRIDE}" != "${CUDA_MAJOR}" ]]; then
    die "--cuda ${CUDA_OVERRIDE} does not match nvcc (${CUDA_RELEASE}); select the intended nvcc through PATH"
fi
echo "Detected CUDA ${CUDA_RELEASE} from $(command -v nvcc)."

detect_torchcomms_architectures() {
    local architecture_suffix
    local cache_key
    local capability
    local compute_code
    local image_csv
    local query_output
    local -A seen_capabilities=()
    local -a cache_architectures=()
    local -a cmake_architectures=()
    local -a gencode_flags=()
    local -a torch_architectures=()

    command -v nvidia-smi >/dev/null 2>&1 || \
        die "nvidia-smi is required to detect the TorchComms build architecture"
    query_output="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader)" || \
        die "nvidia-smi could not query GPU compute capabilities"

    while IFS= read -r capability; do
        capability="${capability//[[:space:]]/}"
        [[ -n "${capability}" ]] || continue
        [[ "${capability}" =~ ^([0-9]+)\.([0-9]+)$ ]] || \
            die "nvidia-smi returned an invalid compute capability: ${capability}"
        [[ -z "${seen_capabilities[${capability}]:-}" ]] || continue
        seen_capabilities["${capability}"]=1

        compute_code="${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
        architecture_suffix=""
        case "${capability}" in
            10.0 | 10.3) architecture_suffix="a" ;;
        esac

        torch_architectures+=("${capability}${architecture_suffix}")
        cmake_architectures+=("${compute_code}${architecture_suffix}")
        TORCHCOMMS_CUDA_IMAGES+=("sm_${compute_code}${architecture_suffix}")
        cache_architectures+=("sm${compute_code}${architecture_suffix}")
        gencode_flags+=(
            "-gencode=arch=compute_${compute_code}${architecture_suffix},code=sm_${compute_code}${architecture_suffix}"
        )
    done <<<"${query_output}"

    ((${#TORCHCOMMS_CUDA_IMAGES[@]} > 0)) || \
        die "no NVIDIA GPUs were found for the TorchComms build"
    TORCHCOMMS_CUDA_ARCH_LIST="$(IFS=';'; printf '%s' "${torch_architectures[*]}")"
    TORCHCOMMS_CMAKE_CUDA_ARCHS="$(IFS=';'; printf '%s' "${cmake_architectures[*]}")"
    TORCHCOMMS_NCCLX_GENCODE="${gencode_flags[*]}"
    image_csv="$(IFS=,; printf '%s' "${TORCHCOMMS_CUDA_IMAGES[*]}")"
    TORCHCOMMS_CUDA_IMAGE_LABEL="${image_csv//,/, }"
    cache_key="$(IFS=-; printf '%s' "${cache_architectures[*]}")"
    TORCHCOMMS_SOURCE_DIR="${REPO_ROOT}/.cache/torchcomms-${TORCHCOMMS_GIT_REVISION}-${cache_key}"
}

CURRENT_STEP="GPU architecture detection"
log "Detecting TorchComms build architectures"
detect_torchcomms_architectures
readonly TORCHCOMMS_CUDA_ARCH_LIST
readonly TORCHCOMMS_CMAKE_CUDA_ARCHS
readonly TORCHCOMMS_NCCLX_GENCODE
readonly TORCHCOMMS_CUDA_IMAGE_LABEL
readonly TORCHCOMMS_SOURCE_DIR
readonly -a TORCHCOMMS_CUDA_IMAGES
echo "Detected TorchComms CUDA images: ${TORCHCOMMS_CUDA_IMAGE_LABEL}"

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

CURRENT_STEP="SGLang build dependencies"
log "Installing SGLang build dependencies"
"${UV_PIP[@]}" \
    "setuptools>=61.0" "setuptools-rust>=1.11" "setuptools-scm>=8.0" wheel

install_cuda12_dependencies() {
    local machine_arch
    local -a cuda13_packages=()
    local -a sglang_requirements=()

    machine_arch="$(uname -m)"

    # Install CUDA 12.9 variants first. Their local-version suffixes satisfy
    # the corresponding public-version pins in pyproject.toml.
    "${UV_PIP[@]}" --reinstall \
        --default-index https://download.pytorch.org/whl/cu129 \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCHAUDIO_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}"
    "${UV_PIP[@]}" --reinstall --no-deps \
        "https://github.com/sgl-project/whl/releases/download/v${SGL_KERNEL_VERSION}/sglang_kernel-${SGL_KERNEL_VERSION}+cu129-cp310-abi3-manylinux2014_${machine_arch}.whl"
    "${UV_PIP[@]}" --reinstall --no-deps \
        "https://github.com/sgl-project/whl/releases/download/v${SGL_DEEP_GEMM_VERSION}/sgl_deep_gemm-${SGL_DEEP_GEMM_VERSION}+cu129-py3-none-manylinux2014_${machine_arch}.whl"
    "${UV_PIP[@]}" --reinstall --no-deps \
        --default-index https://docs.sglang.ai/whl/cu129/ \
        "sgl-deep-ep==${SGL_DEEP_EP_VERSION}+cu129"

    # Resolve the current upstream dependency set after changing only the
    # three CUDA-major-specific requirements. Keeping this transformation in
    # memory avoids modifying the checkout during installation.
    mapfile -t sglang_requirements < <(
        PYPROJECT_PATH="${REPO_ROOT}/python/pyproject.toml" \
            "${PYTHON_BIN}" <<'PY'
import os
import tomllib

with open(os.environ["PYPROJECT_PATH"], "rb") as file:
    project = tomllib.load(file)["project"]

requirements = project["dependencies"] + project["optional-dependencies"]["diffusion"]
for requirement in requirements:
    requirement = requirement.replace("cuda-python>=13.0", "cuda-python>=12.9,<13.0")
    requirement = requirement.replace("flashinfer_python[cu13]", "flashinfer_python[cu12]")
    requirement = requirement.replace("nvidia-cutlass-dsl[cu13]", "nvidia-cutlass-dsl")
    print(requirement)
PY
    )
    ((${#sglang_requirements[@]} > 0)) || die "could not read SGLang dependencies"
    "${UV_PIP[@]}" --extra-index-url https://download.pytorch.org/whl/cu129 \
        "${sglang_requirements[@]}"

    # Some extras publish helper distributions with a -cu13 suffix. They are
    # not usable in a CUDA 12 environment and mirror upstream's Docker cleanup.
    mapfile -t cuda13_packages < <(
        "${PYTHON_BIN}" <<'PY'
import importlib.metadata

for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name", "")
    if name.lower().endswith("-cu13"):
        print(name)
PY
    )
    if ((${#cuda13_packages[@]} > 0)); then
        "${UV_BIN}" pip uninstall --python "${PYTHON_BIN}" \
            "${cuda13_packages[@]}"
    fi

    CARGO_TARGET_DIR="${VENV_DIR}/.cache/sglang-rust-target" \
        "${UV_PIP[@]}" --no-build-isolation --no-deps --editable \
        "${REPO_ROOT}/python[diffusion]"
}

CURRENT_STEP="SGLang CUDA ${CUDA_MAJOR} installation"
log "Installing this SGLang checkout for CUDA ${CUDA_MAJOR}"
if [[ "${CUDA_MAJOR}" == "12" ]]; then
    install_cuda12_dependencies
else
    CARGO_TARGET_DIR="${VENV_DIR}/.cache/sglang-rust-target" \
        "${UV_PIP[@]}" --no-build-isolation --editable \
        "${REPO_ROOT}/python[diffusion]"
fi

CURRENT_STEP="Hugging Face datasets compatibility"
log "Installing datasets ${DATASETS_VERSION} with compatible fsspec ${FSSPEC_VERSION}"
# Installing SGLang into Purlin's existing environment can otherwise preserve a
# newer fsspec and backtrack to datasets 1.1.1. That release cannot import with
# current PyArrow because it still references the removed PyExtensionType API.
"${UV_PIP[@]}" \
    "datasets==${DATASETS_VERSION}" \
    "fsspec==${FSSPEC_VERSION}"

prepare_torchcomms_source() {
    local actual_revision

    mkdir -p "$(dirname "${TORCHCOMMS_SOURCE_DIR}")"
    if [[ -d "${TORCHCOMMS_SOURCE_DIR}/.git" ]]; then
        actual_revision="$(git -C "${TORCHCOMMS_SOURCE_DIR}" rev-parse HEAD)"
        [[ "${actual_revision}" == "${TORCHCOMMS_GIT_REVISION}" ]] || \
            die "${TORCHCOMMS_SOURCE_DIR} is at ${actual_revision}, expected ${TORCHCOMMS_GIT_REVISION}"
        return
    fi
    [[ ! -e "${TORCHCOMMS_SOURCE_DIR}" ]] || \
        die "${TORCHCOMMS_SOURCE_DIR} exists but is not a TorchComms git checkout"

    git init "${TORCHCOMMS_SOURCE_DIR}"
    git -C "${TORCHCOMMS_SOURCE_DIR}" remote add origin \
        "${TORCHCOMMS_GIT_REPOSITORY}"
    git -C "${TORCHCOMMS_SOURCE_DIR}" fetch --depth 1 origin \
        "${TORCHCOMMS_GIT_REVISION}"
    git -C "${TORCHCOMMS_SOURCE_DIR}" checkout --detach FETCH_HEAD
}

apply_torchcomms_patches() {
    [[ -f "${TORCHCOMMS_NCCLX_PATCH}" ]] || \
        die "TorchComms NCCLX patch is missing: ${TORCHCOMMS_NCCLX_PATCH}"

    if git -C "${TORCHCOMMS_SOURCE_DIR}" apply --check \
        "${TORCHCOMMS_NCCLX_PATCH}" 2>/dev/null; then
        git -C "${TORCHCOMMS_SOURCE_DIR}" apply "${TORCHCOMMS_NCCLX_PATCH}"
        echo "Applied the NCCLX response-file patch."
    elif git -C "${TORCHCOMMS_SOURCE_DIR}" apply --reverse --check \
        "${TORCHCOMMS_NCCLX_PATCH}" 2>/dev/null; then
        echo "The NCCLX response-file patch is already applied."
    else
        die "the NCCLX response-file patch does not apply to ${TORCHCOMMS_GIT_REVISION}"
    fi
}

CURRENT_STEP="TorchComms source checkout"
log "Preparing TorchComms ${TORCHCOMMS_GIT_REVISION}"
prepare_torchcomms_source
apply_torchcomms_patches

CURRENT_STEP="TorchComms NCCLX source build"
log "Building TorchComms/NCCLX for ${TORCHCOMMS_CUDA_IMAGE_LABEL}"
"${UV_PIP[@]}" packaging pyyaml setuptools wheel
CUDA_HOME="$(cd "$(dirname "$(readlink -f "$(command -v nvcc)")")/.." && pwd)"
readonly CUDA_HOME
TORCHCOMMS_THIRD_PARTY_DIR="/tmp/third-party"
TORCHCOMMS_THIRD_PARTY_WAS_PRESENT=0
if [[ -e "${TORCHCOMMS_THIRD_PARTY_DIR}" ]]; then
    TORCHCOMMS_THIRD_PARTY_WAS_PRESENT=1
fi
cleanup_torchcomms_third_party() {
    if ((TORCHCOMMS_THIRD_PARTY_WAS_PRESENT == 0)) && \
        [[ -d "${TORCHCOMMS_THIRD_PARTY_DIR}" ]]; then
        find "${TORCHCOMMS_THIRD_PARTY_DIR}" -depth -delete
    fi
}
trap cleanup_torchcomms_third_party EXIT
(
    export CMAKE_BUILD_PARALLEL_LEVEL="${TORCHCOMMS_BUILD_JOBS:-$(nproc)}"
    export CMAKE_BUILD_TYPE=Release
    export CUDA_HOME
    export CUDAARCHS="${TORCHCOMMS_CMAKE_CUDA_ARCHS}"
    export NCCL_BUILD_JOBS="${TORCHCOMMS_BUILD_JOBS:-$(nproc)}"
    export NCCL_SKIP_CONDA_INSTALL=1
    export NVCC_GENCODE="${TORCHCOMMS_NCCLX_GENCODE}"
    export TORCH_CUDA_ARCH_LIST="${TORCHCOMMS_CUDA_ARCH_LIST}"
    export USE_GLOO=OFF
    export USE_NCCL=OFF
    export USE_NCCLX=ON
    export USE_TRANSPORT=OFF
    export USE_TRANSPORT_CCA_HOOK=OFF
    export USE_TRITON=OFF
    "${UV_PIP[@]}" --reinstall --no-build-isolation --no-deps \
        "${TORCHCOMMS_SOURCE_DIR}"
)
cleanup_torchcomms_third_party
trap - EXIT

CURRENT_STEP="MSCCL++ installation"
log "Installing MSCCL++ ${MSCCLPP_VERSION}"
MSCCLPP_SOURCE_DIR="$(mktemp -d -t sglang-mscclpp.XXXXXX)"
cleanup_mscclpp_source() {
    if [[ -n "${MSCCLPP_SOURCE_DIR:-}" && -d "${MSCCLPP_SOURCE_DIR}" ]]; then
        rm -rf -- "${MSCCLPP_SOURCE_DIR}"
    fi
}
trap cleanup_mscclpp_source EXIT
git clone --depth 1 --branch "${MSCCLPP_GIT_TAG}" \
    https://github.com/microsoft/mscclpp.git "${MSCCLPP_SOURCE_DIR}"
"${UV_PIP[@]}" -Ccmake.define.MSCCLPP_USE_IB=OFF \
    "${MSCCLPP_SOURCE_DIR}[cuda${CUDA_MAJOR}]"
cleanup_mscclpp_source
trap - EXIT

CURRENT_STEP="installation verification"
log "Verifying the installation"
EXPECTED_ACCELERATE_VERSION="${ACCELERATE_VERSION}" \
EXPECTED_CUDA_MAJOR="${CUDA_MAJOR}" EXPECTED_DATASETS_VERSION="${DATASETS_VERSION}" \
EXPECTED_FSSPEC_VERSION="${FSSPEC_VERSION}" EXPECTED_PURLIN_VERSION="${PURLIN_VERSION}" \
EXPECTED_MSCCLPP_VERSION="${MSCCLPP_VERSION}" \
    "${PYTHON_BIN}" <<'PY'
import importlib.metadata
import os

import accelerate
import datasets
import fsspec
import mscclpp
import purlin
import pyarrow
import sglang
import torch
import torchcomms
from datasets import Dataset
from diffusers.image_processor import VaeImageProcessor
from sglang.multimodal_gen.configs.pipeline_configs.flux import FluxPipelineConfig

expected_accelerate = os.environ["EXPECTED_ACCELERATE_VERSION"]
expected_cuda = os.environ["EXPECTED_CUDA_MAJOR"]
expected_datasets = os.environ["EXPECTED_DATASETS_VERSION"]
expected_fsspec = os.environ["EXPECTED_FSSPEC_VERSION"]
expected_mscclpp = os.environ["EXPECTED_MSCCLPP_VERSION"]
expected_purlin = os.environ["EXPECTED_PURLIN_VERSION"]
accelerate_version = importlib.metadata.version("accelerate")
datasets_version = importlib.metadata.version("datasets")
fsspec_version = importlib.metadata.version("fsspec")
mscclpp_version = importlib.metadata.version("mscclpp")
purlin_version = importlib.metadata.version("purlin")
torch_cuda = torch.version.cuda

if accelerate_version != expected_accelerate:
    raise SystemExit(
        f"Expected accelerate {expected_accelerate}, but found {accelerate_version}."
    )
if mscclpp_version != expected_mscclpp:
    raise SystemExit(
        f"Expected mscclpp {expected_mscclpp}, but found {mscclpp_version}."
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
if not torchcomms.is_backend_built("ncclx"):
    raise SystemExit("TorchComms was not built with its NCCLX backend.")

dataset_probe = Dataset.from_dict({"value": [1]})
if dataset_probe[0]["value"] != 1:
    raise SystemExit("datasets/PyArrow interoperability probe failed.")

print(f"accelerate: {accelerate_version}")
print(
    f"datasets: {datasets_version} "
    f"(fsspec {fsspec_version}, PyArrow {pyarrow.__version__})"
)
print(f"mscclpp: {mscclpp_version}")
print(f"purlin: {purlin_version}")
print(f"torch: {torch.__version__} (CUDA {torch_cuda})")
print(
    "torchcomms: "
    f"{importlib.metadata.version('torchcomms')} "
    f"(backends: {', '.join(torchcomms.built_backends())})"
)
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
[[ "${SGLANG_HELP}" == *"--enable-torchcomms"* ]] || \
    die "sglang serve does not expose --enable-torchcomms"
[[ "${SGLANG_HELP}" == *"--enable-mscclpp"* ]] || \
    die "sglang serve does not expose --enable-mscclpp"

CURRENT_STEP="TorchComms CUDA image verification"
log "Verifying TorchComms CUDA images"
mapfile -t TORCHCOMMS_LIBRARIES < <(
    "${PYTHON_BIN}" - <<'PY'
from pathlib import Path

import torchcomms

for library in sorted(Path(torchcomms.__file__).parent.glob("*.so")):
    print(library)
PY
)
((${#TORCHCOMMS_LIBRARIES[@]} > 0)) || \
    die "could not find the installed TorchComms shared libraries"
CUOBJDUMP="${CUDA_HOME}/bin/cuobjdump"
[[ -x "${CUOBJDUMP}" ]] || die "cuobjdump was not found at ${CUOBJDUMP}"
TORCHCOMMS_CUOBJDUMP_OUTPUT="$(${CUOBJDUMP} --list-elf "${TORCHCOMMS_LIBRARIES[@]}")"
for architecture in "${TORCHCOMMS_CUDA_IMAGES[@]}"; do
    grep -q "${architecture}" <<<"${TORCHCOMMS_CUOBJDUMP_OUTPUT}" || \
        die "TorchComms is missing the required ${architecture} CUDA image"
done
echo "Verified TorchComms CUDA images: ${TORCHCOMMS_CUDA_IMAGE_LABEL}"

log "Installation complete"
printf 'Environment: %s\n' "${VENV_DIR}"
printf 'CUDA toolkit: %s\n' "${CUDA_RELEASE}"
printf 'MSCCL++: %s\n' "${MSCCLPP_VERSION}"
printf 'Purlin: %s\n' "${PURLIN_VERSION}"
printf 'TorchComms revision: %s\n' "${TORCHCOMMS_GIT_REVISION}"
printf 'TorchComms CUDA images: %s\n' "${TORCHCOMMS_CUDA_IMAGE_LABEL}"
printf 'Datasets: %s\n' "${DATASETS_VERSION}"
printf '\nActivate it with:\n  source %q\n' "${VENV_DIR}/bin/activate"
