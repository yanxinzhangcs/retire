#!/usr/bin/env bash
set -Eeuo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
site_packages=${VLLM_SITE_PACKAGES:-$(python3 - <<'PY'
import site
print(site.getsitepackages()[0])
PY
)}
core_path=${site_packages}/vllm/v1/engine/core.py
export VLLM_SITE_PACKAGES=${site_packages}

python3 - <<'PY'
import importlib.metadata
import os

site_packages = os.environ["VLLM_SITE_PACKAGES"]
versions = [
    distribution.version
    for distribution in importlib.metadata.distributions(path=[site_packages])
    if distribution.metadata["Name"].lower() == "vllm"
]
if not versions:
    raise SystemExit(f"vLLM distribution metadata not found under {site_packages}")
version = versions[0]
if not version.startswith("0.13.0"):
    raise SystemExit(f"RETIRE patch requires vLLM 0.13.0, found {version}")
PY

if [[ ! -f ${core_path} ]]; then
    echo "vLLM EngineCore source not found: ${core_path}" >&2
    exit 1
fi

if ! grep -q 'def retire_fence_and_pin' "${core_path}"; then
    patch --forward --batch -p1 -d "${site_packages}" \
        < "${project_root}/patches/vllm-0.13.0-retire-core.patch"
    grep -q 'def retire_fence_and_pin' "${core_path}"
    echo "Applied RETIRE core hooks to ${core_path}"
else
    echo "RETIRE vLLM core patch already applied"
fi

layer_horizon=${RETIRE_LAYER_HORIZON:-0}
if [[ ${layer_horizon} =~ ^[1-9][0-9]*$ ]]; then
    qwen_path=${site_packages}/vllm/model_executor/models/qwen2.py
    if ! grep -q 'worker_layer_safe_point' "${qwen_path}"; then
        patch --forward --batch -p1 -d "${site_packages}" \
            < "${project_root}/patches/vllm-0.13.0-retire-layer-horizon.patch"
        grep -q 'worker_layer_safe_point' "${qwen_path}"
        echo "Applied RETIRE layer horizon (${layer_horizon} layers)"
    else
        echo "RETIRE layer-horizon patch already applied"
    fi
    gpt_oss_path="${site_packages}/vllm/model_executor/models/gpt_oss.py"
    if ! grep -q 'worker_layer_safe_point' "${gpt_oss_path}"; then
        patch --forward --batch -p1 -d "${site_packages}" \
            < "${project_root}/patches/vllm-0.13.0-retire-gpt-oss-layer-horizon.patch"
        grep -q 'worker_layer_safe_point' "${gpt_oss_path}"
        echo "Applied RETIRE GPT-OSS layer horizon"
    else
        echo "RETIRE GPT-OSS layer horizon already applied"
    fi
fi
