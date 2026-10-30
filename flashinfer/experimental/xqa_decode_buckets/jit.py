# SPDX-License-Identifier: Apache-2.0
"""JIT-only task specialization of the native XQA source."""

import functools
from pathlib import Path

import torch

from ...compilation_context import CompilationContext
from ...jit import env as jit_env
from ...jit.core import gen_jit_spec
from ...jit.xqa import xqa_nvcc_flags


def gen_module(persistent: bool, *, direct: bool = False):
    if direct:
        persistent = False
    source = Path(__file__).resolve().parent / "csrc"
    flags = CompilationContext().get_nvcc_flags_list(supported_major_versions=[12])
    return gen_jit_spec(
        f"experimental_xqa_decode_buckets_q6_kv1_d256_p32_persistent_{int(persistent)}_direct_{int(direct)}",
        [source / "binding.cu"],
        extra_cuda_cflags=xqa_nvcc_flags
        + flags
        + [
            "--expt-relaxed-constexpr",
            "-Xptxas=-v",
            "-DHEAD_ELEMS=256",
            "-DHEAD_GRP_SIZE=6",
            "-DINPUT_FP16=0",
            "-DCACHE_ELEM_ENUM=3",
            "-DTOKENS_PER_PAGE=32",
            "-DSLIDING_WINDOW=0",
            "-DLOW_PREC_OUTPUT=0",
            "-DSPEC_DEC=0",
            "-DFLASHINFER_XQA_TASKS=1",
            f"-DFLASHINFER_XQA_PERSISTENT_TASKS={int(persistent)}",
            f"-DFLASHINFER_XQA_DIRECT_TASKS={int(direct)}",
        ],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR / "xqa"],
        extra_ldflags=["-lcuda"],
    )


@functools.cache
def get_module(persistent: bool, device: torch.device, *, direct: bool = False):
    # Configure dynamic shared memory outside capture, on each owning device.
    with torch.cuda.device(device):
        module = gen_module(persistent, direct=direct).build_and_load()
        resources = torch.empty(4, dtype=torch.int64)
        module.configure(resources)
    return module, tuple(resources.tolist())
