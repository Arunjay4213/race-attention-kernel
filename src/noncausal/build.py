"""JIT build of the race_noncausal CUDA extension (kernels/ + torch binding).

Usage:
    from build import load_extension
    race = load_extension()
    o = race.forward(q, k, v, W, beta)

For training, use ops.race_attention, which wraps forward_train and backward
in an autograd.Function.

Running this file builds the extension and prints where it was loaded from.
torch caches the build under ~/.cache/torch_extensions and rebuilds when a
source file or a build flag changes. Its version check hashes the listed
sources but not included headers, so a hash of the headers is passed as a
define to make header edits trigger a rebuild too.
"""
from __future__ import annotations

import functools
import hashlib
import pathlib
from types import ModuleType

KERNEL_DIR = pathlib.Path(__file__).resolve().parent / "kernels"
SOURCES = [KERNEL_DIR / "torch_binding.cpp", KERNEL_DIR / "race_fwd.cu", KERNEL_DIR / "race_bwd.cu"]
HEADERS = [
    KERNEL_DIR / "race_fwd.h",
    KERNEL_DIR / "race_bwd.h",
    KERNEL_DIR / "race_common.cuh",
    KERNEL_DIR / "race_internal.cuh",
]

# No explicit -std: torch.utils.cpp_extension adds the standard its own headers need.
CXX_FLAGS = ["-O3"]
CUDA_FLAGS = [
    "-O3",
    "-lineinfo",
    "-Xptxas",
    "-v",
    "-gencode=arch=compute_80,code=sm_80",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
    # PTX for the oldest target so a GPU outside this list can still JIT a kernel image.
    "-gencode=arch=compute_80,code=compute_80",
]


def _headers_digest() -> str:
    digest = hashlib.sha256()
    for header in HEADERS:
        digest.update(header.read_bytes())
    return digest.hexdigest()[:16]


@functools.cache
def load_extension(verbose: bool = True) -> ModuleType:
    """Compiles (if needed) and imports the extension; cached per process."""
    from torch.utils.cpp_extension import load

    headers_define = f"-DRACE_HEADERS_DIGEST={_headers_digest()}"
    return load(
        name="race_noncausal",
        sources=[str(source) for source in SOURCES],
        extra_cflags=CXX_FLAGS + [headers_define],
        extra_cuda_cflags=CUDA_FLAGS + [headers_define],
        extra_include_paths=[str(KERNEL_DIR)],
        verbose=verbose,
    )


if __name__ == "__main__":
    module = load_extension(verbose=True)
    print(f"loaded {module.__name__} from {module.__file__}")
