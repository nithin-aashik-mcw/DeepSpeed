# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import ctypes
import sys

# Process-wide cache so pin_memory is loaded once even if OpBuilder is imported
# under both ``op_builder`` and ``deepspeed.ops.op_builder`` (separate _loaded_ops).
_SYS_CACHE_ATTR = "_deepspeed_pin_memory_op_module"


def load_pin_memory_module(builder, verbose=False):
    module = getattr(sys, _SYS_CACHE_ATTR, None)
    if module is None:
        # Call OpBuilder.load without going through subclass load() recursion.
        module = super(type(builder), builder).load(verbose=verbose)
        setattr(sys, _SYS_CACHE_ATTR, module)
    so_path = getattr(module, "__file__", None)
    # RTLD_GLOBAL has no Windows equivalent (there is no flat process-wide symbol
    # table); the module is already resident once loaded, which is all the
    # GetProcAddress-based lookup in deepspeed_pin_tensor_client.cpp needs.
    if so_path and sys.platform != "win32":
        ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
    return module
