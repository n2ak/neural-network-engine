
import subprocess
import ctypes
import tempfile
import os
import numpy as np
from ctypes import c_int, c_float, c_void_p, POINTER as P, c_size_t, byref

MAX_DIMS = 4
HEADER = f"""
#include <cuda_runtime.h>
#include <stdio.h>

#define MAX_DIMS {MAX_DIMS}
#define float32 float
#define float64 double
#define int32 int 
#define int64 long long 
#define _add(a,b) a+b
#define _mul(a,b) a*b
#define _sub(a,b) a-b
#define _div(a,b) a/b

__device__ void unravel_index(int idx, const int* shape, int ndim, int* coords) {{
    for (int d = ndim - 1; d >= 0; --d) {{
        coords[d] = idx % shape[d];
        idx /= shape[d];
    }}
}}
int _size(int *shape,int ndim){{
    int size = 1;
    for(int i = 0;i < ndim; i++){{
        size *= shape[i];
    }}
    return size;
}}

"""


def compile_cuda(code: str, lib_name="libtemp.so"):
    tmpdir = tempfile.mkdtemp()
    cu_path = os.path.join(tmpdir, "kernel.cu")
    so_path = os.path.join(tmpdir, lib_name)
    with open(cu_path, "w") as f:
        f.write(code)

    cmd = [
        nvcc_path, "-shared", "-Xcompiler", "-fPIC", cu_path,
        "-o", so_path,
        "-lcudart",
        "-arch=sm_86",
    ]
    subprocess.check_call(cmd)
    lib = ctypes.CDLL(so_path, mode=RTLD_GLOBAL)
    return lib


def assert_cuda_error(err):
    assert err == 0, err


def _define_func(func, types, ret=None):
    func.argtypes = types
    func.restype = ret
    return func


# for the compiling and running? runtimes to be the same
RTLD_GLOBAL = ctypes.RTLD_GLOBAL

nvcc_path = "/usr/local/cuda-12.8/bin/nvcc"
_cuda = ctypes.CDLL("libcudart.so", mode=RTLD_GLOBAL)


def bin_op_name(name, dtype):
    return f"bin_{name}_{dtype}"


def elemwise_op_name(name, dtype):
    return f"elemwise_{name}_{dtype}"


def uop_name(name, dtype):
    return f"uop_{name}_{dtype}"


def bin_ops_code(name, *dtypes: str):
    op = "_" + name

    def bin_op(dtype: str):
        func_name = bin_op_name(name, dtype)
        kernel_name = f"{func_name}_kernel"
        # print("code for", f"bin: {func_name=} {kernel_name}")
        code = f"""
extern "C" __global__
void {kernel_name}(const {dtype}* A, const {dtype} B, {dtype}* C,const int* shape, int ndim, int totalSize) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS];
    unravel_index(idx, shape, ndim, coords);

    // Example use: convert back to flat index (row-major)
    int stride = 1, flat = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        flat += coords[d] * stride;
        stride *= shape[d];
    }}

    C[flat] = {op}(A[flat], B);
}}
extern "C" void 
{func_name}(const {dtype} *d_a, {dtype} d_b,{dtype} *d_c, int* shape,int ndim){{
    int totalSize = _size(shape,ndim);
    int *d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(int));
    cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(d_a, d_b, d_c, d_shape, ndim, totalSize);
}}
    """
        return code
    assert len(dtypes) > 0
    return "\n\n".join(map(bin_op, dtypes))


def uops_code(name, *dtypes: str):
    op = name

    def uop(dtype: str):
        func_name = uop_name(name, dtype)
        kernel_name = f"{func_name}_kernel"
        # print("code for", f"uop: {func_name=} {kernel_name}")
        code = f"""
extern "C" __global__
void {kernel_name}(const {dtype}* A, {dtype}* C,const int* shape, int ndim, int totalSize) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS];
    unravel_index(idx, shape, ndim, coords);

    // Example use: convert back to flat index (row-major)
    int stride = 1, flat = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        flat += coords[d] * stride;
        stride *= shape[d];
    }}

    C[flat] = {op}(A[flat]);
}}
extern "C" void 
{func_name}(const {dtype} *d_a, {dtype} *d_c, int* shape,int ndim){{
    int totalSize = _size(shape,ndim);
    int *d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(int));
    cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(d_a, d_c, d_shape, ndim, totalSize);
}}
    """
        return code
    assert len(dtypes) > 0
    return "\n\n".join(map(uop, dtypes))


def elemwise_code(name, *dtypes: str):
    op = "_" + name

    def elem_op_code_dype(dtype: str):
        func_name = elemwise_op_name(name, dtype)
        kernel_name = f"{func_name}_kernel"
        # print("code for", f"ew: {func_name=} {kernel_name}")
        code = f"""
extern "C" __global__
void {kernel_name}(const {dtype}* A, const {dtype}* B, {dtype}* C,const int* shape, int ndim, int totalSize) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS];
    unravel_index(idx, shape, ndim, coords);

    // Example use: convert back to flat index (row-major)
    int stride = 1, flat = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        flat += coords[d] * stride;
        stride *= shape[d];
    }}

    C[flat] = {op}(A[flat] , B[flat]);
}}
extern "C" void 
{func_name}(const {dtype} *d_a,const {dtype} *d_b,{dtype} *d_c,int* shape,int ndim){{
    int totalSize = _size(shape,ndim);
    int *d_shape;
    cudaMalloc(&d_shape, ndim * sizeof(int));
    cudaMemcpy(d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(d_a, d_b, d_c, d_shape, ndim, totalSize);
}}
    """
        return code
    assert len(dtypes) > 0
    return "\n\n".join(map(elem_op_code_dype, dtypes))


ELEMWISE_OPS = [
    ("add", "float32", "int32", "float64", "int64"),
    ("mul", "float32", "int32", "float64", "int64"),
    ("sub", "float32", "int32", "float64", "int64"),
    ("div", "float32", "int32", "float64", "int64"),
]
BIN_OPS = [
    ("add", "float32", "float64", "int32", "int64"),
    ("mul", "float32", "float64", "int32", "int64"),
    ("sub", "float32", "float64", "int32", "int64"),
    ("div", "float32", "float64", "int32", "int64"),
]
UOPS = [
    ("exp", "float32", "float64",),
    ("log", "float32", "float64",),
    ("log2", "float32", "float64",),
]


def get_cuda_code():
    uops = "\n\n".join(
        map(lambda v: uops_code(*v), UOPS))
    bin_op = "\n\n".join(
        map(lambda v: bin_ops_code(*v), BIN_OPS))
    elemwise_ops = "\n\n".join(
        map(lambda v: elemwise_code(*v), ELEMWISE_OPS))
    return "\n".join([
        HEADER,
        elemwise_ops,
        bin_op,
        uops
    ])


class Buffer:
    def __init__(self, nbytes: int):
        ptr = c_void_p()
        err = CudaAllocator.alloc(byref(ptr), nbytes)
        assert_cuda_error(err)
        self.ptr = ptr
        self.nbytes = nbytes


class CudaAllocator:
    _cudaMemcpyHostToDevice = 1
    _cudaMemcpyDeviceToHost = 2
    alloc = _define_func(_cuda.cudaMalloc, [P(c_void_p), c_size_t], c_int)
    _free = _define_func(_cuda.cudaFree, [c_void_p], c_int)
    memcpy = _define_func(
        _cuda.cudaMemcpy, [c_void_p, c_void_p, c_size_t, c_int], c_int)
    get_last_error = _define_func(_cuda.cudaGetLastError, [], c_int)
    _sync = _define_func(_cuda.cudaDeviceSynchronize, [], c_int)

    mem = 0

    @classmethod
    def alloc_empty(cls, shape, dtype: np.typing.DTypeLike) -> Buffer:
        buffer = Buffer(np.dtype(dtype).itemsize * np.prod(shape))
        # print(f"Allocated {buffer.nbytes} bytes")
        cls.mem += buffer.nbytes
        return buffer

    @classmethod
    def to_cuda(cls, host_array: np.typing.NDArray):
        buffer = cls.alloc_empty(host_array.shape, host_array.dtype)
        err = cls.memcpy(
            buffer.ptr,
            host_array.ctypes.data_as(c_void_p),
            host_array.nbytes,
            cls._cudaMemcpyHostToDevice
        )
        assert_cuda_error(err)
        return buffer

    @classmethod
    def from_cuda(cls, buffer: Buffer, shape, dtype):
        host_out = np.empty(shape, dtype=dtype)
        cls.synchronize()
        err = cls.memcpy(
            host_out.ctypes.data_as(c_void_p),
            buffer.ptr,
            host_out.nbytes,
            cls._cudaMemcpyDeviceToHost
        )
        assert_cuda_error(err)
        return host_out

    @classmethod
    def synchronize(cls,):
        assert_cuda_error(cls.get_last_error())
        assert_cuda_error(cls._sync())

    @classmethod
    def free(cls, buffer: Buffer):
        cls._free(buffer.ptr)
        cls.mem -= buffer.nbytes
        # print(f"Freed {buffer.nbytes} bytes")
        del buffer


lib = compile_cuda(get_cuda_code(), "cuda_code")


def define_elemwise_op(name: str):
    return _define_func(lib[name], [
        c_void_p,  # a
        c_void_p,  # b
        c_void_p,  # output
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                               flags="C_CONTIGUOUS"),  # shape
    ], None)


def define_bin_op(name: str, type):
    return _define_func(lib[name], [
        c_void_p,  # a
        type,  # b
        c_void_p,  # output
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                               flags="C_CONTIGUOUS"),  # shape
    ], None)


def define_uop(name: str):
    return _define_func(lib[name], [
        c_void_p,  # a
        c_void_p,  # output
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1,
                               flags="C_CONTIGUOUS"),  # shape
    ], None)


_cuda_ops = {}
for name, *dtypes in ELEMWISE_OPS:
    for dtype in dtypes:
        opname = elemwise_op_name(name, dtype)
        _cuda_ops[opname] = define_elemwise_op(opname)

for name, *dtypes in BIN_OPS:
    for dtype in dtypes:
        ctype = {
            "float32": ctypes.c_float,
            "float64": ctypes.c_double,
            "int32": ctypes.c_int32,
            "int64": ctypes.c_longlong,
        }[dtype]
        opname = bin_op_name(name, dtype)
        _cuda_ops[opname] = define_bin_op(opname, ctype)

for name, *dtypes in UOPS:
    for dtype in dtypes:
        opname = uop_name(name, dtype)
        _cuda_ops[opname] = define_uop(opname)
