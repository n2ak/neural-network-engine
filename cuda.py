
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
#define sum(a,b) a+b

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
int _size(const int *shape,int ndim){{
    int size = 1;
    for(int i = 0;i < ndim; i++){{
        size *= shape[i];
    }}
    return size;
}}
void shapeToDevice(const int *shape,int **d_shape,int ndim){{
    cudaMalloc(d_shape, ndim * sizeof(int));
    cudaMemcpy(*d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);
}}
__device__ int flattenIndex(int ndim,const int* coords,const int* stride){{
    int flat = 0;
    for (int d = ndim - 1; d >= 0; --d) {{
        flat += coords[d] * stride[d];
    }}
    return flat;
}}
__device__ float32 reduce_sum_axes(
    const float32* A,
    const int* axes,
    int ndim_A,
    int nAxis,
    const int* coords_A,
    const int* shape_A,
    const int* stride_A
){{

    float32 acc = 0;
    // Use a local copy of coords_A to avoid modifying shared memory
    int local_coords[MAX_DIMS];
    for (int d = 0; d < ndim_A; ++d) local_coords[d] = coords_A[d];

    // Compute total number of elements to sum over
    int total = 1;
    for (int i = 0; i < nAxis; ++i) {{ total *= shape_A[axes[i]]; }}

    // Iterate over all combinations of axes to sum
    for (int idx = 0; idx < total; ++idx) {{
        int t = idx;
        // Copy the base coordinates for each iteration
        for (int d = 0; d < ndim_A; ++d) local_coords[d] = coords_A[d];
        
        for (int i = nAxis - 1; i >= 0; --i) {{
            int ax = axes[i];
            local_coords[ax] = t % shape_A[ax];
            t /= shape_A[ax];
        }}
        int flat_A = flattenIndex(ndim_A, local_coords, stride_A);
        acc += A[flat_A];
    }}

    return acc;
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


def bin_op_name(name, in_dtype, in_dtype2, out_dtype):
    return f"bin_{name}_{in_dtype}_{in_dtype2}_{out_dtype}"


def elemwise_op_name(name, in_dtype, in_dtype2, out_dtype):
    return f"elemwise_{name}_{in_dtype}_{in_dtype2}_{out_dtype}"


def uop_name(name, in_dtype, out_dtype):
    return f"uop_{name}_{in_dtype}_{out_dtype}"


def recuceop_name(name, in_dtype, out_dtype):
    return f"reduce_{name}_{in_dtype}_{out_dtype}"


def bin_ops_code(name, floating_op, *dtypes: str):
    op = "_" + name

    def bin_op(input_dtype1, input_dtype2):
        out_dtype = promote_dtype(input_dtype1, input_dtype2, floating_op)
        func_name = bin_op_name(name, input_dtype1, input_dtype2, out_dtype)
        kernel_name = f"{func_name}_kernel"
        code = f"""
extern "C" __global__
void {kernel_name}(
    const {input_dtype1}* A, const int* stride_A,
    const {input_dtype2} B,
    {out_dtype}* C, const int* stride_C,
    const int* shape,
    int ndim,
    int totalSize
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS];
    unravel_index(idx, shape, ndim, coords);

    int flat_A = flattenIndex(ndim, coords, stride_A);
    int flat_C = flattenIndex(ndim, coords, stride_C);

    C[flat_C] = {op}(({out_dtype})A[flat_A],({out_dtype})B);
}}
extern "C" void
{func_name}(
    const {input_dtype1} *d_a, const int* stride_A,
    {input_dtype2} d_b,
    {out_dtype} *d_c, const int* stride_C,
    int* shape,
    int ndim
){{
    int totalSize = _size(shape,ndim);

    int *d_shape; shapeToDevice(shape,&d_shape,ndim);
    int *d_stride_A; shapeToDevice(stride_A,&d_stride_A,ndim);
    int *d_stride_C; shapeToDevice(stride_C,&d_stride_C,ndim);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(
        d_a, d_stride_A,
        d_b,
        d_c, d_stride_C,
        d_shape,
        ndim,
        totalSize
    );
}}
    """
        return code
    assert len(dtypes) > 0
    return "\n\n".join([bin_op(d1, d2) for d1 in dtypes for d2 in ["int32", "float32"]])


def recude_code(name, *dtypes: tuple[str, str]):
    op = name

    def recudeop(types: tuple[str, str]):
        input_dtype, out_dtype = types
        func_name = recuceop_name(name, input_dtype, out_dtype)
        kernel_name = f"{func_name}_kernel"
        code = f"""
extern "C" __global__
void {kernel_name}(
    const {input_dtype}* input,
    {out_dtype}* output,
    const int* in_shape,
    const int* in_strides,
    const int* out_shape,
    const int* out_strides,
    const int* axes,
    int nreduce_axes,
    int ndim_in,
    int ndim_out,
    int out_size
){{
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (out_idx >= out_size) return;

    int out_coords[16]; // assume ndim_out <= 16
    int tmp = out_idx;
    for (int d = 0; d < ndim_out; ++d) {{
        out_coords[d] = tmp / out_strides[d];
        tmp %= out_strides[d];
    }}

    int in_coords[16];
    int j = 0;
    for (int d = 0; d < ndim_in; ++d) {{
        bool is_reduced = false;
        for (int r = 0; r < nreduce_axes; ++r) {{
            if (axes[r] == d) {{ is_reduced = true; break; }}
        }}
        if (is_reduced) {{
            in_coords[d] = 0;
        }} else {{
            in_coords[d] = out_coords[j++];
        }}
    }}

    {out_dtype} acc = 0.0f;

    int total_reduce = 1;
    for (int r = 0; r < nreduce_axes; ++r) {{
        total_reduce *= in_shape[axes[r]];
    }}

    for (int r = 0; r < total_reduce; ++r) {{
        int rem = r;
        for (int k = 0; k < nreduce_axes; ++k) {{
            int ax = axes[k];
            int size = in_shape[ax];
            int coord = rem % size;
            rem /= size;
            in_coords[ax] = coord;
        }}

        int in_off = 0;
        for (int d = 0; d < ndim_in; ++d) {{
            in_off += in_coords[d] * in_strides[d];
        }}
        acc = {op}(acc,({out_dtype}) input[in_off]);
    }}

    output[out_idx] = acc;
}}

extern "C" void
{func_name}(
    const {input_dtype}* A, const int* stride_A, const int* shape_A,
    {out_dtype}* C, const int* stride_C, const int* shape_C,
    const int* axis,
    int ndim_A, int ndim_C,
    int nAxis,
    int keepdim
){{
    int totalSize = _size(shape_C, ndim_C);

    int *d_axis; shapeToDevice(axis, &d_axis, ndim_A);

    int *d_shape_A; shapeToDevice(shape_A, &d_shape_A, ndim_A);
    int *d_shape_C; shapeToDevice(shape_C, &d_shape_C, ndim_A);

    int *d_stride_A; shapeToDevice(stride_A, &d_stride_A, ndim_A);
    int *d_stride_C; shapeToDevice(stride_C, &d_stride_C, ndim_A);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(
        A,
        C,
        d_shape_A,
        d_stride_A,
        d_shape_C,
        d_stride_C,
        d_axis,
        nAxis,
        ndim_A,
        ndim_C,
        totalSize
    );
}}
"""
        return code
    assert len(dtypes) > 0
    return "\n\n".join(map(recudeop, dtypes))


def promote_uop_dtype(input_dtype, floating_operation):
    input_dtype = np.dtype(input_dtype)
    if floating_operation:
        out_dtype = "float32"
        if input_dtype == "float64":
            out_dtype = "float64"
    else:
        out_dtype = input_dtype
    return np.dtype(out_dtype)


def uops_code(name, floating_operation: bool, *dtypes: str):
    op = name

    def uop(input_dtype: str):
        out_dtype = promote_uop_dtype(input_dtype, floating_operation)
        func_name = uop_name(name, input_dtype, out_dtype)
        kernel_name = f"{func_name}_kernel"
        code = f"""
extern "C" __global__
void {kernel_name}(
    const {input_dtype}* A, const int* stride_A,
    {out_dtype}* C, const int* stride_C,
    const int* shape,
    int ndim,
    int totalSize
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS]; unravel_index(idx, shape, ndim, coords);

    int flat_A = flattenIndex(ndim, coords, stride_A);
    int flat_C = flattenIndex(ndim, coords, stride_C);

    C[flat_C] = {op}(({out_dtype})A[flat_A]);
}}
extern "C" void
{func_name}(
    const {input_dtype} *d_a, const int* stride_A,
    {out_dtype} *d_c, const int* stride_C,
    int* shape,
    int ndim
){{
    int totalSize = _size(shape,ndim);
    int *d_shape; shapeToDevice(shape, &d_shape, ndim);
    int *d_stride_A; shapeToDevice(stride_A, &d_stride_A, ndim);
    int *d_stride_C; shapeToDevice(stride_C, &d_stride_C, ndim);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(
        d_a, d_stride_A,
        d_c, d_stride_C,
        d_shape,
        ndim,
        totalSize
    );
}}
    """
        return code
    assert len(dtypes) > 0
    return "\n\n".join(map(uop, dtypes))


def elemwise_code(name, floating_op: bool, *dtypes: str):
    op = "_" + name

    def elem_op_code_dype(input_dtype1, input_dtype2):
        out_dtype = promote_dtype(input_dtype1, input_dtype2, floating_op)
        func_name = elemwise_op_name(
            name, input_dtype1, input_dtype2, out_dtype)
        kernel_name = f"{func_name}_kernel"
        code = f"""
extern "C" __global__
void {kernel_name}(
    const {input_dtype1}* A, const int* stride_A,
    const {input_dtype2}* B, const int* stride_B,
    {out_dtype}* C, const int* stride_C,
    const int* shape,
    int ndim,
    int totalSize
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS];
    unravel_index(idx, shape, ndim, coords);

    int flat_A = flattenIndex(ndim, coords, stride_A);
    int flat_B = flattenIndex(ndim, coords, stride_B);
    int flat_C = flattenIndex(ndim, coords, stride_C);


    C[flat_C] = {op}(({out_dtype})A[flat_A] ,({out_dtype}) B[flat_B]);
}}
extern "C" void
{func_name}(
    const {input_dtype1} *d_a, const int* stride_A,
    const {input_dtype2} *d_b, const int* stride_B,
    {out_dtype} *d_c, const int* stride_C,
    int* shape,
    int ndim
){{
    int totalSize = _size(shape,ndim);
    int *d_shape; shapeToDevice(shape,&d_shape,ndim);
    int *d_stride_A; shapeToDevice(stride_A,&d_stride_A,ndim);
    int *d_stride_B; shapeToDevice(stride_B,&d_stride_B,ndim);
    int *d_stride_C; shapeToDevice(stride_C,&d_stride_C,ndim);

    int blockSize = 256;
    int gridSize = (totalSize + blockSize - 1) / blockSize;
    {kernel_name}<<<gridSize, blockSize>>>(
        d_a, d_stride_A,
        d_b, d_stride_B,
        d_c, d_stride_C,
        d_shape,
        ndim,
        totalSize
    );
}}
    """
        return code
    assert len(dtypes) > 0

    return "\n\n".join([elem_op_code_dype(d1, d2)for d1 in dtypes for d2 in dtypes])


def promote_dtype(dtype1, dtype2, floating_op: bool) -> str:
    dt1 = np.dtype(dtype1)
    dt2 = np.dtype(dtype2)
    if np.issubdtype(dt1, np.floating) or np.issubdtype(dt2, np.floating):
        if dt1 == np.float64 or dt2 == np.float64:
            return "float64"
        return "float32"
    if floating_op:
        return "float32"
    if dt1 == np.int64 or dt2 == np.int64:
        return "int64"
    return dtype1 if dt1.itemsize >= dt2.itemsize else dtype2


ELEMWISE_OPS = [
    ("add", False, "float32", "float64", "int32", "int64"),
    ("sub", False, "float32", "float64", "int32", "int64"),
    ("mul", False, "float32", "float64", "int32", "int64"),
    ("div", True, "float32", "float64", "int32", "int64"),
]
BIN_OPS = [
    ("add", False, "float32", "float64", "int32", "int64"),
    ("sub", False, "float32", "float64", "int32", "int64"),
    ("mul", False, "float32", "float64", "int32", "int64"),
    ("div", True, "float32", "float64", "int32", "int64"),
]
UOPS = [
    ("exp", True, "float32", "float64", "int32", "int64"),
    ("log", True, "float32", "float64", "int32", "int64"),
    ("log2", True, "float32", "float64", "int32", "int64"),
]
REDUCE_OPS = [
    ("sum",
        ("float32", "float32"), ("float64", "float64"),
        ("int32", "int64"), ("int64", "int64")
     ),
    ("max",
        ("float32", "float32"), ("float64", "float64"),
        ("int32", "int32"), ("int64", "int64")),
]


def get_cuda_code():
    uops = "\n\n".join(map(lambda v: uops_code(*v), UOPS))
    bin_op = "\n\n".join(map(lambda v: bin_ops_code(*v), BIN_OPS))
    recudeops = "\n\n".join(map(lambda v: recude_code(*v), REDUCE_OPS))
    elemwise_ops = "\n\n".join(map(lambda v: elemwise_code(*v), ELEMWISE_OPS))
    return "\n".join([
        HEADER,
        recudeops,
        elemwise_ops,
        bin_op,
        uops,
    ])


class Buffer:
    def __init__(self, nbytes: int, allocated=False):
        ptr = c_void_p()
        err = CudaAllocator.alloc(byref(ptr), nbytes)
        assert_cuda_error(err)
        self.ptr = ptr
        self.nbytes = nbytes
        self.allocated = allocated
        self.refcount = 1

    def free(self):
        self.refcount -= 1
        if self.refcount <= 0:
            CudaAllocator._free(self.ptr)
            CudaAllocator.mem -= self.nbytes

    def ref(self):
        self.refcount += 1
        return self


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
        buffer = Buffer(np.dtype(dtype).itemsize *
                        np.prod(shape), allocated=True)
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
    def from_cuda(cls, buffer: Buffer, shape, dtype, stride):
        assert buffer.refcount > 0
        assert isinstance(buffer, Buffer)
        assert buffer.allocated

        arr = np.empty(shape, dtype=dtype)

        nbytes = get_nbytes(shape, stride, dtype)
        arr = np.lib.stride_tricks.as_strided(
            arr, shape=shape, strides=[s * arr.itemsize for s in stride]
        )

        cls.synchronize()
        err = cls.memcpy(
            arr.ctypes.data_as(c_void_p),
            buffer.ptr,
            nbytes,
            cls._cudaMemcpyDeviceToHost
        )
        assert_cuda_error(err)
        return arr

    @classmethod
    def synchronize(cls,):
        assert_cuda_error(cls.get_last_error())
        assert_cuda_error(cls._sync())

    @classmethod
    def free(cls, buffer: Buffer):
        buffer.free()


def get_nbytes(shape, stride, dtype):
    itemsize = np.dtype(dtype).itemsize
    size = 1
    for i in range(len(shape)):
        if stride[i] == 0:
            continue
        size *= shape[i]
    return size * itemsize


lib = compile_cuda(get_cuda_code(), "cuda_code")


def define_elemwise_op(name: str):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(),  # a
        c_void_p, _int_1d_array(),  # b
        c_void_p, _int_1d_array(),  # output
        _int_1d_array(),  # shape
        c_int,
    ], None)


def _int_1d_array():
    return np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")


def define_bin_op(name: str, type):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(),  # a , stride of a
        type,  # b
        c_void_p, _int_1d_array(),  # output,stride of output
        _int_1d_array(),  # shape
        c_int,
    ], None)


def define_uop(name: str):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(),  # a
        c_void_p, _int_1d_array(),  # output
        _int_1d_array(),  # shape
        c_int,
    ], None)


def define_reduce_op(name: str):
    return _define_func(lib[name], [
        c_void_p, _int_1d_array(), _int_1d_array(),  # a
        c_void_p, _int_1d_array(), _int_1d_array(),  # output
        _int_1d_array(),
        c_int,
        c_int,
        c_int,
    ], None)


def register_ops():
    ops = {}
    for name, floating_op, *dtypes in ELEMWISE_OPS:
        for in_dtype1 in dtypes:
            for in_dtype2 in dtypes:
                out_dtype = promote_dtype(in_dtype1, in_dtype2, floating_op)
                opname = elemwise_op_name(
                    name, in_dtype1, in_dtype2, out_dtype)
                ops[opname] = define_elemwise_op(opname)

    for name, floating_op, *dtypes in BIN_OPS:
        for in_dtype1 in dtypes:
            for in_dtype2 in ["float32", "int32"]:
                ctype = {
                    "float32": ctypes.c_float,
                    "int32": ctypes.c_int32,
                }[in_dtype2]
                out_dtype = promote_dtype(in_dtype1, in_dtype2, floating_op)
                opname = bin_op_name(name, in_dtype1, in_dtype2, out_dtype)
                ops[opname] = define_bin_op(opname, ctype)

    for name, floating_op, *dtypes in UOPS:
        for in_dtype in dtypes:
            out_dtype = promote_uop_dtype(in_dtype, floating_op)
            opname = uop_name(name, in_dtype, out_dtype)
            ops[opname] = define_uop(opname)
    for name, *dtypes in REDUCE_OPS:
        for (in_dtype, out_dtypes) in dtypes:
            opname = recuceop_name(name, in_dtype, out_dtypes)
            ops[opname] = define_reduce_op(opname)
    return ops


_cuda_ops = register_ops()
