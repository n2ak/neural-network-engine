
template<typename T>
__global__ void set_item_kernel(
    const T* Values, const int* stride_Val,
    const bool* Condition,
    T* Out, const int* shape_Out,const int* stride_Out,
    int ndim,
    int totalSize
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalSize) return;

    int coords[MAX_DIMS];
    unravel_index(idx, shape_Out, ndim, coords);
    int flat_Out = flattenIndex(ndim, coords, stride_Out);
    int flat_Val = flattenIndex(ndim, coords, stride_Val);

    if(Condition[idx] == true) Out[flat_Out] = Values[flat_Val];
}}

template<typename T>
void set_item(
    const T* Values,const int* stride_Val,
    const bool *Condition,
    T* Out, const int* shape_Out,const int* stride_Out,
    int ndim
){{
    int totalSize = _size(shape_Out,ndim);

    int *d_shape_Out; shapeToDevice(shape_Out,&d_shape_Out,ndim);
    int *d_stride_Out; shapeToDevice(stride_Out,&d_stride_Out,ndim);

    int *d_stride_Val; shapeToDevice(stride_Val,&d_stride_Val,ndim);

    int blockSize = BLOCK_SIZE;
    int gridSize = (totalSize + blockSize - 1) / blockSize;

    set_item_kernel<<<gridSize, blockSize>>>(
        Values, d_stride_Val,
        Condition,
        Out, d_shape_Out,d_stride_Out,
        ndim,
        totalSize
    );
}}