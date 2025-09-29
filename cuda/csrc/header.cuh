#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdint>

#define BLOCK_SIZE 1024
#define MAX_DIMS 4
#define float32 float
#define float64 double
#define int32 int
#define int64 long long
#define bool int8_t

#define sum(a,b) a+b

#define _add(a,b) a+b
#define _mul(a,b) a*b
#define _sub(a,b) a-b
#define _div(a,b) a/b

__device__ inline void unravel_index(int idx, const int* shape, int ndim, int* coords) {
    for (int d = ndim - 1; d >= 0; --d) {
        coords[d] = idx % shape[d];
        idx /= shape[d];
    }
}

inline int _size(const int *shape,int ndim){
    int size = 1;
    for(int i = 0;i < ndim; i++){
        size *= shape[i];
    }
    return size;
}

void shapeToDevice(const int *shape,int **d_shape,int ndim){
    cudaMalloc(d_shape, ndim * sizeof(int));
    cudaMemcpy(*d_shape, shape, ndim * sizeof(int), cudaMemcpyHostToDevice);
}

__device__ inline int flattenIndex(int ndim,const int* coords,const int* stride){
    int flat = 0;
    for (int d = ndim - 1; d >= 0; --d) {
        flat += coords[d] * stride[d];
    }
    return flat;
}