#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

#define N 3

#define gpuErrchk(ans) { gpuAssert((ans),__FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort=true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

typedef enum
{
    BLUE,
    GREEN,
    RED
} color;

const float gaussMat[N][N] = 
{
    
};

const int horizontalEdgeMask[N][N] = 
{

};

const int verticalEdgeMask[N][N] = 
{

};