#include <Engine.hpp>

__global__ void Engine::gaussianBlur(float * imageGPU, float* imageCPU, unsigned int height, unsigned int width)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(ix < width && iy < height)
    {
        
    }
}

__global__ void Engine::brightness(float * imageGPU, float* imageCPU, unsigned int height, unsigned int width, int brightness)
{

}