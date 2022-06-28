#include <ImageProcessing.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void gaussianBlurKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width, unsigned int channel, unsigned int step)
{
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;

    if(iy < 2 || ix < 2 || ix >= width-3 || iy >= height-3)
        return;

    int mask[3][3] = {1,2,1, 2,3,2, 1,2,1};

    int sum = 0;

    for (int j = -1; j < 1; ++j)
    {
        for (int i = -1; i < 1; ++i)
        {
            int color = input[(iy + j) * width + (ix + i)];
            sum += color * mask[i + 1][j + 1];
        }
        
    }
    output[iy * width + ix] = sum/15;
    
}

__global__ void brightnessKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width, int step, int channels, int brightness)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //boundry check
    if(ix < width && iy < height)
    {
        //Finding the thread id
        //const int tid = (ix + iy * width) * 3;
        const int tid = iy * step + (channels * ix);
        for (int i = 0; i < channels; i++)
        {
            int pixel = input[tid + i] + brightness;
            if(pixel > 255)
            {
                pixel = 255;
            }
            else if(pixel < 0)
            {
                pixel = 0;
            }
            output[tid + i] = pixel;
        }

    }
    

}

__global__ void bgrToGray(unsigned char* input, unsigned char* output, int width, int height, int colorStep, int grayStep)
{
    //2D indexes of current thread
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    
    //Always check for boundries
    if((ix < width) && (iy < height))
    {
        const int colorTid = iy * colorStep + (3 * ix);

        const int grayTid = iy * grayStep + ix;

        const unsigned char blue = input[colorTid];
        const unsigned char green = input[colorTid + 1];
        const unsigned char red = input[colorTid + 2];

        const float gray = red * 0.3f + green * 0.59f + blue * 0.11f;

        output[grayTid] = static_cast<unsigned char>(gray);
    }

}