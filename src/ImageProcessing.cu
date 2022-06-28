#include <ImageProcessing.cuh>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void gaussianBlurKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width)
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

__global__ void brightnessKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width, int brightness, int step)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //boundry check
    if(ix < width && iy < height)
    {
        //Finding the thread id
        const int tid = (iy * width + ix);

        for (int i = 0; i < 3; i++)
        {
            output[tid + i] = input[tid + i];
        }

        
        // // //Pixel thread of bgr image
        // // const unsigned char blue = input[tid];
        // // const unsigned char green = input[tid + 1];
        // // const unsigned char red = input[tid + 2];

        // // unsigned int valueR = red + brightness;
        // // unsigned int valueG  = green + brightness;
        // // unsigned int valueB = blue + brightness;
        
        // for (int i = 0; i < 3; i++)
        // {
        //     int newPixel = input[tid + i] + brightness;

        //     if(newPixel > 255)
        //     {
        //         newPixel = 255;
        //     }
        //     else if(newPixel < 0)
        //     {
        //         newPixel = 0;
        //     }

        //     output[tid + i] = newPixel;
        // }
        
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