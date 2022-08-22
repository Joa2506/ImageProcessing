#include "ImageProcessing.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>



// __global__ void gaussianBlurKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width, unsigned int channel, unsigned int step)
// {
//     //Calculate the global thread positions
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     const int tid = row * step + (channel * col);
//     //Starting index for calculation
//     int start_r = row - MASK_OFFSET;
//     int start_c = col - MASK_OFFSET;

//     //Temp to accumulate result
//     int temp = 0;

//     for (int i = 0; i < MASK_DIM; i++)
//     {
//         for (int j = 0; j < MASK_DIM; j++)
//         {
//             if((start_r + 1) >= 0 && (start_r + i) < width*height*channel)
//             {
//                 if((start_c + j) >= 0 && (start_r + j) < width*height*channel)
//                 {
//                     temp += input[(start_r + i)* width*height*channel + (start_c + j)] * mask[i * MASK_DIM + j];
//                 }
//             }
//         }
        
//     }
//     output[tid] = temp;
// }

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
//Swaps the red and green pixel values
__global__ void swapPixelKernel(unsigned char * input, unsigned char* output, int width, int height, int step, int channels, color c1, color c2)
{
        //2D indexes of current thread
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int missingChannel;

    if(channels != 3)
    {
        return;
    }

    missingChannel = abs((c1 + c2)-channels);

    if((ix < width) && (iy < height))
    {
        const int tid = iy * step + (channels * ix);
        
        int firstPixel = input[tid + c1];
        int secondPixel = input[tid + c2];

        /*
        Swapping with XOR. Obviously redundant as I could just have placed the pixels where they needed to go in the ouput array. 
        A decent solution if swapping elements on the same array to return an altered version.
        */
        firstPixel ^= secondPixel;
        secondPixel ^= firstPixel;
        firstPixel ^= secondPixel;
         
        output[tid + c1] = firstPixel;
        output[tid + c2] = secondPixel;
        output[tid + missingChannel] = input[tid + missingChannel];

    }
}

__global__ void convolution2d(unsigned char * input, unsigned char * output, int size, int channel)
{
    //2D indexes of current thread
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    
}