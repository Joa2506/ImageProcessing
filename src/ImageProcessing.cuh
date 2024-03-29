#include "Engine.hpp"
#define WIDTH 16
#define HEIGHT 16

#define MASK_DIM 3
#define MASK_OFFSET (MASK_DIM/2)
__constant__ int mask[3*3];

__global__ void gaussianBlurKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width, unsigned int channel, unsigned int step);

//Increases the brightness of the input image
__global__ void brightnessKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width,int step, int channels, int brightness);

//creates a output buffer that is the greyscale variant of the input image
__global__ void bgrToGray(unsigned char* input, unsigned char* output, int width, int height, int colorStep, int grayStep);

//Swaps two pixels values
__global__ void swapPixelKernel(unsigned char * input, unsigned char* output, int width, int height, int step, int channels, color c1, color c2);

__global__ void sobelKernel(unsigned char * input, unsigned char* output, int width, int height, int step);

__global__ void convolution2d(unsigned char * input, unsigned char * output, int size, int channel);