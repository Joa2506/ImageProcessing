#define WIDTH 16
#define HEIGHT 16

__global__ void gaussianBlurKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width, unsigned int channels, unsigned int step);

//Increases the brightness of the input image
__global__ void brightnessKernel(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width,int step, int channels, int brightness);

//creates a output buffer that is the greyscale variant of the input image
__global__ void bgrToGray(unsigned char* input, unsigned char* output, int width, int height, int colorStep, int grayStep);