#include "Engine.h"


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

__global__ void brightness(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width, int brightness)
{
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;
    const int iy = blockIdx.y * blockDim.y + threadIdx.y;

    //boundry check
    if(ix < width && iy < height)
    {
        
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


bool Engine::convertToGray()
{

    const int colorBytes = mImageInput.step * mImageInput.rows;
	const int grayBytes = mImageOutput.step * mImageOutput.rows;

    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, colorBytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, grayBytes));
    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), colorBytes, cudaMemcpyHostToDevice));
    printf("Images copied to GPU\n");

    const dim3 block(16,16);
    const dim3 grid((mImageInput.cols + block.x -1)/block.x, (mImageInput.rows + block.y - 1)/block.y);

    bgrToGray <<<grid, block>>>(d_input, d_output, mImageInput.cols, mImageInput.rows, mImageInput.step, mImageOutput.step);
    
    //cudaDeviceSynchronize();
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(mImageOutput.ptr(), d_output, grayBytes, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));

    fflush(stdout);

    return true;
}


bool Engine::gaussianBlur()
{

    const int colorBytes = mImageInput.step * mImageInput.rows;
	
    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, colorBytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, colorBytes));
    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), colorBytes, cudaMemcpyHostToDevice));
    printf("Images copied to GPU\n");

    const dim3 block(16,16);
    const dim3 grid((mImageInput.cols + block.x -1)/block.x, (mImageInput.rows + block.y - 1)/block.y);

    gaussianBlurKernel <<<grid, block>>>(d_input, d_output, mImageInput.cols, mImageInput.rows);
    
    //cudaDeviceSynchronize();
    gpuErrchk(cudaDeviceSynchronize());
    
    gpuErrchk(cudaMemcpy(mImageOutput.ptr(), d_output, colorBytes, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));

    fflush(stdout);

    return true;
}

extern "C" void Engine::Hello()
{
    printf("Hello\n");
}