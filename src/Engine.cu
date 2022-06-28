#include "Engine.hpp"
#include <ImageProcessing.cuh>
#include "utils.hpp"
bool Engine::convertToGray()
{

    const int colorBytes = mImageInput.step * mImageInput.rows;
	const int grayBytes = mImageOutput.step * mImageOutput.rows;

    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, colorBytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, grayBytes));
    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), colorBytes, cudaMemcpyHostToDevice));
    printf("Images copied to GPU\n");

    const dim3 block(WIDTH, HEIGHT);
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

    //Size of image in bytes
    const int bytes = mConfig.width * mConfig.height * mConfig.channels;

    //Device buffers
    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, bytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, bytes));

    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), bytes, cudaMemcpyHostToDevice));
    printf("Image uploaded to GPU\n");

    const dim3 block(WIDTH, HEIGHT);
    //const dim3 grid(mImageInput.cols, mImageInput.rows);
    const dim3 grid((mConfig.width + block.x -1)/block.x, (mConfig.height + block.y - 1)/block.y);
    printf("Channels %d\n",mConfig.channels);
    gaussianBlurKernel <<<grid, block>>>(d_input, d_output, mConfig.height, mConfig.width, mConfig.channels, mConfig.step);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(mImageOutput.ptr(), d_output, bytes, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));

    return true;
}

void Engine::brightness(int brightnessLevel)
{
    //Size of image in bytes
    const int bytes = mConfig.width * mConfig.height * mConfig.channels;

    //Device buffers
    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, bytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, bytes));

    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), bytes, cudaMemcpyHostToDevice));
    printf("Image uploaded to GPU\n");

    const dim3 block(WIDTH, HEIGHT);
    //const dim3 grid(mImageInput.cols, mImageInput.rows);
    const dim3 grid((mConfig.width + block.x -1)/block.x, (mConfig.height + block.y - 1)/block.y);
    printf("Channels %d\n",mConfig.channels);
    brightnessKernel <<<grid, block>>>(d_input, d_output, mConfig.width, mConfig.height, mConfig.step, mConfig.channels, brightnessLevel);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(mImageOutput.ptr(), d_output, bytes, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));


}
