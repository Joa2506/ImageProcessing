//#include "Engine.hpp"
#include "ImageProcessing.cuh"
bool Engine::convertToGray()
{
    mImageOutput = cv::Mat(mImageInput.rows, mImageInput.cols, CV_8UC1);
    const int colorBytes = mBytes;
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

void Engine::brightness(int brightnessLevel)
{
    //Output image of type unsigned char, with three channels
    mImageOutput = cv::Mat(mImageInput.rows, mImageInput.cols, CV_8UC3);

    //Device buffers
    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, mBytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, mBytes));

    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), mBytes, cudaMemcpyHostToDevice));
    printf("Image uploaded to GPU\n");

    const dim3 block(WIDTH, HEIGHT);
    //const dim3 grid(mImageInput.cols, mImageInput.rows);
    const dim3 grid((mConfig.width + block.x -1)/block.x, (mConfig.height + block.y - 1)/block.y);
    printf("Channels %d\n",mConfig.channels);
    brightnessKernel <<<grid, block>>>(d_input, d_output, mConfig.width, mConfig.height, mConfig.step, mConfig.channels, brightnessLevel);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(mImageOutput.ptr(), d_output, mBytes, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));


}

void Engine::swapPixels(color c1, color c2)
{
    //Output image of type unsigned char, with three channels
    mImageOutput = cv::Mat(mImageInput.rows, mImageInput.cols, CV_8UC3);
    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, mBytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, mBytes));

    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), mBytes, cudaMemcpyHostToDevice));

    const dim3 block(WIDTH, HEIGHT);
    const dim3 grid((mConfig.width + block.x -1)/block.x, (mConfig.height + block.y - 1)/block.y);

    //Inset swap kernel
    swapPixelKernel<<<grid, block>>>(d_input, d_output, mConfig.width, mConfig.height, mConfig.step, mConfig.channels, c1, c2);

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(mImageOutput.ptr(), d_output, mBytes, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));


}

//Kernel convolutions
void Engine::edgeDetectionSobel()
{
    mImageOutput = cv::Mat(mImageInput.rows, mImageInput.cols, CV_8UC1);
    const int colorBytes = mBytes;
	const int grayBytes = mImageOutput.step * mImageOutput.rows;

    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, colorBytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, grayBytes));
    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), colorBytes, cudaMemcpyHostToDevice));
    printf("Images copied to GPU\n");

    const dim3 block(WIDTH, HEIGHT);
    const dim3 grid((mImageInput.cols + block.x -1)/block.x, (mImageInput.rows + block.y - 1)/block.y);
    //First transform to gray
    bgrToGray <<<grid, block>>>(d_input, d_input, mImageInput.cols, mImageInput.rows, mImageInput.step, mImageOutput.step);

    
}

bool Engine::gaussianBlur()
{

    mImageOutput = cv::Mat(mImageInput.rows, mImageInput.cols, CV_8UC3);

    //Device buffers
    unsigned char * d_input, *d_output;

    gpuErrchk(cudaMalloc<unsigned char>(&d_input, mBytes));
    gpuErrchk(cudaMalloc<unsigned char>(&d_output, mBytes));

    gpuErrchk(cudaMemcpy(d_input, mImageInput.ptr(), mBytes, cudaMemcpyHostToDevice));
    printf("Image uploaded to GPU\n");

    const dim3 block(WIDTH, HEIGHT);
    //const dim3 grid(mImageInput.cols, mImageInput.rows);
    const dim3 grid((mConfig.width + block.x -1)/block.x, (mConfig.height + block.y - 1)/block.y);
    printf("Channels %d\n",mConfig.channels);
    gaussianBlurKernel <<<grid, block>>>(d_input, d_output, mConfig.height, mConfig.width, mConfig.channels, mConfig.step);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(mImageOutput.ptr(), d_output, mBytes, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));

    return true;
}