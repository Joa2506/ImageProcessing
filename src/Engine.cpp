#include <Engine.hpp>
#include <iostream>
#include <fstream>

#include <stdio.h>


Engine::Engine(const Configurations& config) : mConfig(config) {}
Engine::~Engine() {}

bool Engine::readFile()
{

    mImage = cv::imread(mConfig.imageName);
    if (mImage.empty()) 
    {
        std::cout << "Could not open or find the image" << std::endl;
        return false;
    }

    // cv::imshow("image", mImage);
    // cv::waitKey(0);
    return true; 
}
bool Engine::preProcessImage()
{

    mConfig.height = mImage.size().height;
    mConfig.width = mImage.size().width;
    mConfig.channels = mImage.channels();
    mConfig.size = mImage.size().area();
    cv::cvtColor(mImage, mImage, cv::COLOR_BGR2RGB);
    mImage.convertTo(mImage, CV_32FC1);
    printf("Size of image is %d\n", mImage.size().area());

    // imageBufferCPU = (float*)malloc(mConfig.height*mConfig.width*sizeof(float)*3);
    // memset(imageBufferCPU, 0, mConfig.height*mConfig.width*sizeof(float)*3);

    imageBufferCPU = (float*)malloc(mConfig.size*sizeof(float));
    memset(imageBufferCPU, 0, mConfig.size*sizeof(float));
    //cv::imshow("image", mImage);
    //cv::waitKey(0);

    //memcpy(imageBufferCPU, (float*)mImage.data, mImage.size().area()*sizeof(float));
    
    int i = 0;
    if(mImage.isContinuous())
    {
        imageBufferCPU = (float*)mImage.data;
    }    
    printf("Image copied to float buffer\n");
    fflush(stdout);
    //for (i = 0; i < mConfig.size; i++)
    // {
    //     printf("%f\n", imageBufferCPU[i]);
    // }
    
    gpuErrchk(cudaMalloc(&imageBufferGPU, mConfig.size));
    gpuErrchk(cudaMemcpy(imageBufferGPU, imageBufferCPU, mConfig.size, cudaMemcpyHostToDevice));
    printf("Images copied to GPU\n");
    fflush(stdout);

    return true;
}

bool Engine::processImage()
{

    return true;
}

void Engine::clean()
{
    gpuErrchk(cudaFree(imageBufferGPU));
    free(imageBufferCPU);
    gpuErrchk(cudaDeviceReset());

}
bool Engine::run()
{
    //Print device properties
    printDevice();
    //Getting files ready
    readFile();
    preProcessImage();
    //Run program
    
    //Clean up;
    clean();

    return true;
}

void Engine::printDevice()
{
    int nDevice;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&nDevice);

    for (int i = 0; i < nDevice; ++i)
    {
        cudaGetDeviceProperties(&deviceProp, i);
        printf("Device Number: %d\n", i);
        printf("    Device Name: %s\n", deviceProp.name);
        printf("    Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
        printf("    Memory bus widdth: %d(bits)\n", deviceProp.memoryBusWidth);
        printf("    Peak memory bandwidth (GB/s): %f\n\n", 2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);
    }
    

}


