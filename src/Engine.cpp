#include "Engine.h"
#include <iostream>
#include <fstream>

#include <stdio.h>


Engine::Engine(const Configurations& config) : mConfig(config) {}
Engine::~Engine() {}

extern "C" void Hello();
bool Engine::readFile()
{
    mImageInput = cv::imread(mConfig.imageName);
    mImageOutput = cv::Mat(mImageInput.rows, mImageInput.cols, CV_8UC1);
    if (mImageInput.empty()) 
    {
        std::cout << "Could not open or find the image" << std::endl;
        return false;
    }

    if(!mImageInput.isContinuous())
    {
        return false;
    }    
    return true; 
}


void Engine::clean()
{
    gpuErrchk(cudaDeviceReset());
}
bool Engine::run()
{
    //Print device properties
    printDevice();
    //Getting files ready
    readFile();
    //convertToGray();
    gaussianBlur();
    //Hello();
    //Run program
    cv::imshow("Input",mImageInput);
	cv::imshow("Output",mImageOutput);
    cv::waitKey(0);
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


