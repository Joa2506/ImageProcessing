#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector_types.h>
#include "utils.hpp"
struct Configurations {

    //Image meta data
    std::string imageName;
    int height;
    int width;
    int channels;
    int size;
    int step;
};



class Engine
{
    private:
        //Variables
        struct Configurations mConfig;
        float * imageBufferCPU;
        float * imageBufferGPU;
        cv::Mat mImageInput;
        cv::Mat mImageOutput;
        int mBytes;
        //Methods
        bool fileExists(std::string filename);
        void clean();
        bool init();
        void printDevice();

        //Functions that initiates the CUDA kernel.
        //This function converts a openCV image from BGR to Gray by invoking the bgrToGray CUDA Kernel.
        bool convertToGray();
        bool gaussianBlur();
        void brightness(int brightnessLevel);
        void swapPixels(color c1, color c2);
        void edgeDetection();

        void Hello ();


    public:
    
        //Public function for running program
        bool run();
        //Constructor
        Engine(const Configurations& config);
        //Destructor
        ~Engine();



};
