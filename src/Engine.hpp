#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector_types.h>


struct Configurations {
    std::string imageName;
    int height;
    int width;
    int channels;
    int size;
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
        //Methods
        bool fileExist(std::string filename);
        void clean();
        bool readFile();
        void printDevice();

        //Functions that initiates the CUDA kernel.
        //This function converts a openCV image from BGR to Gray by invoking the bgrToGray CUDA Kernel.
        bool convertToGray();
        bool gaussianBlur();
        void brightness(int brightnessLevel);


        void Hello ();


    public:
    
        //Public function for running program
        bool run();
        //Constructor
        Engine(const Configurations& config);
        //Destructor
        ~Engine();



};
