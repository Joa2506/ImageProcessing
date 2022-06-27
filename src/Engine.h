#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector_types.h>

#define gpuErrchk(ans) { gpuAssert((ans),__FILE__, __LINE__);}
inline void gpuAssert(cudaError_t code, const char * file, int line, bool abort=true)
{
    if(code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if(abort) exit(code);
    }
}

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

        //Functions that initiates the CUDA kernel.
        //This function converts a openCV image from BGR to Gray by invoking the bgrToGray CUDA Kernel.
        bool convertToGray();
        bool gaussianBlur();
        void clean();
        bool readFile();
        void printDevice();

        void Hello ();


    public:
    
        //Public function for running program
        bool run();
        //Constructor
        Engine(const Configurations& config);
        //Destructor
        ~Engine();



};

        //Global cuda functions
        __global__ void gaussianBlur(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width);
        __global__ void brightness(unsigned char * input, unsigned char* output, unsigned int height, unsigned int width, int brightness);
        __global__ void bgrToGray(unsigned char* input, unsigned char* output, int width, int height, int colorStep, int grayStep);