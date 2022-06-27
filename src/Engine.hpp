#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
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
        cv::Mat mImage;
        //Methods
        bool fileExist(std::string filename);
        bool preProcessImage();
        bool processImage();
        void clean();
        bool readFile();
        void printDevice();

        //Global cuda functions
        __global__ void gaussianBlur(float * imageGPU, float* imageCPU, unsigned int height, unsigned int width);
        __global__ void brightness(float * imageGPU, float* imageCPU, unsigned int height, unsigned int width, int brightness);
    public:
        //Public function for running program
        bool run();
        //Constructor
        Engine(const Configurations& config);
        //Destructor
        ~Engine();



};