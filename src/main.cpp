#include "Engine.hpp"

int main()
{

    struct Configurations config;
    config.imageName = "images/Lenna.png";
    Engine engine(config);
    
    engine.run();


    return 0;

}