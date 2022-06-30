#include "Engine.hpp"

int main()
{

    struct Configurations config;
    //config.imageName = "images/GrassAndSea.jpg";
    config.imageName = "images/LennaGray.png";
    Engine engine(config);
    
    engine.run();


    return 0;

}