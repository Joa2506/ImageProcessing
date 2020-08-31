#include <iostream>
#include "ImageProcessing.h"

int main(int argc, char const *argv[])
{
    float img_hist[NUMBER_OF_GRAYLEVELS];
    int img_width, img_height, img_bit_depth;
    unsigned char img_header[BMP_HEADER_SIZE];
    unsigned char img_color[BMP_COLOR_TABLE_SIZE];
    unsigned char img_in_buf[_512by512_IMG_SIZE];
    unsigned char img_out_buf[_512by512_IMG_SIZE];

    const char img_name[] = "Images/mickey.bmp";
    const char img_new_name[] = "Images/mickey_eqz.bmp";

    ImageProcessing *myImage = new ImageProcessing(img_name, img_new_name, &img_height, &img_width, &img_bit_depth, img_header, img_color, img_in_buf, img_out_buf);
    myImage->read_image();
    myImage->equalize_histogram(img_in_buf, img_out_buf,img_height, img_width);
    myImage->write_image();
    
 
    std::cout << "Success!" << std::endl;

    return 0;
}
