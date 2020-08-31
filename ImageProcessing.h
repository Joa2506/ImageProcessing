#ifndef IMAGEPROCESSING_H
#define IMAGEPROCESSING_H

static const int _512by512_IMG_SIZE = 262144;
static const int BMP_COLOR_TABLE_SIZE = 1024;
static const int BMP_HEADER_SIZE = 54;
static const int MAX_COLOR = 255;
static const int MIN_COLOR = 0;
static const int WHITE = MAX_COLOR;
static const int BLACK = MIN_COLOR;
static const int NUMBER_OF_GRAYLEVELS = 255; 

class ImageProcessing
{
    public:
        ImageProcessing(
                        const char *_in_img_name, 
                        const char * _out_img_name,
                        int * _height,
                        int * _width,
                        int * _bit_depth,
                        unsigned char * _header,
                        unsigned char * _colortable,
                        unsigned char * _in_buf,
                        unsigned char * _out_buff
                        );
        void read_image();
        void write_image();
        void copy_image_data(unsigned char * _src_buf, unsigned char * _dest_buf, int buf_size);
        void increase_brightness(unsigned char * _input_image_data, unsigned char * _out_image_data, int img_size, int brightness);
        void decrease_brightness(unsigned char * _input_image_data, unsigned char * _out_image_data, int img_size, int brightness);
        void compute_histogram(unsigned char * _img_data, int img_rows, int img_cols, float hist[], const char * histfile);
        void equalize_histogram(unsigned char * _input_img_data, unsigned char * _output_img_data, int img_rows, int img_cols);
        virtual ~ImageProcessing();

    protected:

    private:
        const char *input_name;
        const char *output_name;
        int * height;
        int * width;
        int * bit_depth;
        unsigned char * header;
        unsigned char * colortable;
        unsigned char * in_buf;
        unsigned char * out_buf;


};
#endif