#include <iostream>
#include "ImageProcessing.h"
#include <stdlib.h>
#include <stdio.h>


//Constructor

ImageProcessing::ImageProcessing(
                                const char *_in_img_name, 
                                const char * _out_img_name,
                                int * _height,
                                int * _width,
                                int * _bit_depth,
                                unsigned char * _header,
                                unsigned char * _colortable,
                                unsigned char * _in_buf,
                                unsigned char * _out_buff
                                )
{
        input_name  = _in_img_name;
        output_name = _out_img_name;
        height      = _height;
        width       = _width;
        bit_depth   = _bit_depth;
        header      = _header;
        colortable  = _colortable;
        in_buf      = _in_buf;
        out_buf     = _out_buff;
}

void ImageProcessing::read_image()
{
    int i;
    FILE * stream_in;
    stream_in = fopen(input_name, "rb");
    if(stream_in == (FILE *) 0)
    {
        std::cout << "Unable to open file";
        exit(0);
    }

    //Bitmap header er 54
    for(i = 0; i < BMP_HEADER_SIZE; i++)
    {
        header[i] = getc(stream_in);
    }
    *width = *(int * )&header[18]; //Leser header 18 som inneholder info om bredde
    *height = *(int *)&header[22]; //Leser header 22 som inneholder info om hoyde
    *bit_depth = *(int *)&header[28]; // Leser header 28 som inneholder info om bit dybde

    //Er det fargebilde?
    if(*bit_depth <= 8)
    {
        fread(colortable, sizeof(unsigned char), BMP_COLOR_TABLE_SIZE, stream_in);
    }

    fread(in_buf, sizeof(unsigned char), _512by512_IMG_SIZE, stream_in);
    fclose(stream_in);
}

void ImageProcessing::write_image()
{
    FILE * stream_out;
    stream_out = fopen(output_name, "wb");
    fwrite(header, sizeof(unsigned char), BMP_HEADER_SIZE, stream_out);
    if(*bit_depth <= 8)
    {
        fwrite(colortable, sizeof(unsigned char), BMP_COLOR_TABLE_SIZE, stream_out);
    }

    fwrite(out_buf, sizeof(unsigned char),_512by512_IMG_SIZE, stream_out);
    fclose(stream_out);

}

void ImageProcessing::increase_brightness(unsigned char * _input_image_data, unsigned char * _out_image_data, int img_size, int brightness)
{
    int i;
    for(i = 0; i < img_size; i++)
    {
        int temp = _input_image_data[i] + brightness;
        _out_image_data[i] = (temp > MAX_COLOR)? MAX_COLOR : temp;
    }
}

void ImageProcessing::decrease_brightness(unsigned char * _input_image_data, unsigned char * _out_image_data, int img_size, int brightness)
{
    int i;
    for (i = 0; i < img_size; i++)
    {
        int temp = _input_image_data[i] - brightness;
        _out_image_data[i]  = (temp < MIN_COLOR)? MIN_COLOR : temp;
    }
    
}
void ImageProcessing::copy_image_data(unsigned char * _src_buf, unsigned char * _dest_buf, int buf_size)
{
    int i;
    for (i = 0; i < buf_size; i++)
    {
        _dest_buf[i] = _src_buf[i];
    }
    
}

void ImageProcessing::compute_histogram(unsigned char * _img_data, int img_rows, int img_cols, float hist[], const char * histfile)
{
    FILE * fptr;
    fptr = fopen(histfile, "w");
    int x,y,i,j;
    long int ihist[256], sum;
    for(i = 0; i <= 255; i++)
    {
        ihist[i] = 0;
    }
    sum = 0;
    for (y = 0; y < img_rows; y++)
    {   
        
        
        for (x = 0; x < img_cols; x++)
        {
            
            j = *(_img_data+x+y*img_cols);
            
            ihist[j] = ihist[j] + 1;
            sum = sum + 1;
        }
        
    }
    
    //Legger data inn i histogram array bassert på sannsynligheten for hver pixel
    for (i = 0; i <= 255; i++)
    {
        hist[i] = (float)ihist[i]/(float)sum;
    }
    //Skriver til fil
    for(i = 0; i <= 255; i++)
    {
        fprintf(fptr, "\n%f", hist[i]);
    }
    
    fclose(fptr);
}
//Brukes for å øke kontrasten i bildet
void ImageProcessing::equalize_histogram(unsigned char * _input_img_data, unsigned char * _output_img_data, int img_rows, int img_cols)
{
    int x, y, i, j;
    int histeq[256];
    float hist[256];
    float sum;
    const char init_hist[] = "init_hist.txt";
    const char final_hist[] = "final_hist.txt";

    compute_histogram(_input_img_data, img_rows, img_cols, &hist[0], init_hist);

    for (i = 0; i < 255; i++)
    {
        sum  =  0.0;
        for (j = 0; j <= i; j++)
        {
            sum = sum+hist[j];
        }
        histeq[i] = (int)(255*sum+0.5);
    }
    for (y = 0; y < img_rows; y++)
    {
        for (x = 0; x < img_cols; x++)
        {
            *(_output_img_data+x+y*img_cols) = histeq[*(_input_img_data+x+y*img_cols)];
        }
        
    }
    compute_histogram(&_output_img_data[0], img_rows, img_cols, &hist[0], final_hist);
    

}

ImageProcessing::~ImageProcessing()
{
    
}

