#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../include/stb_image_write.h"
#include "../include/mandelbrot.h"

#include <stdlib.h>

Frame
create_frame(int w, int h)
{
    Frame img;
    
    img.width = w;
    img.height = h;
    
    img.data = malloc(w * h * sizeof(Pixel));
    
    img.pixel = malloc(h * sizeof(Pixel*));
    for (int i = 0; i < h; i++) 
        img.pixel[i] = &img.data[i * w];
    
    return img;
}

void
render_frame(Frame *img, unsigned short **ref)
{
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            color_pixel(&img->pixel[y][x], ref[y][x]);
        }
    }
}

void
save_frame(Frame *img)
{
    stbi_write_bmp("mandelbrot.bmp", img->width, img->height, CHANNELS, img->data);
}