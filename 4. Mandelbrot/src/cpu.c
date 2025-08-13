#include "../include/mandelbrot.h"
#include "../include/cpu.h"

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

unsigned short**
serial(Frame* img)
{
    unsigned short** result = malloc(img->height * sizeof(unsigned short*));
    for (int i = 0; i < img->height; i++)
        result[i] = malloc(img->width * sizeof(unsigned short));
    
    printf("Processing set in serial...\n");
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            result[y][x] = is_pixel_mandelbrot(x, y, img);
        }
    }
    
    return result;
}

unsigned short**
parallel(Frame* img)
{
    unsigned short** result = malloc(img->height * sizeof(unsigned short*));
    for (int i = 0; i < img->height; i++)
        result[i] = malloc(img->width * sizeof(unsigned short));
    
    printf("Processing set in parallel with %d threads...\n", omp_get_max_threads());
    #pragma omp parallel for
    for (int y = 0; y < img->height; y++) {
        #pragma omp parallel for
        for (int x = 0; x < img->width; x++) {            
            result[y][x] = is_pixel_mandelbrot(x, y, img);
        }
    }
    
    return result;
}