#include "../include/frame.h"

#include <complex.h>

#define MAX_ITER 500

double is_mandelbrot(double _Complex c);

int is_pixel_mandelbrot(int x, int y, Frame* img);

void color_pixel(Pixel* p, double iter);