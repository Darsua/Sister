#include "../include/mandelbrot.h"

#include <math.h>

double
is_mandelbrot(double complex c) 
{
    double complex z = 0.0 + 0.0 * I;
    int iter;
    for (iter = 0; iter < MAX_ITER; iter++) {
        z = z * z + c;
        if (cabs(z) > 2.0) {
            double log_zn = log(cabs(z));
            double nu = log(log_zn / log(2)) / log(2);
            return iter + 1 - nu;
        }
    }
    return MAX_ITER;
}

int
is_pixel_mandelbrot(int x, int y, Frame* img)
{
    double real = (x - img->width / 2.0) * 4.0 / img->width;
    double imag = (y - img->height / 2.0) * 4.0 / img->height;
    return is_mandelbrot(real + imag * I);
}

void
hsv_to_rgb(double h, double s, double v, unsigned char* r, unsigned char* g, unsigned char* b)
{
    int i = (int)(h * 6);
    double f = h * 6 - i;
    double p = v * (1 - s);
    double q = v * (1 - f * s);
    double t = v * (1 - (1 - f) * s);
    switch (i % 6) {
        case 0: *r = v * 255; *g = t * 255; *b = p * 255; break;
        case 1: *r = q * 255; *g = v * 255; *b = p * 255; break;
        case 2: *r = p * 255; *g = v * 255; *b = t * 255; break;
        case 3: *r = p * 255; *g = q * 255; *b = v * 255; break;
        case 4: *r = t * 255; *g = p * 255; *b = v * 255; break;
        case 5: *r = v * 255; *g = p * 255; *b = q * 255; break;
    }
}

void
color_pixel(Pixel* p, double iter)
{
    if (iter >= MAX_ITER) {
        p->r = 0;
        p->g = 0;
        p->b = 0;
    } else {
        double hue = iter / MAX_ITER;
        double sat = 1.0;
        double val = 1.0;
        hsv_to_rgb(hue, sat, val, &p->r, &p->g, &p->b);
    }
}