#include "mandelbrot.h"

// This is weird
#ifdef __cplusplus
extern "C" {
#endif

unsigned short** cuda(Frame* img);
unsigned short** cuda_next(Frame* img, double center_real, double center_imag, double scale);

#ifdef __cplusplus
}
#endif