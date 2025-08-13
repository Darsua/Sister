#include "../include/gpu.h"
#include <stdio.h>

__global__ void
kernel_mandelbrot(unsigned short* result, int width, int height, 
                  double center_real, double center_imag, double scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    // I don't seem to be able to call any functions or use any structs from outside the kernel...
    
    // Relative coordinates
    double real = center_real + (x - width / 2.0) * 4.0 / (scale * width);
    double imag = center_imag + (y - height / 2.0) * 4.0 / (scale * height);
    
    double z_real = 0.0, z_imag = 0.0;
    int i;
    for (i = 0; i < MAX_ITER; i++) {
        double temp = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2.0 * z_real * z_imag + imag;
        z_real = temp;
        if (z_real * z_real + z_imag * z_imag > 4.0) break;
    }
    result[y * width + x] = i;
}

unsigned short**
cuda(Frame* img)
{
    printf("Processing set using CUDA...\n");
    return cuda_next(img, 0.0, 0.0, 1.0);
}

unsigned short**
cuda_next(Frame* img, double center_real, double center_imag, double scale)
{
    int img_size = img->width * img->height * sizeof(unsigned short);
    unsigned short* h_result_flat = (unsigned short*)malloc(img_size);
    
    // Copy host to device
    unsigned short* d_result;
    cudaMalloc(&d_result, img_size);

    // Launch kernel
    dim3 blockDim(16, 16);
    dim3 gridDim((img->width + 15) / 16, (img->height + 15) / 16);
    
    kernel_mandelbrot<<<gridDim, blockDim>>>
    (d_result, img->width, img->height, center_real, center_imag, scale);

    // Copy device to host
    cudaMemcpy(h_result_flat, d_result, img_size, cudaMemcpyDeviceToHost);
    cudaFree(d_result);

    // Return 2D array (very tricky)
    unsigned short** h_result_2d = (unsigned short**)malloc(img->height * sizeof(unsigned short*));
    for (int i = 0; i < img->height; i++)
        h_result_2d[i] = h_result_flat + i * img->width;

    return h_result_2d;
}
