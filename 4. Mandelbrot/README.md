# Mandelbrot of Madness

This program renders the Mandelbrot set using three different methods using both the CPU and the GPU mostly written in C.

## Core

Each method utilizes the same basic logic within.

```c
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
```

The `is_mandelbrot()` function determines if a number in the complex plane is part of the Mandelbrot set. The returned value is how many iterations it takes a value to "escape" the set until it reaches the `MAX_ITER` limit which is the threshold a value is considered to be in the Mandelbrot set.

```c
int
is_pixel_mandelbrot(int x, int y, Frame* img)
{
    double real = (x - img->width / 2.0) * 4.0 / img->width;
    double imag = (y - img->height / 2.0) * 4.0 / img->height;
    return is_mandelbrot(real + imag * I);
}
```

A `is_pixel_mandelbrot()` function then uses this to check if a pixel of a given frame centered around the point of origin and extending to the known bounds of the Mandelbrot set (a circle of radius 2 in the complex plane) in the set.

## Serial

The underlying algorithm is then to just loop through each pixel in a frame to calculate the Mandelbrot set.

```c
for (int y = 0; y < img->height; y++) {
    for (int x = 0; x < img->width; x++) {
        result[y][x] = is_pixel_mandelbrot(x, y, img);
    }
}
```

## Parallel

Utilizing the OpenMP library, it's very simple to extend this calculation to multiple threads, paralleling the process.

```c
#pragma omp parallel for
for (int y = 0; y < img->height; y++) {
    #pragma omp parallel for
    for (int x = 0; x < img->width; x++) {
        result[y][x] = is_pixel_mandelbrot(x, y, img);
    }
}
```

## CUDA

This process is extended to the CUDA platform by utilizing a kernel which re-implements the `is_mandelbrot()` and `is_pixel_mandelbrot()` using thread-block indexing with support for scaling and panning to calculate only a part of the set.

```c
__global__ void
kernel_mandelbrot(unsigned short* result, int width, int height, 
                  double center_real, double center_imag, double scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    
    // I don't seem to be able to call any functions
    // or use any structs from outside the kernel...
    
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
```

# Compiling the Program

To compile the program, ensure that the necessary dependencies are installed. Once they are set up, use the provided `Makefile` to build the executable.

### Prerequisites

- GCC (with OpenMP)

- CUDA Toolkit

- Raylib

### Building the Executable

The project includes a `Makefile` that automates the compilation process.

1. Navigate to the project directory.

2. Run the `make` command.
   
   ```bash
   make
   ```

3. The resulting binary will be available at `bin/mandelbrot`

### Running the Program

Simply execute the provided or compiled binary.

```bash
bin/mandelbrot
```

# Benchmarks

| Resolution | Serial   | Parallel (I9-11900H x 16) | CUDA (RTX 3050)    |
| ---------- | -------- | ------------------------- | ------------------ |
| 1000x1000  | ~444 ms  | ~150 ms (296.00%)         | ~247 ms (179.75%)  |
| 2000x2000  | ~1890 ms | ~597 ms (316.58%)         | ~292 ms (647.26%)  |
| 4000x4000  | ~7177 ms | ~2367 ms (303.21%)        | ~459 ms (1506.36%) |
| 8000x8000  | 29336 ms | 9596 ms (305.71%)         | 961 ms (3052.65%)  |

# Fractal Example

<img title="" src="mandelbrot.png" alt="">
