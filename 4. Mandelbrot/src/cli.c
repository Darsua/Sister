#include "../include/cpu.h"
#include "../include/gpu.h"
#include "../include/bench.h"
 
#include <stdio.h>
#include <stdlib.h>

int
cli()
{
    system("clear");
    printf("Mandelbrot of Madness!!!\n");
    
    int size;
    printf("Enter the resolution of the image: ");
    scanf("%d", &size);
    printf("\n");
    
    Frame img = create_frame(size, size);
    Timer timer; long time_serial, time_parallel, time_cuda;
    unsigned short **result_serial, **result_parallel, **result_cuda;
    
    
    // Serial processing
    timer_start(&timer);
    result_serial = serial(&img);
    timer_stop(&timer);
    
    printf("Elapsed time: %ld ms\n\n", time_serial = timer_elapsed_ms(&timer));
    
    
    // Parallel processing
    timer_start(&timer);
    result_parallel = parallel(&img);
    timer_stop(&timer);
    
    printf("Elapsed time: %ld ms\n", time_parallel = timer_elapsed_ms(&timer));
    printf("Speedup: %.2f%%\n\n", (double)time_serial / time_parallel * 100.0);
    
    
    // CUDA processing
    timer_start(&timer);
    result_cuda = cuda(&img);
    timer_stop(&timer);
    
    printf("Elapsed time: %ld ms\n", time_cuda = timer_elapsed_ms(&timer));
    printf("Speedup: %.2f%%\n\n", (double)time_serial / time_cuda * 100.0);

    
    // Render and save the image
    printf("Rendering image...\n");
    render_frame(&img, result_cuda);
    printf("Saving image...\n");
    save_frame(&img);
    
    printf("\nAll done!\n");
    printf("You can find the image in the current directory as 'mandelbrot.bmp'.\n");
    return 0;
}