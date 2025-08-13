#include <time.h>

typedef struct {
    struct timespec start;
    struct timespec end;
} Timer;

void timer_start(Timer* timer);

void timer_stop(Timer* timer);

long timer_elapsed_ms(Timer* timer);

int verify_result(unsigned short **resultA, unsigned short **resultB, int width, int height);