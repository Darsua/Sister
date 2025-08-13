#include "../include/bench.h"

void timer_start(Timer* timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->start);
}

void timer_stop(Timer* timer) {
    clock_gettime(CLOCK_MONOTONIC, &timer->end);
}

long timer_elapsed_ms(Timer* timer) {
    long seconds = timer->end.tv_sec - timer->start.tv_sec;
    long nanoseconds = timer->end.tv_nsec - timer->start.tv_nsec;
    return seconds * 1000 + nanoseconds / 1000000;
}

// Unused :p
int
verify_result(unsigned short **resultA, unsigned short **resultB, int width, int height)
{
    int mismatches = 0;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            if (resultA[y][x] != resultB[y][x]) {
                mismatches++;
            }
        }
    }
    return mismatches;
}