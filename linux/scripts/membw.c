// Multithreaded streaming-read bandwidth benchmark (approximates llama.cpp
// CPU expert reads). Sums a large buffer with N threads, reports GB/s.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>

#define THREADS 16
#define CHUNK_GB 1L
static double *bufs[THREADS];
static const size_t n = CHUNK_GB * 1024L*1024*1024 / sizeof(double);
static volatile double sink;

static void *reader(void *arg) {
    double *b = bufs[(long)arg];
    double s = 0;
    for (int rep = 0; rep < 4; rep++)
        for (size_t i = 0; i < n; i += 8)
            s += b[i] + b[i+1] + b[i+2] + b[i+3] + b[i+4] + b[i+5] + b[i+6] + b[i+7];
    sink = s;
    return NULL;
}

int main(void) {
    for (long t = 0; t < THREADS; t++) {
        bufs[t] = malloc(n * sizeof(double));
        memset(bufs[t], 1, n * sizeof(double));
    }
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    pthread_t th[THREADS];
    for (long t = 0; t < THREADS; t++) pthread_create(&th[t], NULL, reader, (void*)t);
    for (long t = 0; t < THREADS; t++) pthread_join(th[t], NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
    double gb = (double)THREADS * 4 * CHUNK_GB;
    printf("read bandwidth: %.1f GB/s (%ld threads, %.0f GB in %.2f s)\n", gb/sec, (long)THREADS, gb, sec);
    return 0;
}
