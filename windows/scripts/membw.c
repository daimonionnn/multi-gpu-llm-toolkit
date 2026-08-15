// Multithreaded streaming-read bandwidth benchmark (approximates llama.cpp
// CPU expert reads). Sums a large buffer with N threads, reports GB/s.
//
// Windows port of linux/scripts/membw.c - same buffer size, same repeat count
// and the same output line, so the two numbers are directly comparable. Only
// the threading and clock are platform code.
//
// Build and run with windows/scripts/run-membw.ps1, or by hand:
//   cl /O2 /MT membw.c            (from a Developer PowerShell)
//   clang -O2 membw.c -o membw.exe
//
// Optional argument: thread count (default 16, as on Linux).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#define MAX_THREADS 128
#define CHUNK_GB 1L

static int threads = 16;
static double *bufs[MAX_THREADS];
static const size_t n = CHUNK_GB * 1024L * 1024 * 1024 / sizeof(double);
static volatile double sink;

static DWORD WINAPI reader(LPVOID arg) {
    double *b = bufs[(intptr_t)arg];
    double s = 0;
    for (int rep = 0; rep < 4; rep++)
        for (size_t i = 0; i < n; i += 8)
            s += b[i] + b[i+1] + b[i+2] + b[i+3] + b[i+4] + b[i+5] + b[i+6] + b[i+7];
    sink = s;
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1) {
        threads = atoi(argv[1]);
        if (threads < 1 || threads > MAX_THREADS) {
            fprintf(stderr, "thread count must be 1..%d\n", MAX_THREADS);
            return 1;
        }
    }

    for (intptr_t t = 0; t < threads; t++) {
        bufs[t] = (double *)malloc(n * sizeof(double));
        if (!bufs[t]) {
            fprintf(stderr, "allocation failed at thread %lld - need %ld GB total\n",
                    (long long)t, (long)threads * CHUNK_GB);
            return 1;
        }
        // Touch every page so the read loop measures RAM, not page faults.
        memset(bufs[t], 1, n * sizeof(double));
    }

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    HANDLE th[MAX_THREADS];
    for (intptr_t t = 0; t < threads; t++)
        th[t] = CreateThread(NULL, 0, reader, (LPVOID)t, 0, NULL);
    WaitForMultipleObjects(threads, th, TRUE, INFINITE);

    QueryPerformanceCounter(&t1);
    double sec = (double)(t1.QuadPart - t0.QuadPart) / (double)freq.QuadPart;
    double gb = (double)threads * 4 * CHUNK_GB;
    printf("read bandwidth: %.1f GB/s (%d threads, %.0f GB in %.2f s)\n",
           gb / sec, threads, gb, sec);

    for (intptr_t t = 0; t < threads; t++) {
        CloseHandle(th[t]);
        free(bufs[t]);
    }
    return 0;
}
