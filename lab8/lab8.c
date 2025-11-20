#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define NUM_ITER 5

uint64_t read_tsc() {
    uint32_t lo, hi;
    asm volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

void forward(int* array, int N) {
    for (int i = 0; i < N - 1; i++) array[i] = i + 1;
    array[N - 1] = 0;
}

void backward(int* array, int N) {
    for (int i = 1; i < N; i++) array[i] = i - 1;
    array[0] = N - 1;
}

void random(int* array, int N) {
    int* indices = malloc(N * sizeof(int));
    for (int i = 0; i < N; i++) {
        indices[i] = i;
    }

    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    for (int i = 0; i < N - 1; i++) {
        array[indices[i]] = indices[i + 1];
    }
    array[indices[N - 1]] = indices[0];

    free(indices);
}

void warm_up(const int* array, int n) {
    int k = 0;
    for (int i = 0; i < n; i++) k = array[k];
    asm volatile ("" : : "r"(k));
}

double measure_pass(const int* array, int N, int K) {
    int k = 0;
    uint64_t start = read_tsc();
    for (int i = 0; i < N * K; i++) k = array[k];
    uint64_t end = read_tsc();
    asm volatile ("" : : "r"(k));
    return (double)(end - start) / (N * K);
}

int get_K(int N) {
    if (N <= 256) return 10000;
    if (N <= 1024) return 5000;
    if (N <= 8192) return 2000;
    if (N <= 65536) return 1000;
    if (N <= 262144) return 500;
    if (N <= 1048576) return 200;
    if (N <= 4194304) return 100;
    return 50;
}

double find_min_time(const int* array, int N, int K) {
    double min_time = 1e20;
    for (int i = 0; i < NUM_ITER; i++) {
        warm_up(array, N);
        double t = measure_pass(array, N, K);
        if (t < min_time) min_time = t;
    }
    return min_time;
}

int main() {
    srand(time(NULL));
    printf("Elements,Size_KB,Size_MB,Forward,Backward,Random\n");

    int sizes[] = {
        16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240,
        256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192,
        12288, 16384, 24576, 32768, 49152, 65536, 98304, 131072, 196608, 262144,
        393216, 524288, 786432, 1048576, 1572864, 2097152, 3145728, 4194304,
        6291456, 8388608, 12582912, 16777216
    };

    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int n = sizes[i];
        int* array = malloc(n * sizeof(int));
        if (!array) {
            printf("Memory allocation failed for size %d\n", n);
            continue;
        }

        int K = get_K(n);
        int size_kb = (n * sizeof(int)) / 1024;
        double size_mb = size_kb / 1024.0;

        forward(array, n);
        double t_forward = find_min_time(array, n, K);

        backward(array, n);
        double t_backward = find_min_time(array, n, K);

        random(array, n);
        double t_random = find_min_time(array, n, K / 2);

        if (size_kb < 1024) {
            printf("%d,%d,0.0,%.2f,%.2f,%.2f\n", n, size_kb, size_mb, t_forward, t_backward, t_random);
        } else {
            printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n", n, size_kb, size_mb, t_forward, t_backward, t_random);
        }
        free(array);
    }
    fprintf(stderr, "Measurement completed successfully!\n");
    return 0;
}

