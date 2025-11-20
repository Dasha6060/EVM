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
    for (int i = 0; i < N; i++) indices[i] = i;

    for (int i = N - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }

    for (int i = 0; i < N - 1; i++) array[indices[i]] = indices[i + 1];
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

int get_iterations(int N) {
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

    int min_elements = 16;
    int max_elements = 32 * 1024 * 1024; 

    for (int n = min_elements; n <= max_elements; ) {
        int* array = malloc(n * sizeof(int));
        if (!array) {
            n = (int)(n * 1.2);
            continue;
        }

        int K = get_iterations(n);
        int size_kb = (n * sizeof(int)) / 1024;
        double size_mb = size_kb / 1024.0;

        forward(array, n);
        double t_forward = find_min_time(array, n, K);

        backward(array, n);
        double t_backward = find_min_time(array, n, K);

        random(array, n);
        double t_random = find_min_time(array, n, K / 2); 

        if (size_kb < 1024) {
            printf("%d,%d,0.00,%.2f,%.2f,%.2f\n", n, size_kb, t_forward, t_backward, t_random);
        } else {
            printf("%d,%d,%.2f,%.2f,%.2f,%.2f\n", n, size_kb, size_mb, t_forward, t_backward, t_random);
        }

        free(array);
        
        if (n < 1024) n += 32;
        else if (n < 8192) n += 256;
        else if (n < 65536) n += 2048;
        else if (n < 524288) n += 16384;
        else if (n < 4194304) n += 131072;
        else n += 524288;

        if (n > max_elements) n = max_elements;
    }

    return 0;
}

