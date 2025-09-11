#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <sys/times.h>

void bubble_sort(int arr[], int N) {
    for (int j = N - 1; j > 0; j--) {
        for (int i = 0; i < j; i++) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }
    }
}

int main() {
    int N;
    printf("N: ");
    scanf("%d", &N);

    int* arr = (int*)malloc(N * sizeof(int));
    srand(12345);
    for (int i = 0; i < N; i++) {
        arr[i] = rand();
    }

    struct tms start_tms, end_tms;
    long ticks_per_sec = sysconf(_SC_CLK_TCK);

    clock_t start_ticks = times(&start_tms);
    bubble_sort(arr, N);
    clock_t end_ticks = times(&end_tms);

    double result = (double)(end_ticks - start_ticks) / ticks_per_sec;

    printf("Time: %.6f sec\n", result);

    free(arr);
    return 0;
}
