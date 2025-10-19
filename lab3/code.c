#include <stdio.h>

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
    scanf("%d", &N);
    int arr[N];
    
    for (int i = 0; i < N; i++) {
        scanf("%d", &arr[i]);
    }
    bubble_sort(arr, N);

    for (int i = 0; i < N; i++) {
        printf("%d ", arr[i]);
    }
    
    return 0;
}
