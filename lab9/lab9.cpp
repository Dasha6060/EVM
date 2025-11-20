#include <iostream>
#include <climits>
#include <x86intrin.h>

const unsigned int N = 400000000;
const unsigned int RUN_TIMES = 6;

void initArray(unsigned int *array, unsigned int fragments, size_t offset, size_t size) {
    size_t j = 1;
    size_t i = 0;
    for(; i < size; i++) {
        for(j = 1; j < fragments; j++) {
            array[i + (j - 1) * offset] = i + j * offset;
        }
        array[i + (j - 1) * offset] = i + 1;
    }
    array[i - 1 + (j - 1) * offset] = 0;
}

unsigned long long runArray(unsigned int const *array) {
    unsigned long long startTime, endTime;
    unsigned long long minTime = ULLONG_MAX;
    for(size_t j = 0; j < RUN_TIMES; j++) {
        startTime = __rdtsc();

        for(volatile size_t k = 0, i = 0; i < N; i++) {
            k = array[k];
        }
        endTime = __rdtsc();
        if (minTime > endTime - startTime) {
            minTime = endTime - startTime;
        }
    }
    return minTime;
}

void countTime(unsigned int *array, unsigned int fragments, int offset, int size) {
    initArray(array, fragments, offset, size);
    std::cout << fragments << " fragments \t" << runArray(array) / N << " tacts" << std::endl;
}

int main() {
    auto *array = (unsigned int *) malloc(N * sizeof(unsigned int));

    if (array == nullptr) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }

    unsigned int offset = 16 * 1024 * 1024; // 16MB
    unsigned int BlockSize = 128 * 1024; // 128KB

    for(int fragments = 1; fragments <= 32; fragments++) {
        countTime(array, fragments, offset / sizeof(int), BlockSize / sizeof(int));
    }
    free(array);
    return 0;
}
