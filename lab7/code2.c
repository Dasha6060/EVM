//С ручной векторизацией (Расширение GCC)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <immintrin.h>

float** allocate_aligned_matrix(int N) {
    // массив указателей на строки матрицы
    float** matrix = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        // Выделяем память с выравниванием по 16 байтам для SSE (_mm_malloc)
        matrix[i] = (float*)_mm_malloc(N * sizeof(float), 16);
        for (int j = 0; j < N; j++) {
            matrix[i][j] = 0.0f;
        }
    }
    return matrix;
}

// Функция для освобождения выровненной памяти
void free_matrix(float** matrix, int N) {
    for (int i = 0; i < N; i++) {
        _mm_free(matrix[i]);
    }
    free(matrix);
}

// Функция для создания единичной матрицы
void identity_matrix(float** I, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            I[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

// Функция для копирования матрицы
void copy_matrix(float** dest, float** src, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

// Функция вычисления нормы (максимальная сумма по столбцам)
float matrix_norm1(float** A, int N) {
    float max_sum = 0.0f;

    for (int j = 0; j < N; j++) {
        float col_sum = 0.0f;
        for (int i = 0; i < N; i++) {
            col_sum += fabsf(A[i][j]);
        }
        if (col_sum > max_sum) {
            max_sum = col_sum;
        }
    }

    return max_sum;
}

// Функция вычисления нормы (максимальная сумма по строкам)
float matrix_norm_inf(float** A, int N) {
    float max_sum = 0.0f;

    for (int i = 0; i < N; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row_sum += fabsf(A[i][j]);
        }
        if (row_sum > max_sum) {
            max_sum = row_sum;
        }
    }

    return max_sum;
}

// умножение матриц с использованием SSE
void sse_matrix_multiply(float** C, float** A, float** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {  // Обрабатываем по 4 элемента за раз (размер SSE вектора)
            //Инициализируем вектор суммы четырьмя нулями
            __m128 sum = _mm_setzero_ps();

            for (int k = 0; k < N; k++) {
                // Создаем вектор, где все 4 элемента равны A[i][k]
                __m128 a_vec = _mm_set1_ps(A[i][k]);

                // Загружаем 4 элемента из строки B[k]
                __m128 b_vec = _mm_load_ps(&B[k][j]);

                // Умножаем и добавляем к сумме
                __m128 product = _mm_mul_ps(a_vec, b_vec);
                sum = _mm_add_ps(sum, product);
            }

            // Сохраняем 4 результата в матрицу С
            _mm_store_ps(&C[i][j], sum);
        }

        // Обработка оставшихся элементов (если N не кратен 4)
        for (int j = N - (N % 4); j < N; j++) {
            C[i][j] = 0.0f;
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// сложение матриц с использованием SSE
void sse_matrix_add(float** C, float** A, float** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            // Загружаем по 4 элемента из A и B
            __m128 a_vec = _mm_load_ps(&A[i][j]);
            __m128 b_vec = _mm_load_ps(&B[i][j]);

            // Складываем векторы
            __m128 sum = _mm_add_ps(a_vec, b_vec);

            // Сохраняем результат
            _mm_store_ps(&C[i][j], sum);
        }

        // Обработка оставшихся элементов
        for (int j = N - (N % 4); j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// вычитание матриц с использованием SSE
void sse_matrix_sub(float** C, float** A, float** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            // Загружаем по 4 элемента из A и B
            __m128 a_vec = _mm_load_ps(&A[i][j]);
            __m128 b_vec = _mm_load_ps(&B[i][j]);

            // Вычитаем векторы
            __m128 diff = _mm_sub_ps(a_vec, b_vec);

            // Сохраняем результат
            _mm_store_ps(&C[i][j], diff);
        }

        // Обработка оставшихся элементов
        for (int j = N - (N % 4); j < N; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// умножение матрицы на скаляр
void sse_matrix_scalar_multiply(float** B, float** A, float scalar, int N) {
    __m128 scalar_vec = _mm_set1_ps(scalar);  // Вектор из 4х скаляров

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            // Загружаем 4 элемента из A
            __m128 a_vec = _mm_load_ps(&A[i][j]);

            // Умножаем на скаляр
            __m128 result = _mm_mul_ps(a_vec, scalar_vec);

            // Сохраняем результат
            _mm_store_ps(&B[i][j], result);
        }

        // Обработка оставшихся элементов
        for (int j = N - (N % 4); j < N; j++) {
            B[i][j] = A[i][j] * scalar;
        }
    }
}

// Функция транспонирования матрицы
void matrix_transpose(float** AT, float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            AT[i][j] = A[j][i];
        }
    }
}

// функция обращения матрицы с SSE
void sse_matrix_inverse_series(float** A_inv, float** A, int N, int M) {
    float** I = allocate_aligned_matrix(N);
    float** AT = allocate_aligned_matrix(N);
    float** B = allocate_aligned_matrix(N);
    float** BA = allocate_aligned_matrix(N);
    float** R = allocate_aligned_matrix(N);
    float** R_power = allocate_aligned_matrix(N);
    float** temp = allocate_aligned_matrix(N);
    float** sum = allocate_aligned_matrix(N);

    identity_matrix(I, N);

    float norm1 = matrix_norm1(A, N);
    float norm_inf = matrix_norm_inf(A, N);
    float denominator = norm1 * norm_inf;

    if (std::abs(denominator) < 1e-12f) {
        fprintf(stderr, "Error\n");
        return;
    }

    // Вычисляем B
    matrix_transpose(AT, A, N);
    sse_matrix_scalar_multiply(B, AT, 1.0f / denominator, N);

    // Вычисляем R
    sse_matrix_multiply(BA, B, A, N);
    sse_matrix_sub(R, I, BA, N);

    // Инициализируем сумму
    copy_matrix(sum, I, N);

    // Инициализируем R_power
    copy_matrix(R_power, R, N);

    // Вычисляем ряд
    for (int k = 1; k < M; k++) {
        // Добавляем текущую степень R к сумме
        sse_matrix_add(temp, sum, R_power, N);
        copy_matrix(sum, temp, N);

        if (k < M - 1) {
            // Вычисляем следующую степень
            sse_matrix_multiply(temp, R_power, R, N);
            copy_matrix(R_power, temp, N);
        }
    }

    copy_matrix(A_inv, sum, N);

    free_matrix(I, N);
    free_matrix(AT, N);
    free_matrix(B, N);
    free_matrix(BA, N);
    free_matrix(R, N);
    free_matrix(R_power, N);
    free_matrix(temp, N);
    free_matrix(sum, N);
}

// Функция инициализации матрицы на рандоме
void initialize_random_matrix(float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
             A[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

int main() {
    int N = 256;
    int M = 10;

    float** A = allocate_aligned_matrix(N);
    float** A_inv = allocate_aligned_matrix(N);

    initialize_random_matrix(A, N);

    // Замеряем время выполнения
    clock_t start = clock();

    sse_matrix_inverse_series(A_inv, A, N, M);

    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC * 1000000; 

    printf("Time: %.0f\n", duration);

    free_matrix(A, N);
    free_matrix(A_inv, N);

    return 0;
}

