#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

#define MAT_ELEM(matrix, i, j, N) ((matrix)[(i) * (N) + (j)])

// Функция для выделения памяти под матрицу
float* allocate_matrix(int N) {
    return (float*)calloc(N * N, sizeof(float));
}

// Функция освобождения памяти матрицы
void free_matrix(float* matrix) {
    free(matrix);
}

// Функция для создания единичной матрицы
void identity_matrix(float* I, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            MAT_ELEM(I, i, j, N) = (i == j) ? 1.0f : 0.0f;
        }
    }
}

// Функция для копирования матрицы
void copy_matrix(float* dest, float* src, int N) {
    for (int i = 0; i < N * N; i++) {
        dest[i] = src[i];
    }
}

// Функция для вычисления первой нормы (максимальная сумма по столбцам)
float matrix_norm1(float* A, int N) {
    float max_sum = 0.0f;
    for (int j = 0; j < N; j++) {
        float col_sum = 0.0f;
        for (int i = 0; i < N; i++) {
            col_sum += fabsf(MAT_ELEM(A, i, j, N));
        }
        if (col_sum > max_sum) {
            max_sum = col_sum;
        }
    }
    return max_sum;
}

// Функция для вычисления второй нормы (максимальная сумма по строкам)
float matrix_norm_inf(float* A, int N) {
    float max_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            row_sum += fabsf(MAT_ELEM(A, i, j, N));
        }
        if (row_sum > max_sum) {
            max_sum = row_sum;
        }
    }
    return max_sum;
}

// Функция транспонирования
void matrix_transpose(float* AT, float* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            MAT_ELEM(AT, i, j, N) = MAT_ELEM(A, j, i, N);
        }
    }
}

// SSE умножение матриц
void matrix_multiply(float* C, float* A, float* B, int N) {
    float* BT = allocate_matrix(N);
    matrix_transpose(BT, B, N);

    // Инициализируем матрицу C нулями
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 8) {  // Обрабатываем по 8 элементов
            __m128 sum0 = _mm_setzero_ps();
            __m128 sum1 = _mm_setzero_ps();

            for (int k = 0; k < N; k++) {
                float a_val = MAT_ELEM(A, i, k, N);
                __m128 a = _mm_set1_ps(a_val);

                __m128 b0 = _mm_loadu_ps(&MAT_ELEM(BT, j, k, N));
                __m128 b1 = _mm_loadu_ps(&MAT_ELEM(BT, j+4, k, N));

                sum0 = _mm_add_ps(sum0, _mm_mul_ps(a, b0));
                sum1 = _mm_add_ps(sum1, _mm_mul_ps(a, b1));
            }


            _mm_storeu_ps(&MAT_ELEM(C, i, j, N), sum0);
            _mm_storeu_ps(&MAT_ELEM(C, i, j+4, N), sum1);
        }
    }
    free_matrix(BT);
}

// SSE сложение матриц
void matrix_add(float* C, float* A, float* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            if (j + 4 <= N) {
                __m128 a_vec = _mm_loadu_ps(&MAT_ELEM(A, i, j, N));
                __m128 b_vec = _mm_loadu_ps(&MAT_ELEM(B, i, j, N));
                __m128 result = _mm_add_ps(a_vec, b_vec);
                _mm_storeu_ps(&MAT_ELEM(C, i, j, N), result);
            } else {
                for (int k = j; k < N; k++) {
                    MAT_ELEM(C, i, k, N) = MAT_ELEM(A, i, k, N) + MAT_ELEM(B, i, k, N);
                }
            }
        }
    }
}

//вычитание матриц
void matrix_sub(float* C, float* A, float* B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            if (j + 4 <= N) {
                __m128 a_vec = _mm_loadu_ps(&MAT_ELEM(A, i, j, N));
                __m128 b_vec = _mm_loadu_ps(&MAT_ELEM(B, i, j, N));
                __m128 result = _mm_sub_ps(a_vec, b_vec);
                _mm_storeu_ps(&MAT_ELEM(C, i, j, N), result);
            } else {
                // Обработка хвоста
                for (int k = j; k < N; k++) {
                    MAT_ELEM(C, i, k, N) = MAT_ELEM(A, i, k, N) - MAT_ELEM(B, i, k, N);
                }
            }
        }
    }
}

//умножение на скаляр
void matrix_scalar_multiply(float* B, float* A, float scalar, int N) {
    __m128 scalar_vec = _mm_set1_ps(scalar);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            if (j + 4 <= N) {
                __m128 a_vec = _mm_loadu_ps(&MAT_ELEM(A, i, j, N));
                __m128 result = _mm_mul_ps(a_vec, scalar_vec);
                _mm_storeu_ps(&MAT_ELEM(B, i, j, N), result);
            } else {
                // Обработка хвоста
                for (int k = j; k < N; k++) {
                    MAT_ELEM(B, i, k, N) = MAT_ELEM(A, i, k, N) * scalar;
                }
            }
        }
    }
}

// Основная функция обращения матрицы
void matrix_inverse_series(float* A_inv, float* A, int N, int M) {
    float* I = allocate_matrix(N);
    float* AT = allocate_matrix(N);
    float* B = allocate_matrix(N);
    float* BA = allocate_matrix(N);
    float* R = allocate_matrix(N);
    float* R_power = allocate_matrix(N);
    float* temp = allocate_matrix(N);
    float* sum = allocate_matrix(N);

    identity_matrix(I, N);

    float norm1 = matrix_norm1(A, N);
    float norm_inf = matrix_norm_inf(A, N);
    float denominator = norm1 * norm_inf;

    if (fabsf(denominator) < 1e-12f) {
        fprintf(stderr, "Error: denominator is too small\n");
        return;
    }

    matrix_transpose(AT, A, N);
    matrix_scalar_multiply(B, AT, 1.0f / denominator, N);

    matrix_multiply(BA, B, A, N);
    matrix_sub(R, I, BA, N);

    copy_matrix(sum, I, N);
    copy_matrix(R_power, R, N);

    for (int k = 1; k < M; k++) {
        matrix_add(temp, sum, R_power, N);
        copy_matrix(sum, temp, N);

        if (k < M - 1) {
            matrix_multiply(temp, R_power, R, N);
            copy_matrix(R_power, temp, N);
        }
    }

    matrix_multiply(temp, sum, B, N);
    copy_matrix(A_inv, temp, N);

    free_matrix(I);
    free_matrix(AT);
    free_matrix(B);
    free_matrix(BA);
    free_matrix(R);
    free_matrix(R_power);
    free_matrix(temp);
    free_matrix(sum);
}

// Функция для инициализации матрицы на рандоме
void initialize_random_matrix(float* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            MAT_ELEM(A, i, j, N) = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

int main() {
    int N = 2048;
    int M = 10;

    float* A = allocate_matrix(N);
    float* A_inv = allocate_matrix(N);

    initialize_random_matrix(A, N);
    matrix_inverse_series(A_inv, A, N, M);

    free_matrix(A);
    free_matrix(A_inv);

    return 0;
}
