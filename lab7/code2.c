#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>

float** allocate_matrix(int N) {
    float** matrix = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (float*)calloc(N, sizeof(float));
    }
    return matrix;
}

void free_matrix(float** matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void identity_matrix(float** I, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            I[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void copy_matrix(float** dest, float** src, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

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

void matrix_transpose(float** AT, float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            AT[i][j] = A[j][i];
        }
    }
}

void sse_matrix_multiply(float** C, float** A, float** B, int N) {
    // Транспонируем B один раз перед всеми умножениями
    float** BT = allocate_matrix(N);
    matrix_transpose(BT, B, N);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 8) {  // Обрабатываем по 8 элементов
            __m128 sum0 = _mm_setzero_ps();
            __m128 sum1 = _mm_setzero_ps();

            for (int k = 0; k < N; k++) {
                __m128 a = _mm_set1_ps(A[i][k]);

                // Загружаем 8 элементов за две операции
                __m128 b0 = _mm_loadu_ps(&BT[j][k]);
                __m128 b1 = _mm_loadu_ps(&BT[j+4][k]);

                sum0 = _mm_add_ps(sum0, _mm_mul_ps(a, b0));
                sum1 = _mm_add_ps(sum1, _mm_mul_ps(a, b1));
            }

            _mm_storeu_ps(&C[i][j], sum0);
            _mm_storeu_ps(&C[i][j+4], sum1);
        }
    }
    free_matrix(BT, N);
}


void sse_matrix_add(float** C, float** A, float** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            if (j + 4 <= N) {
                __m128 a_vec = _mm_loadu_ps(&A[i][j]);
                __m128 b_vec = _mm_loadu_ps(&B[i][j]);
                __m128 result = _mm_add_ps(a_vec, b_vec);
                _mm_storeu_ps(&C[i][j], result);
            } else {
                for (int k = j; k < N; k++) {
                    C[i][k] = A[i][k] + B[i][k];
                }
            }
        }
    }
}

void sse_matrix_sub(float** C, float** A, float** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            if (j + 4 <= N) {
                __m128 a_vec = _mm_loadu_ps(&A[i][j]);
                __m128 b_vec = _mm_loadu_ps(&B[i][j]);
                __m128 result = _mm_sub_ps(a_vec, b_vec);
                _mm_storeu_ps(&C[i][j], result);
            } else {
                for (int k = j; k < N; k++) {
                    C[i][k] = A[i][k] - B[i][k];
                }
            }
        }
    }
}

void sse_matrix_scalar_multiply(float** B, float** A, float scalar, int N) {
    __m128 scalar_vec = _mm_set1_ps(scalar);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j += 4) {
            if (j + 4 <= N) {
                __m128 a_vec = _mm_loadu_ps(&A[i][j]);
                __m128 result = _mm_mul_ps(a_vec, scalar_vec);
                _mm_storeu_ps(&B[i][j], result);
            } else {
                for (int k = j; k < N; k++) {
                    B[i][k] = A[i][k] * scalar;
                }
            }
        }
    }
}


void sse_matrix_inverse_series(float** A_inv, float** A, int N, int M) {
    float** I = allocate_matrix(N);
    float** AT = allocate_matrix(N);
    float** B = allocate_matrix(N);
    float** BA = allocate_matrix(N);
    float** R = allocate_matrix(N);
    float** R_power = allocate_matrix(N);
    float** temp = allocate_matrix(N);
    float** sum = allocate_matrix(N);

    identity_matrix(I, N);

    float norm1 = matrix_norm1(A, N);
    float norm_inf = matrix_norm_inf(A, N);
    float denominator = norm1 * norm_inf;

    if (fabsf(denominator) < 1e-12f) {
        fprintf(stderr, "Error: denominator is too small\n");
        return;
    }

    matrix_transpose(AT, A, N);
    sse_matrix_scalar_multiply(B, AT, 1.0f / denominator, N);

    sse_matrix_multiply(BA, B, A, N);
    sse_matrix_sub(R, I, BA, N);

    copy_matrix(sum, I, N);
    copy_matrix(R_power, R, N);

    for (int k = 1; k < M; k++) {
        sse_matrix_add(temp, sum, R_power, N);
        copy_matrix(sum, temp, N);

        if (k < M - 1) {
            sse_matrix_multiply(temp, R_power, R, N);
            copy_matrix(R_power, temp, N);
        }
    }

    sse_matrix_multiply(temp, sum, B, N);
    copy_matrix(A_inv, temp, N);

    free_matrix(I, N);
    free_matrix(AT, N);
    free_matrix(B, N);
    free_matrix(BA, N);
    free_matrix(R, N);
    free_matrix(R_power, N);
    free_matrix(temp, N);
    free_matrix(sum, N);
}

void initialize_random_matrix(float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

int main() {
    int N = 2048;
    int M = 10;

    float** A = allocate_matrix(N);
    float** A_inv = allocate_matrix(N);

    initialize_random_matrix(A, N);
    sse_matrix_inverse_series(A_inv, A, N, M);

    free_matrix(A, N);
    free_matrix(A_inv, N);

    return 0;
}

