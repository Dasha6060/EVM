#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cblas.h>

float** my_allocate_matrix(int N) {
    float** matrix = (float**)malloc(N * sizeof(float*));
    for (int i = 0; i < N; i++) {
        matrix[i] = (float*)calloc(N, sizeof(float));
    }
    return matrix;
}

void my_free_matrix(float** matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void my_identity_matrix(float** I, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            I[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

void my_copy_matrix(float** dest, float** src, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

float my_matrix_norm1(float** A, int N) {
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

float my_matrix_norm_inf(float** A, int N) {
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

// Умножение матриц с BLAS
void my_matrix_multiply(float** C, float** A, float** B, int N) {
    // Преобразуем матрицы в одномерные массивы
    float* A_flat = (float*)malloc(N * N * sizeof(float));
    float* B_flat = (float*)malloc(N * N * sizeof(float));
    float* C_flat = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A[i][j];
            B_flat[i * N + j] = B[i][j];
        }
    }

    // BLAS: C = alpha * A * B + beta * C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                N, N, N, 1.0f, A_flat, N, B_flat, N, 0.0f, C_flat, N);

    // Конвертируем обратно
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = C_flat[i * N + j];
        }
    }

    free(A_flat);
    free(B_flat);
    free(C_flat);
}

// Сложение матриц с BLAS
void my_matrix_add(float** C, float** A, float** B, int N) {
    float* A_flat = (float*)malloc(N * N * sizeof(float));
    float* B_flat = (float*)malloc(N * N * sizeof(float));
    float* C_flat = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A[i][j];
            B_flat[i * N + j] = B[i][j];
        }
    }

    // Копируем A в C
    for (int i = 0; i < N * N; i++) {
        C_flat[i] = A_flat[i];
    }

    // Затем добавляем B: C = A + B
    cblas_saxpy(N * N, 1.0f, B_flat, 1, C_flat, 1);

    // Конвертируем обратно
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = C_flat[i * N + j];
        }
    }

    free(A_flat);
    free(B_flat);
    free(C_flat);
}

// Вычитание матриц с BLAS
void my_matrix_sub(float** C, float** A, float** B, int N) {
    float* A_flat = (float*)malloc(N * N * sizeof(float));
    float* B_flat = (float*)malloc(N * N * sizeof(float));
    float* C_flat = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A[i][j];
            B_flat[i * N + j] = B[i][j];
        }
    }

    // Копируем A в C
    for (int i = 0; i < N * N; i++) {
        C_flat[i] = A_flat[i];
    }

    // C = A + (-1.0) * B
    cblas_saxpy(N * N, -1.0f, B_flat, 1, C_flat, 1);

    // Конвертируем обратно
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = C_flat[i * N + j];
        }
    }

    free(A_flat);
    free(B_flat);
    free(C_flat);
}

// Умножение матрицы на скаляр с BLAS
void my_matrix_scalar_multiply(float** B, float** A, float scalar, int N) {
    float* A_flat = (float*)malloc(N * N * sizeof(float));
    float* B_flat = (float*)malloc(N * N * sizeof(float));

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A[i][j];
        }
    }

    // Копируем A в B
    for (int i = 0; i < N * N; i++) {
        B_flat[i] = A_flat[i];
    }

    // BLAS: скалярное умножение
    cblas_sscal(N * N, scalar, B_flat, 1);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = B_flat[i * N + j];
        }
    }

    free(A_flat);
    free(B_flat);
}

// Функция транспонирования матрицы
void my_matrix_transpose(float** AT, float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            AT[i][j] = A[j][i];
        }
    }
}

// Основная функция обращения матрицы с BLAS
void my_matrix_inverse_series(float** A_inv, float** A, int N, int M) {
    float** I = my_allocate_matrix(N);
    float** AT = my_allocate_matrix(N);
    float** B = my_allocate_matrix(N);
    float** BA = my_allocate_matrix(N);
    float** R = my_allocate_matrix(N);
    float** R_power = my_allocate_matrix(N);
    float** temp = my_allocate_matrix(N);
    float** sum = my_allocate_matrix(N);

    my_identity_matrix(I, N);

    float norm1 = my_matrix_norm1(A, N);
    float norm_inf = my_matrix_norm_inf(A, N);
    float denominator = norm1 * norm_inf;

    if (fabsf(denominator) < 1e-12f) {
        fprintf(stderr, "Error: denominator is too small\n");
        return;
    }

    // Вычисляем B
    my_matrix_transpose(AT, A, N);
    my_matrix_scalar_multiply(B, AT, 1.0f / denominator, N);

    my_matrix_multiply(BA, B, A, N);

    // R = I - BA
    my_matrix_sub(R, I, BA, N);

    // Инициализируем сумму
    my_copy_matrix(sum, I, N);

    // Инициализируем R_power
    my_copy_matrix(R_power, R, N);

    // Вычисляем ряд
    for (int k = 1; k < M; k++) {
        // Добавляем текущую степень R к сумме
        my_matrix_add(temp, sum, R_power, N);
        my_copy_matrix(sum, temp, N);

        if (k < M - 1) {
            // Вычисляем следующую степень: R_power = R_power * R
            my_matrix_multiply(temp, R_power, R, N);
            my_copy_matrix(R_power, temp, N);
        }
    }

    // A_inv = sum * B
    my_matrix_multiply(temp, sum, B, N);
    my_copy_matrix(A_inv, temp, N);

    my_free_matrix(I, N);
    my_free_matrix(AT, N);
    my_free_matrix(B, N);
    my_free_matrix(BA, N);
    my_free_matrix(R, N);
    my_free_matrix(R_power, N);
    my_free_matrix(temp, N);
    my_free_matrix(sum, N);
}

// Функция для инициализации матрицы на рандоме
void my_initialize_random_matrix(float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

int main() {
    int N = 2048;
    int M = 10;

    float** A = my_allocate_matrix(N);
    float** A_inv = my_allocate_matrix(N);

    my_initialize_random_matrix(A, N);
    my_matrix_inverse_series(A_inv, A, N, M);

    my_free_matrix(A, N);
    my_free_matrix(A_inv, N);

    return 0;
}
