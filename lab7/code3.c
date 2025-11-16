#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void cblas_sgemm(int, int, int, int, int, int, float,
                 const float*, int, const float*, int, float, float*, int);
void cblas_saxpy(int, float, const float*, int, float*, int);
void cblas_sscal(int, float, float*, int);

enum { CblasRowMajor=101, CblasNoTrans=111, CblasTrans=112 };

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

// Функция транспонирования матрицы
void matrix_transpose(float* AT, float* A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            MAT_ELEM(AT, i, j, N) = MAT_ELEM(A, j, i, N);
        }
    }
}

//умножение матриц с транспонированием и BLAS
void matrix_multiply(float* C, float* A, float* B, int N) {
    float* BT = allocate_matrix(N);
    matrix_transpose(BT, B, N);

    // Инициализируем матрицу C нулями
    for (int i = 0; i < N * N; i++) {
        C[i] = 0.0f;
    }

    // BLAS: C = alpha * A * B^T + beta * C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                N, N, N, 1.0f, A, N, BT, N, 0.0f, C, N);

    free_matrix(BT);
}

// Сложение матриц
void matrix_add(float* C, float* A, float* B, int N) {
    copy_matrix(C, A, N);
    cblas_saxpy(N * N, 1.0f, B, 1, C, 1);
}

// Вычитание матриц
void matrix_sub(float* C, float* A, float* B, int N) {
    copy_matrix(C, A, N);
    // C = A + (-1.0) * B
    cblas_saxpy(N * N, -1.0f, B, 1, C, 1);
}

// Умножение матрицы на скаляр
void matrix_scalar_multiply(float* B, float* A, float scalar, int N) {
    copy_matrix(B, A, N);
    cblas_sscal(N * N, scalar, B, 1);
}

// Основная функция обращения матрицы
void my_matrix_inverse_series(float* A_inv, float* A, int N, int M) {
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

    // Вычисляем B
    matrix_transpose(AT, A, N);
    matrix_scalar_multiply(B, AT, 1.0f / denominator, N);

    // Вычисляем BA = B × A
    matrix_multiply(BA, B, A, N);

    // R = I - BA
    matrix_sub(R, I, BA, N);

    // Инициализируем сумму
    copy_matrix(sum, I, N);

    // Инициализируем R_power
    copy_matrix(R_power, R, N);

    // Вычисляем ряд
    for (int k = 1; k < M; k++) {
        // Добавляем текущую степень R к сумме
        matrix_add(temp, sum, R_power, N);
        copy_matrix(sum, temp, N);

        if (k < M - 1) {
            // Вычисляем следующую степень: R_power = R_power × R
            matrix_multiply(temp, R_power, R, N);
            copy_matrix(R_power, temp, N);
        }
    }

    // A_inv = sum × B
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
void my_initialize_random_matrix(float* A, int N) {
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

    my_initialize_random_matrix(A, N);
    my_matrix_inverse_series(A_inv, A, N, M);

    free_matrix(A);
    free_matrix(A_inv);

    return 0;
}
