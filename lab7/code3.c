//Матричные операции
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cblas.h>

float** allocate_matrix(int N) {
    float** matrix = (float**)malloc(N * sizeof(float*)); // Mассив указателей на float* - это строки матрицы
    for (int i = 0; i < N; i++) {
        matrix[i] = (float*)calloc(N, sizeof(float));
    }
    return matrix;
}

// Функция освобождения памяти матрицы
void free_matrix(float** matrix, int N) {
    for (int i = 0; i < N; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Функция создания единичной матрицы
void identity_matrix(float** I, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            I[i][j] = (i == j) ? 1.0f : 0.0f;
        }
    }
}

// Функция копирования матрицы
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

// Умножение матриц с BLAS
void blas_matrix_multiply(float** C, float** A, float** B, int N) {
    // Преобразуем матрицы в одномерные массивы
    float* A_flat = new float[N * N];
    float* B_flat = new float[N * N];
    float* C_flat = new float[N * N];

    //элементы одной строки идут последовательно друг за другом
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A[i][j];
            B_flat[i * N + j] = B[i][j];
        }
    }

    // BLAS: C = alpha * A * B + beta * C (С - существующа матрица может содержать предыдущий результат)
    cblas_sgemm(CblasRowMajor,    // Порядок хранения
                CblasNoTrans,     // Не транспонировать A
                CblasNoTrans,     // Не транспонировать B
                N,                // Число строк A и C
                N,                // Число столбцов B и C
                N,                // Число столбцов A и строк B
                1.0f,             // alpha коэффициент для А*В
                A_flat,           // Матрица A
                N,                // шаг между строками A
                B_flat,           // Матрица B
                N,                // шаг между строками B
                0.0f,             // beta коэффициент для С
                C_flat,           // Матрица C (результат)
                N);           // шаг между строками C

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
void blas_matrix_add(float** C, float** A, float** B, int N) {
    // Преобразуем в одномерные массивы
    float* A_flat = new float[N * N];
    float* B_flat = new float[N * N];
    float* C_flat = new float[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A[i][j];
            B_flat[i * N + j] = B[i][j];
        }
    }

    // BLAS: Y = alpha * X + Y
    // Сначала копируем A в C
    for (int i = 0; i < N * N; i++) {
        C_flat[i] = A_flat[i];
    }
    // Затем добавляем B: C = A + B
    cblas_saxpy(N * N,           // количество элементов
                1.0f,            // alpha
                B_flat,          // X
                1,               // шаг по X
                C_flat,          // Y (результат)
                1);              // шаг по Y

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
void blas_matrix_sub(float** C, float** A, float** B, int N) {
    float* A_flat = new float[N * N];
    float* B_flat = new float[N * N];
    float* C_flat = new float[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A[i][j];
            B_flat[i * N + j] = B[i][j];
        }
    }

    // копируем A в C
    for (int i = 0; i < N * N; i++) {
        C_flat[i] = A_flat[i];
    }

    // C = A + (-1.0) * B
    cblas_saxpy(N * N,           // количество элементов
                -1.0f,           // alpha = -1.0
                B_flat,          // B
                1,               // шаг
                C_flat,          // C (результат)
                1);              // шаг

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
void blas_matrix_scalar_multiply(float** B, float** A, float scalar, int N) {
    float* A_flat = new float[N * N];
    float* B_flat = new float[N * N];

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A_flat[i * N + j] = A[i][j];
        }
    }

    // Копируем A в B чтобы сохранить исходную матрицу А
    for (int i = 0; i < N * N; i++) {
        B_flat[i] = A_flat[i];
    }

    // BLAS: скалярное умножение
    cblas_sscal(N * N,           // количество элементов
                scalar,          // скаляр
                B_flat,          // вектор
                1);              // шаг

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = B_flat[i * N + j];
        }
    }

    free(A_flat);
    free(B_flat);
}

// Функция транспонирования матрицы
void matrix_transpose(float** AT, float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            AT[i][j] = A[j][i];
        }
    }
}

// Основная функция обращения матрицы с BLAS
void blas_matrix_inverse_series(float** A_inv, float** A, int N, int M) {
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

    if (std::abs(denominator) < 1e-12f) {
        fprintf(stderr, "Error\n");
        return;
    }

    // Вычисляем B
    matrix_transpose(AT, A, N);
    blas_matrix_scalar_multiply(B, AT, 1.0f / denominator, N);

    blas_matrix_multiply(BA, B, A, N);

    // R
    blas_matrix_sub(R, I, BA, N);

    // Инициализируем сумму
    copy_matrix(sum, I, N);

    // Инициализируем R_power
    copy_matrix(R_power, R, N);

    // Вычисляем ряд
    for (int k = 1; k < M; k++) {
        // Добавляем текущую степень R к сумме
        blas_matrix_add(temp, sum, R_power, N);
        copy_matrix(sum, temp, N);

        if (k < M - 1) {
            // Вычисляем следующую степень: R_power
            blas_matrix_multiply(temp, R_power, R, N);
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

// Функция для инициализации матрицы на рандоме
void initialize_random_matrix(float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
        }
    }
}

int main() {
    int N = 4;
    int M = 100000;

    float** A = allocate_matrix(N);
    float** A_inv = allocate_matrix(N);

    initialize_random_matrix(A, N);

    // Замеряем время
    clock_t start = clock();

    // Вычисляем обратную матрицу
    blas_matrix_inverse_series(A_inv, A, N, M);

    clock_t end = clock();
    double duration = (double)(end - start) / CLOCKS_PER_SEC * 1000000; 

    printf("Time: %.0f\n", duration);

    free_matrix(A, N);
    free_matrix(A_inv, N);

    return 0;
}

