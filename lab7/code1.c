//Без ручной векторизации
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Функция для выделения памяти под матрицу
float** allocate_matrix(int N) {
    float** matrix = (float**)malloc(N * sizeof(float*)); // Mассив указателей на float* - строки матрицы
    for (int i = 0; i < N; i++) {
        matrix[i] =  (float*)malloc(N * sizeof(float)); // Выделяем память для каждой строки матрицы
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

// Функция для создания единичной матрицы
void identity_matrix(float** I, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            //Если находимся на главной диагонали, ставим 1
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

// Функция для вычисления первой нормы (максимальная сумма по столбцам)
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

// Функция для вычисления второй нормы (максимальная сумма по строкам)
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

// Функция умножения
void matrix_multiply(float** C, float** A, float** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            float a_ik = A[i][k];
            for (int j = 0; j < N; j++) {
                C[i][j] += a_ik * B[k][j];
            }
        }
    }
}

// Функция сложения
void matrix_add(float** C, float** A, float** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
}

// Функция вычитания
void matrix_sub(float** C, float** A, float** B, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
}

// Функция умножения на скаляр
void matrix_scalar_multiply(float** B, float** A, float scalar, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            B[i][j] = scalar * A[i][j];
        }
    }
}

// Функция транспонирования
void matrix_transpose(float** AT, float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            AT[i][j] = A[j][i];
        }
    }
}

// Функция обращения матрицы методом разложения в ряд
void matrix_inverse_series(float** A_inv, float** A, int N, int M) {
    float** I = allocate_matrix(N);
    float** AT = allocate_matrix(N);
    float** B = allocate_matrix(N);
    float** BA = allocate_matrix(N);
    float** R = allocate_matrix(N);
    float** R_power = allocate_matrix(N);
    float** temp = allocate_matrix(N);
    float** sum = allocate_matrix(N);

    // Создаем единичную матрицу
    identity_matrix(I, N);

    // Вычисляем нормы матрицы A
    float norm1 = matrix_norm1(A, N);
    float norm_inf = matrix_norm_inf(A, N);
    float denominator = norm1 * norm_inf;

    // Проверка знаменателя
    if (fabsf(denominator) < 1e-12f) {
        fprintf(stderr, "Error\n");
        return;
    }

    // Вычисляем B
    matrix_transpose(AT, A, N);
    matrix_scalar_multiply(B, AT, 1.0f / denominator, N);

    // Вычисляем R = I - B * A
    matrix_multiply(BA, B, A, N);
    matrix_sub(R, I, BA, N);

    copy_matrix(sum, I, N);

    // R_power - текущая степень матрицы R
    copy_matrix(R_power, R, N);

    // Вычисляем ряд sum
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
            A[i][j] = (float)rand() / RAND_MAX * 2.0f - 1.0f; // [-1, 1]
        }
    }
}

// Функция печати матрицы
void print_matrix(float** A, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}


int main() {
    int N = 2048;
    int M = 10;

    float** A = allocate_matrix(N);
    float** A_inv = allocate_matrix(N);

    initialize_random_matrix(A, N);

    // printf("matrix A:\n");
    // print_matrix(A, N);

    matrix_inverse_series(A_inv, A, N, M);

    // printf("matrix A_inv:\n");
    // print_matrix(A_inv, N);

    free_matrix(A, N);
    free_matrix(A_inv, N);

    return 0;
}
