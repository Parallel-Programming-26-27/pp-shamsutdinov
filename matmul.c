#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef void (*Func)(int, double *, double *, double *, int);

void ijk_seq(int n, double *A, double *B, double *C, int bs) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

void ijk_omp(int n, double *A, double *B, double *C, int bs) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double sum = 0;
            for (int k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
}

void ikj_seq(int n, double *A, double *B, double *C, int bs) {
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
}

void ikj_omp(int n, double *A, double *B, double *C, int bs) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++)
        for (int k = 0; k < n; k++) {
            double a = A[i * n + k];
            for (int j = 0; j < n; j++)
                C[i * n + j] += a * B[k * n + j];
        }
}

void blocked_omp(int n, double *A, double *B, double *C, int bs) {
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += bs)
        for (int jj = 0; jj < n; jj += bs)
            for (int kk = 0; kk < n; kk += bs)
                for (int i = ii; i < ii + bs && i < n; i++)
                    for (int k = kk; k < kk + bs && k < n; k++) {
                        double a = A[i * n + k];
                        for (int j = jj; j < jj + bs && j < n; j++)
                            C[i * n + j] += a * B[k * n + j];
                    }
}

double measure(Func f, int n, double *A, double *B, double *C, int bs, int reps) {
    double total = 0;
    for (int r = 0; r < reps; r++) {
        for (int i = 0; i < n * n; i++) C[i] = 0;
        double t0 = omp_get_wtime();
        f(n, A, B, C, bs);
        total += omp_get_wtime() - t0;
    }
    return total / reps;
}

int main() {
    int n, bs, threads, reps;
    printf("Matrix size N: ");
    scanf("%d", &n);
    printf("Block size: ");
    scanf("%d", &bs);
    printf("Threads: ");
    scanf("%d", &threads);
    printf("Repeats: ");
    scanf("%d", &reps);
    omp_set_num_threads(threads);

    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C = malloc(n * n * sizeof(double));
    double *Cref = malloc(n * n * sizeof(double));

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            A[i * n + j] = ((i + 2 * j) % 10) / 10.0;
            B[i * n + j] = ((3 * i + j) % 10) / 10.0;
        }

    const char *names[] = {"ijk_seq", "ijk_omp", "ikj_seq", "ikj_omp", "blocked_omp"};
    Func funcs[] = {ijk_seq, ijk_omp, ikj_seq, ikj_omp, blocked_omp};
    double t[5];

    printf("N=%d block=%d threads=%d repeats=%d\n", n, bs, threads, reps);
    for (int v = 0; v < 5; v++) {
        t[v] = measure(funcs[v], n, A, B, C, bs, reps);
        if (v == 0)
            for (int i = 0; i < n * n; i++) Cref[i] = C[i];
        double diff = 0;
        for (int i = 0; i < n * n; i++) {
            double d = fabs(C[i] - Cref[i]);
            if (d > diff) diff = d;
        }
        double s = 0;
        if (v == 1) s = t[0] / t[1];
        if (v == 3) s = t[2] / t[3];
        if (v == 4) s = t[2] / t[4];
        printf("%-12s %8.4f s", names[v], t[v]);
        if (s > 0) printf("  S=%.2f E=%.2f", s, s / threads);
        printf("  diff=%.1e %s\n", diff, diff < 1e-9 ? "OK" : "ERROR");
    }

    free(A);
    free(B);
    free(C);
    free(Cref);
    return 0;
}