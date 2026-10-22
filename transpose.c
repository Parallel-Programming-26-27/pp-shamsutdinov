#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

typedef double T;

enum { ROWS, COLS, BLOCKED, ROWS_OMP, BLOCKED_OMP };

T *A, *B, *Bref;
int cur_N = 0;
long total_errors = 0;

void transpose_rows(T *A, T *B, int N)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            B[j * N + i] = A[i * N + j];
}

void transpose_cols(T *A, T *B, int N)
{
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++)
            B[j * N + i] = A[i * N + j];
}

void transpose_blocked(T *A, T *B, int N, int b)
{
    for (int ii = 0; ii < N; ii += b)
        for (int jj = 0; jj < N; jj += b)
            for (int i = ii; i < ii + b && i < N; i++)
                for (int j = jj; j < jj + b && j < N; j++)
                    B[j * N + i] = A[i * N + j];
}

void transpose_rows_omp(T *A, T *B, int N, int chunk)
{
    if (chunk == 0) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                B[j * N + i] = A[i * N + j];
    } else {
        #pragma omp parallel for schedule(static, chunk)
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                B[j * N + i] = A[i * N + j];
    }
}

void transpose_blocked_omp(T *A, T *B, int N, int b)
{
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += b)
        for (int jj = 0; jj < N; jj += b)
            for (int i = ii; i < ii + b && i < N; i++)
                for (int j = jj; j < jj + b && j < N; j++)
                    B[j * N + i] = A[i * N + j];
}

void run(int ver, int N, int b, int chunk)
{
    if (ver == ROWS)              transpose_rows(A, B, N);
    else if (ver == COLS)         transpose_cols(A, B, N);
    else if (ver == BLOCKED)      transpose_blocked(A, B, N, b);
    else if (ver == ROWS_OMP)     transpose_rows_omp(A, B, N, chunk);
    else if (ver == BLOCKED_OMP)  transpose_blocked_omp(A, B, N, b);
}

double bench(int ver, int N, int b, int threads, int chunk)
{
    size_t size = (size_t)N * N;

    if (N != cur_N) {
        for (size_t k = 0; k < size; k++)
            A[k] = (T)(rand() % 1000);
        transpose_rows(A, Bref, N);
        cur_N = N;
    }

    omp_set_num_threads(threads);
    run(ver, N, b, chunk);

    int reps = (int)(30000000L / ((long)N * N)) + 1;

    double t[5];
    for (int r = 0; r < 5; r++) {
        memset(B, 0, size * sizeof(T));
        double t0 = omp_get_wtime();
        for (int q = 0; q < reps; q++)
            run(ver, N, b, chunk);
        double t1 = omp_get_wtime();
        t[r] = (t1 - t0) / reps;
        for (size_t k = 0; k < size; k++)
            if (B[k] != Bref[k])
                total_errors++;
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4 - i; j++)
            if (t[j] > t[j + 1]) {
                double tmp = t[j];
                t[j] = t[j + 1];
                t[j + 1] = tmp;
            }
    return t[2];
}

double bandwidth(int N, double t)
{
    return 2.0 * N * N * sizeof(T) / t / 1e9;
}

int main()
{
    int Nv = 3000;

    int maxN = Nv > 2048 ? Nv : 2048;
    size_t size = (size_t)maxN * maxN;
    A = malloc(size * sizeof(T));
    B = malloc(size * sizeof(T));
    Bref = malloc(size * sizeof(T));
    if (A == NULL || B == NULL || Bref == NULL) {
        printf("Not enough memory\n");
        return 1;
    }

    int sizes[5] = {256, 512, 1024, 2048, Nv};
    int blocks[6] = {4, 8, 16, 32, 64, 128};
    int thr[5] = {1, 2, 4, 8, 16};
    int chunks[3] = {0, 1, 8};
    const char *chunk_names[3] = {"schedule(static)", "schedule(static, 1)", "schedule(static, 8)"};

    printf("\nExperiment 1: traversal order, 1 thread\n");
    printf("%-8s %-14s %-14s %-16s %-16s\n", "N", "rows, s", "cols, s", "BW rows, GB/s", "BW cols, GB/s");
    double T1 = 0;
    for (int i = 0; i < 5; i++) {
        double tr = bench(ROWS, sizes[i], 32, 1, 0);
        double tc = bench(COLS, sizes[i], 32, 1, 0);
        printf("%-8d %-14.6f %-14.6f %-16.2f %-16.2f\n", sizes[i], tr, tc,
               bandwidth(sizes[i], tr), bandwidth(sizes[i], tc));
        if (i == 4)
            T1 = tr;
    }

    printf("\nExperiment 2: block size, 1 thread, N=%d\n", Nv);
    printf("%-8s %-14s %-16s\n", "b", "time, s", "BW, GB/s");
    int bstar = blocks[0];
    double best = 1e9;
    for (int i = 0; i < 6; i++) {
        double t = bench(BLOCKED, Nv, blocks[i], 1, 0);
        printf("%-8d %-14.6f %-16.2f\n", blocks[i], t, bandwidth(Nv, t));
        if (t < best) {
            best = t;
            bstar = blocks[i];
        }
    }
    printf("Best block size b* = %d\n", bstar);

    printf("\nExperiment 3: threads, N=%d, b=%d, T1 (rows, 1 thread) = %.6f s\n", Nv, bstar, T1);
    printf("%-8s %-18s %-14s %-20s %-14s\n", "Threads", "rows_omp, s", "S rows_omp", "blocked_omp, s", "S blocked_omp");
    for (int i = 0; i < 5; i++) {
        double tr = bench(ROWS_OMP, Nv, bstar, thr[i], 0);
        double tb = bench(BLOCKED_OMP, Nv, bstar, thr[i], 0);
        printf("%-8d %-18.6f %-14.2f %-20.6f %-14.2f\n", thr[i], tr, T1 / tr, tb, T1 / tb);
    }

    printf("\nExperiment 4: false sharing, 8 threads, N=%d\n", Nv);
    printf("%-24s %-14s %-16s\n", "Strategy", "time, s", "BW, GB/s");
    for (int i = 0; i < 3; i++) {
        double t = bench(ROWS_OMP, Nv, bstar, 8, chunks[i]);
        printf("%-24s %-14.6f %-16.2f\n", chunk_names[i], t, bandwidth(Nv, t));
    }

    printf("\nCheck: %s (mismatches: %ld)\n", total_errors == 0 ? "OK" : "FAIL", total_errors);

    free(A);
    free(B);
    free(Bref);
    return 0;
}