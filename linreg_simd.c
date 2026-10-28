#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define REPS 3
#define TOL 1e-8
#define EPS 1e-6
#define ETA 0.1
#define LAMBDA 0.001
#define K 100

static unsigned int state = 12345;

double rnd(void) {
    state = state * 1664525u + 1013904223u;
    return ((state >> 8) / 8388608.0) - 1.0;
}

double soft(double z, double a) {
    if (z > a) return z - a;
    if (z < -a) return z + a;
    return 0.0;
}

void update(double *w, double *b, double *gw, double gb, int N, int d, double eta, double lambda) {
    for (int j = 0; j < d; j++)
        w[j] = soft(w[j] - eta * gw[j] / N, eta * lambda);
    *b -= eta * gb / N;
}

double loss(double *X, double *y, double *w, double b, int N, int d, double lambda) {
    double s = 0.0, reg = 0.0;
    for (int i = 0; i < N; i++) {
        double p = b;
        for (int j = 0; j < d; j++) p += X[(long)i * d + j] * w[j];
        s += (p - y[i]) * (p - y[i]);
    }
    for (int j = 0; j < d; j++) reg += fabs(w[j]);
    return s / (2.0 * N) + lambda * reg;
}

void train_serial(double *X, double *y, double *w, double *b, int N, int d, double eta, double lambda, int iters) {
    double *gw = malloc(d * sizeof(double));
    for (int k = 0; k < iters; k++) {
        for (int j = 0; j < d; j++) gw[j] = 0.0;
        double gb = 0.0;
        for (int i = 0; i < N; i++) {
            double *row = X + (long)i * d;
            double p = *b;
            for (int j = 0; j < d; j++) p += row[j] * w[j];
            double e = p - y[i];
            gb += e;
            for (int j = 0; j < d; j++) gw[j] += e * row[j];
        }
        update(w, b, gw, gb, N, d, eta, lambda);
    }
    free(gw);
}

void train_omp(double *X, double *y, double *w, double *b, int N, int d, double eta, double lambda, int iters) {
    double *gw = malloc(d * sizeof(double));
    for (int k = 0; k < iters; k++) {
        for (int j = 0; j < d; j++) gw[j] = 0.0;
        double gb = 0.0;
        #pragma omp parallel
        {
            double *lg = calloc(d, sizeof(double));
            double lb = 0.0;
            #pragma omp for
            for (int i = 0; i < N; i++) {
                double *row = X + (long)i * d;
                double p = *b;
                for (int j = 0; j < d; j++) p += row[j] * w[j];
                double e = p - y[i];
                lb += e;
                for (int j = 0; j < d; j++) lg[j] += e * row[j];
            }
            #pragma omp critical
            {
                for (int j = 0; j < d; j++) gw[j] += lg[j];
                gb += lb;
            }
            free(lg);
        }
        update(w, b, gw, gb, N, d, eta, lambda);
    }
    free(gw);
}

void train_simd(double *X, double *y, double *w, double *b, int N, int d, double eta, double lambda, int iters) {
    double *gw = malloc(d * sizeof(double));
    for (int k = 0; k < iters; k++) {
        for (int j = 0; j < d; j++) gw[j] = 0.0;
        double gb = 0.0;
        for (int i = 0; i < N; i++) {
            double *row = X + (long)i * d;
            double p = *b;
            #pragma omp simd reduction(+:p)
            for (int j = 0; j < d; j++) p += row[j] * w[j];
            double e = p - y[i];
            gb += e;
            #pragma omp simd
            for (int j = 0; j < d; j++) gw[j] += e * row[j];
        }
        update(w, b, gw, gb, N, d, eta, lambda);
    }
    free(gw);
}

void train_omp_simd(double *X, double *y, double *w, double *b, int N, int d, double eta, double lambda, int iters) {
    double *gw = malloc(d * sizeof(double));
    for (int k = 0; k < iters; k++) {
        for (int j = 0; j < d; j++) gw[j] = 0.0;
        double gb = 0.0;
        #pragma omp parallel
        {
            double *lg = calloc(d, sizeof(double));
            double lb = 0.0;
            #pragma omp for
            for (int i = 0; i < N; i++) {
                double *row = X + (long)i * d;
                double p = *b;
                #pragma omp simd reduction(+:p)
                for (int j = 0; j < d; j++) p += row[j] * w[j];
                double e = p - y[i];
                lb += e;
                #pragma omp simd
                for (int j = 0; j < d; j++) lg[j] += e * row[j];
            }
            #pragma omp critical
            {
                for (int j = 0; j < d; j++) gw[j] += lg[j];
                gb += lb;
            }
            free(lg);
        }
        update(w, b, gw, gb, N, d, eta, lambda);
    }
    free(gw);
}

void experiment(int N, int d, double lambda, int iters, int threads, double *t, double *f, double *dw, int *zeros) {
    omp_set_num_threads(threads);
    state = 12345;

    double *X = malloc((long)N * d * sizeof(double));
    double *y = malloc(N * sizeof(double));
    double *wt = malloc(d * sizeof(double));
    for (int j = 0; j < d; j++) wt[j] = (j % 3 == 0) ? 1.0 + 0.1 * j : 0.0;
    for (long i = 0; i < (long)N * d; i++) X[i] = rnd();
    for (int i = 0; i < N; i++) {
        double s = 0.5;
        for (int j = 0; j < d; j++) s += X[(long)i * d + j] * wt[j];
        y[i] = s + 0.1 * rnd();
    }

    double *w[4];
    double b[4];
    for (int v = 0; v < 4; v++) {
        w[v] = calloc(d, sizeof(double));
        b[v] = 0.0;
    }

    for (int v = 0; v < 4; v++) {
        double t0 = omp_get_wtime();
        if (v == 0) train_serial(X, y, w[v], &b[v], N, d, ETA, lambda, iters);
        if (v == 1) train_omp(X, y, w[v], &b[v], N, d, ETA, lambda, iters);
        if (v == 2) train_simd(X, y, w[v], &b[v], N, d, ETA, lambda, iters);
        if (v == 3) train_omp_simd(X, y, w[v], &b[v], N, d, ETA, lambda, iters);
        t[v] = omp_get_wtime() - t0;
    }

    for (int v = 0; v < 4; v++) {
        f[v] = loss(X, y, w[v], b[v], N, d, lambda);
        dw[v] = 0.0;
        for (int j = 0; j < d; j++) {
            double diff = fabs(w[v][j] - w[0][j]);
            if (diff > dw[v]) dw[v] = diff;
        }
    }

    *zeros = 0;
    for (int j = 0; j < d; j++)
        if (fabs(w[0][j]) < EPS) (*zeros)++;

    for (int v = 0; v < 4; v++) free(w[v]);
    free(X);
    free(y);
    free(wt);
}

void run(int N, int d, double lambda, int iters, int threads, double *t, double *f, double *dw, int *zeros) {
    double all[REPS][4];
    for (int r = 0; r < REPS; r++)
        experiment(N, d, lambda, iters, threads, all[r], f, dw, zeros);
    for (int v = 0; v < 4; v++) {
        double a[REPS];
        for (int r = 0; r < REPS; r++) a[r] = all[r][v];
        for (int i = 1; i < REPS; i++)
            for (int j = i; j > 0 && a[j - 1] > a[j]; j--) {
                double tmp = a[j];
                a[j] = a[j - 1];
                a[j - 1] = tmp;
            }
        t[v] = a[REPS / 2];
    }
}

int main(void) {
    int P = 0;
    printf("threads: ");
    if (scanf("%d", &P) != 1 || P < 1) return 1;

    const char *names[4] = {"serial", "openmp", "simd", "openmp+simd"};
    double t[4], f[4], dw[4];
    int z;

    printf("\neta=%g K=%d repeats=%d (median time) threads=%d\n", ETA, K, REPS, P);

    {
        int N = 1000, d = 8;
        run(N, d, LAMBDA, K, P, t, f, dw, &z);
        printf("\n=== 3.1 correctness: N=%d d=%d lambda=%g threads=%d, tolerance %g ===\n", N, d, LAMBDA, P, TOL);
        printf("%-12s %14s %12s %6s\n", "variant", "loss", "dw", "ok");
        for (int v = 0; v < 4; v++)
            printf("%-12s %14.8f %12.3e %6s\n", names[v], f[v], dw[v], dw[v] < TOL ? "yes" : "no");
    }

    {
        int N = 100000, d = 32;
        run(N, d, LAMBDA, K, P, t, f, dw, &z);
        printf("\n=== 3.2 comparison: N=%d d=%d lambda=%g threads=%d ===\n", N, d, LAMBDA, P);
        printf("%-12s %10s %10s\n", "variant", "time, s", "speedup");
        for (int v = 0; v < 4; v++)
            printf("%-12s %10.4f %10.2f\n", names[v], t[v], t[0] / t[v]);
        fflush(stdout);
    }

    {
        int Ns[3] = {10000, 100000, 1000000};
        int d = 32;
        printf("\n=== 3.3 effect of N: d=%d lambda=%g threads=%d ===\n", d, LAMBDA, P);
        printf("%-10s %10s %10s %10s %12s\n", "N", "serial", "openmp", "simd", "openmp+simd");
        for (int k = 0; k < 3; k++) {
            run(Ns[k], d, LAMBDA, K, P, t, f, dw, &z);
            printf("%-10d %10.4f %10.4f %10.4f %12.4f\n", Ns[k], t[0], t[1], t[2], t[3]);
        }
    }

    {
        int N = 100000;
        int ds[4] = {8, 32, 128, 256};
        printf("\n=== 3.4 effect of d: N=%d lambda=%g threads=%d ===\n", N, LAMBDA, P);
        printf("%-10s %10s %10s %10s %12s\n", "d", "serial", "openmp", "simd", "openmp+simd");
        for (int k = 0; k < 4; k++) {
            run(N, ds[k], LAMBDA, K, P, t, f, dw, &z);
            printf("%-10d %10.4f %10.4f %10.4f %12.4f\n", ds[k], t[0], t[1], t[2], t[3]);
        }
    }

    {
        int N = 100000, d = 32;
        double b1 = 0.0, b3 = 0.0;
        int p = 1;
        printf("\n=== 3.5 effect of threads: N=%d d=%d lambda=%g ===\n", N, d, LAMBDA);
        printf("%-8s %10s %8s %8s %14s %8s %8s\n", "threads", "openmp", "S", "E", "openmp+simd", "S", "E");
        while (1) {
            run(N, d, LAMBDA, K, p, t, f, dw, &z);
            if (p == 1) {
                b1 = t[1];
                b3 = t[3];
            }
            printf("%-8d %10.4f %8.2f %8.2f %14.4f %8.2f %8.2f\n",
                   p, t[1], b1 / t[1], b1 / t[1] / p, t[3], b3 / t[3], b3 / t[3] / p);
            if (p == P) break;
            p *= 2;
            if (p > P) p = P;
        }
    }

    {
        int N = 100000, d = 32, K6 = 500;
        double ls[5] = {0.0, 1e-4, 1e-3, 1e-2, 1e-1};
        printf("\n=== 3.6 effect of lambda: N=%d d=%d K=%d threads=%d, eps=%g ===\n", N, d, K6, P, EPS);
        printf("%-10s %14s %12s %12s %14s\n", "lambda", "loss", "zeros", "serial, s", "openmp+simd, s");
        for (int k = 0; k < 5; k++) {
            run(N, d, ls[k], K6, P, t, f, dw, &z);
            printf("%-10g %14.8f %9d/%d %12.4f %14.4f\n", ls[k], f[0], z, d, t[0], t[3]);
        }
    }

    return 0;
}