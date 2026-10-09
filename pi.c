#include <stdio.h>
#include <math.h>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double pi_serial(long long n)
{
    double sum = 0.0;
    for (long long i = 0; i < n; i++) {
        if (i % 2 == 0)
            sum += 1.0 / (2 * i + 1);
        else
            sum -= 1.0 / (2 * i + 1);
    }
    return 4.0 * sum;
}

double pi_parallel(long long n)
{
    double sum = 0.0;
#pragma omp parallel for reduction(+:sum)
    for (long long i = 0; i < n; i++) {
        if (i % 2 == 0)
            sum += 1.0 / (2 * i + 1);
        else
            sum -= 1.0 / (2 * i + 1);
    }
    return 4.0 * sum;
}

int main()
{
    int thread_list[5] = {1, 2, 4, 8, 16};
    long long n_list[4] = {10000000LL, 100000000LL, 1000000000LL, 10000000000LL};

    long long n = 1000000000LL;

    printf("EXPERIMENT 1: n = %lld, different number of threads\n\n", n);

    double start = omp_get_wtime();
    double pi = pi_serial(n);
    double time_serial = omp_get_wtime() - start;

    printf("Serial: pi = %.15f, error = %.3e, time = %.4f s\n\n", pi, fabs(pi - M_PI), time_serial);
    printf("threads | pi                | error     | time, s  | speedup\n");

    for (int k = 0; k < 5; k++) {
        omp_set_num_threads(thread_list[k]);

        start = omp_get_wtime();
        pi = pi_parallel(n);
        double time_parallel = omp_get_wtime() - start;

        printf("%7d | %.15f | %.3e | %8.4f | %6.2f\n",
               thread_list[k], pi, fabs(pi - M_PI), time_parallel, time_serial / time_parallel);
    }

    printf("\nEXPERIMENT 2: 8 threads, different n\n\n");
    printf("n           | serial, s | OpenMP, s | speedup | error (serial) | error (OpenMP)\n");

    omp_set_num_threads(8);

    for (int k = 0; k < 4; k++) {
        n = n_list[k];

        start = omp_get_wtime();
        double pi1 = pi_serial(n);
        double t1 = omp_get_wtime() - start;

        start = omp_get_wtime();
        double pi2 = pi_parallel(n);
        double t2 = omp_get_wtime() - start;

        printf("%11lld | %9.4f | %9.4f | %7.2f | %.3e      | %.3e\n",
               n, t1, t2, t1 / t2, fabs(pi1 - M_PI), fabs(pi2 - M_PI));
    }

    return 0;
}