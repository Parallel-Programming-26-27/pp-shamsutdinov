
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define PI 3.14159265358979323846
#define REPEATS 5
#define MAX_N 1000000000LL

static volatile double g_sink = 0.0;

double f(double x)
{
    double c = cos(x);
    double s = sin(x);

    return c * c * c * s * s;
}

double exact_integral(void)
{
    return 11.0 * sqrt(3.0) / 160.0;
}

double sequential(double a, double b, long long n)
{
    double h = (b - a) / (double)n;
    double sum = 0.0;

    for (long long i = 0; i < n; i++)
    {
        double x = a + ((double)i + 0.5) * h;
        sum += f(x);
    }

    g_sink = sum;
    return sum * h;
}

double parallel(double a, double b, long long n, int threads)
{
    double h = (b - a) / (double)n;
    double sum = 0.0;

    omp_set_num_threads(threads);

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; i++)
    {
        double x = a + ((double)i + 0.5) * h;
        sum += f(x);
    }

    g_sink = sum;
    return sum * h;
}

double average_sequential_time(
    double a, double b, long long n, int repeats)
{
    double total = 0.0;

    for (int i = 0; i < repeats; i++)
    {
        double start = omp_get_wtime();
        double result = sequential(a, b, n);
        total += omp_get_wtime() - start;

        g_sink = result;
    }

    return total / repeats;
}

double average_parallel_time(
    double a, double b, long long n,
    int threads, int repeats)
{
    double total = 0.0;

    for (int i = 0; i < repeats; i++)
    {
        double start = omp_get_wtime();
        double result = parallel(a, b, n, threads);
        total += omp_get_wtime() - start;

        g_sink = result;
    }

    return total / repeats;
}

void experiment_integral(double a, double b)
{
    long long n;
    int threads;

    printf("Enter n: ");
    if (scanf("%lld", &n) != 1 || n <= 0)
    {
        printf("Incorrect n.\n");
        return;
    }

    printf("Enter number of threads: ");
    if (scanf("%d", &threads) != 1 || threads <= 0)
    {
        printf("Incorrect number of threads.\n");
        return;
    }

    double start = omp_get_wtime();
    double result_seq = sequential(a, b, n);
    double time_seq = omp_get_wtime() - start;

    start = omp_get_wtime();
    double result_par = parallel(a, b, n, threads);
    double time_par = omp_get_wtime() - start;

    printf("\n--- Integral calculation ---\n");
    printf("Exact integral:       %.12f\n", exact_integral());
    printf("Sequential result:    %.12f\n", result_seq);
    printf("Sequential time:      %.6f sec\n", time_seq);
    printf("Sequential abs error: %.12e\n",
           fabs(result_seq - exact_integral()));

    printf("\nParallel result:      %.12f\n", result_par);
    printf("Parallel time:        %.6f sec\n", time_par);
    printf("Parallel abs error:   %.12e\n",
           fabs(result_par - exact_integral()));
    printf("Threads:              %d\n", threads);

    if (time_par > 0.0)
        printf("Speedup:              %.3f\n",
               time_seq / time_par);
}

void experiment_threads(double a, double b)
{
    const long long n = 100000000LL;
    const int thread_values[] = {1, 2, 4, 8, 16};
    const int count = sizeof(thread_values) / sizeof(thread_values[0]);

    printf("\n--- Experiment 2: number of threads ---\n");
    printf("n = %lld\n", n);
    printf("Repeats per measurement: %d\n", REPEATS);

    double seq_result = sequential(a, b, n);
    double seq_time = average_sequential_time(a, b, n, REPEATS);

    printf("Sequential result: %.12f\n", seq_result);
    printf("Sequential average time: %.6f sec\n", seq_time);
    printf("Threads  Avg time (sec)  Speedup  Result\n");

    for (int i = 0; i < count; i++)
    {
        int threads = thread_values[i];

        parallel(a, b, n, threads);

        double par_time = average_parallel_time(
            a, b, n, threads, REPEATS);

        double par_result = parallel(a, b, n, threads);
        double speedup = seq_time / par_time;

        printf("%7d  %15.6f  %7.3f  %.12f\n",
               threads, par_time, speedup, par_result);
    }
}

void experiment_n(double a, double b)
{
    const long long values[] = {
        10000LL,
        100000LL,
        1000000LL,
        10000000LL,
        100000000LL
    };

    const int count = sizeof(values) / sizeof(values[0]);
    const int threads = 8;
    double exact = exact_integral();

    printf("\n--- Experiment 3: number of subdivisions ---\n");
    printf("Threads: %d\n", threads);
    printf("Repeats per measurement: %d\n", REPEATS);
    printf("Exact integral: %.12f\n\n", exact);

    printf("%12s  %15s  %15s  %15s\n",
           "n", "Integral", "Abs error", "Avg time (sec)");

    for (int i = 0; i < count; i++)
    {
        long long n = values[i];

        parallel(a, b, n, threads);

        double avg_time = average_parallel_time(
            a, b, n, threads, REPEATS);

        double result = parallel(a, b, n, threads);
        double abs_error = fabs(result - exact);

        printf("%12lld  %15.10f  %15.6e  %15.6f\n",
               n, result, abs_error, avg_time);
    }
}

void experiment_accuracy(double a, double b)
{
    double eps;
    int mode;
    int threads = 8;

    printf("Enter accuracy (for example 0.000001): ");

    if (scanf("%lf", &eps) != 1 ||
        !isfinite(eps) || eps <= 0.0)
    {
        printf("Incorrect accuracy.\n");
        return;
    }

    printf("1 - Sequential, 2 - Parallel: ");

    if (scanf("%d", &mode) != 1 ||
        (mode != 1 && mode != 2))
    {
        printf("Incorrect mode.\n");
        return;
    }

    long long n = 1000;
    double previous;
    double current = 0.0;
    double error_estimate = INFINITY;
    int converged = 0;

    double start = omp_get_wtime();

    if (mode == 1)
        previous = sequential(a, b, n);
    else
        previous = parallel(a, b, n, threads);

    while (n <= MAX_N / 2)
    {
        n *= 2;

        if (mode == 1)
            current = sequential(a, b, n);
        else
            current = parallel(a, b, n, threads);

        error_estimate = fabs(current - previous) / 3.0;

        if (error_estimate < eps)
        {
            converged = 1;
            break;
        }

        previous = current;
    }

    double elapsed = omp_get_wtime() - start;

    printf("\n--- Calculation with accuracy ---\n");
    printf("Method: %s\n",
           mode == 1 ? "Sequential" : "Parallel");
    printf("Integral: %.12f\n", current);
    printf("Exact integral: %.12f\n", exact_integral());
    printf("Actual absolute error: %.12e\n",
           fabs(current - exact_integral()));
    printf("Estimated error: %.12e\n", error_estimate);
    printf("Requested accuracy: %.12e\n", eps);
    printf("n: %lld\n", n);
    printf("Time: %.6f sec\n", elapsed);

    if (converged)
        printf("Requested estimated accuracy reached.\n");
    else
        printf("Maximum n reached; requested accuracy was not confirmed.\n");
}

int main(void)
{
    const double a = 0.0;
    const double b = PI / 3.0;
    int choice;

    omp_set_dynamic(0);

    printf("Variant H\n");
    printf("Integral: integral from 0 to PI/3 of cos^3(x)*sin^2(x) dx\n");
    printf("1 - Calculate integral\n");
    printf("2 - Experiment with threads\n");
    printf("3 - Experiment with n\n");
    printf("4 - Calculate with accuracy\n");
    printf("Your choice: ");

    if (scanf("%d", &choice) != 1)
    {
        printf("Incorrect input.\n");
        return 1;
    }

    switch (choice)
    {
        case 1:
            experiment_integral(a, b);
            break;

        case 2:
            experiment_threads(a, b);
            break;

        case 3:
            experiment_n(a, b);
            break;

        case 4:
            experiment_accuracy(a, b);
            break;

        default:
            printf("Incorrect choice.\n");
            return 1;
    }

    return 0;
}
