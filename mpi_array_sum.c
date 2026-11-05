#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <errno.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, processes, n = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    if (rank == 0) {
        if (argc == 2) {
            char *end;
            errno = 0;
            long value = strtol(argv[1], &end, 10);
            if (errno == 0 && end != argv[1] && *end == '\0' &&
                value > 0 && value <= INT_MAX && value % processes == 0) {
                n = (int)value;
            }
        }
        if (n < 0) {
            fprintf(stderr, "Usage: mpiexec -n P ./mpi_array_sum N (N > 0 and N divisible by P)\n");
        }
    }

    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (n < 0) {
        MPI_Finalize();
        return 1;
    }

    int count = n / processes;
    int *array = NULL;
    int *part = malloc((size_t)count * sizeof(int));
    long long *local_sums = NULL;

    if (rank == 0) {
        array = malloc((size_t)n * sizeof(int));
        local_sums = malloc((size_t)processes * sizeof(long long));
    }

    int allocated = part != NULL && (rank != 0 || (array != NULL && local_sums != NULL));
    int all_allocated;
    MPI_Allreduce(&allocated, &all_allocated, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (!all_allocated) {
        if (rank == 0) fprintf(stderr, "Memory allocation failed\n");
        free(array);
        free(part);
        free(local_sums);
        MPI_Finalize();
        return 1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    if (rank == 0) {
        for (int i = 0; i < n; i++) array[i] = i + 1;
    }

    MPI_Scatter(array, count, MPI_INT, part, count, MPI_INT, 0, MPI_COMM_WORLD);

    long long local_sum = 0;
    for (int i = 0; i < count; i++) local_sum += part[i];

    MPI_Gather(&local_sum, 1, MPI_LONG_LONG_INT,
               local_sums, 1, MPI_LONG_LONG_INT, 0, MPI_COMM_WORLD);

    long long gather_sum = 0;
    if (rank == 0) {
        for (int i = 0; i < processes; i++) gather_sum += local_sums[i];
    }

    long long reduce_sum = 0;
    MPI_Reduce(&local_sum, &reduce_sum, 1, MPI_LONG_LONG_INT,
               MPI_SUM, 0, MPI_COMM_WORLD);

    long long allreduce_sum = 0;
    MPI_Allreduce(&local_sum, &allreduce_sum, 1, MPI_LONG_LONG_INT,
                  MPI_SUM, MPI_COMM_WORLD);

    double elapsed = MPI_Wtime() - start;
    double max_time = 0;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long expected = (long long)n * (n + 1LL) / 2;
        int correct = gather_sum == expected && reduce_sum == expected &&
                      allreduce_sum == expected;
        printf("P=%d N=%d Expected=%lld Gather=%lld Reduce=%lld Allreduce=%lld Correct=%s Time=%.9f s\n",
               processes, n, expected, gather_sum, reduce_sum, allreduce_sum,
               correct ? "YES" : "NO", max_time);
    }

    free(array);
    free(part);
    free(local_sums);
    MPI_Finalize();
    return 0;
}