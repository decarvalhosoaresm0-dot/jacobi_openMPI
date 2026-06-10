#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <mpi.h>

#define N 1000
#define SEED 42

// functions declarations
int jacobi_sequential(int n, double A[n][n], double b[n], double x[n], int max_iter, double tol, double *final_error);

int jacobi_parallel_mpi(int n, double A[n][n], double b[n], double x[n], int max_iter, double tol, double *final_error);

// generates a system
void generate_system(int n, double A[n][n], double b[n], double x[n]) {
    srand(SEED);

    for(int i = 0; i < n; i++){

        double row_sum = 0.0;

        for(int j = 0; j < n; j++){

            if(i != j){
                A[i][j] = ((double) rand() / RAND_MAX) * 10.0;
                row_sum += fabs(A[i][j]);
            }
        }

        A[i][i] = row_sum + 10.0;

        b[i] = 1.0;
        x[i] = 0.0;
    }
}

// prints the system with a limit of N <= 5
void print_system(int n, double A[n][n], double b[n]) {

    int limit = (n < 5) ? n : 5;

    printf("\nSystem Ax = b (showing first %d rows):\n\n", limit);

    for(int i = 0; i < limit; i++){

        // matrix A
        printf("[ ");
        for(int j = 0; j < limit; j++){
            printf("%6.2f ", A[i][j]);
        }

        if(n > limit) printf("... ");
        printf("] ");

        // xN
        printf("[x%d]", i + 1);

        // sinal =
        if(i == limit / 2)
            printf(" = ");
        else
            printf("   ");

        // vector b
        printf("[ %6.2f ]", b[i]);

        printf("\n");
    }

    // cut indication
    if(n > limit){
        printf("  ...\n");
    }

    printf("\n");
}

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    static double A[N][N];
    static double b[N];
    static double x_seq[N];
    static double x_par[N];

    int max_iter = 5000;
    double tol = 1e-3;

    double error_seq = 0.0;
    double error_par = 0.0;

    // apenas o processo 0 gera o sistema
    if(rank == 0){

        printf("Number of processes: %d\n\n", size);

        generate_system(N, A, b, x_seq);

        // copia x inicial para o paralelo
        for(int i = 0; i < N; i++){
            x_par[i] = x_seq[i];
        }

        print_system(N, A, b);
    }

    // compartilha os dados com todos os processos
    MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(b, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(x_par, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    // ----------------- SEQUENTIAL -----------------
    double time_seq = 0.0;

    int iter_seq = 0;

    if(rank == 0){

        double start = MPI_Wtime();

        iter_seq = jacobi_sequential(N, A, b, x_seq, max_iter, tol, &error_seq);

        time_seq = MPI_Wtime() - start;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // ----------------- MPI Parallel -----------------
    double start = MPI_Wtime();

    int iter_par = jacobi_parallel_mpi(N, A, b, x_par, max_iter, tol, &error_par);

    double time_par = MPI_Wtime() - start;

    // ------------------ RESULTS -----------------
    if(rank == 0){

        double speedup = time_seq / time_par;

        printf("=== Sequential ===\n");
        printf("Iterations: %d\n", iter_seq);
        printf("Final error: %e\n", error_seq);
        printf("Time: %f seconds\n\n", time_seq);
        printf("=== MPI Parallel ===\n");
        printf("Iterations: %d\n", iter_par);
        printf("Final error: %e\n", error_par);
        printf("Time: %f seconds\n\n", time_par);
        printf("Speedup: %f\n", speedup);
        printf("Efficiency: %f\n", speedup / size);
    }

    MPI_Finalize();

    return 0;
}