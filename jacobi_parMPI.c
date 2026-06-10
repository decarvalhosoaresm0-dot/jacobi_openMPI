#include <math.h>
#include <mpi.h>

int jacobi_parallel_mpi(
    int n,
    double A[n][n],
    double b[n],
    double x[n],
    int max_iter,
    double tol,
    double *final_error
) {

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // quantidade de linhas por processo
    int rows_per_process = n / size;

    // intervalo de linhas deste processo
    int start = rank * rows_per_process;

    int end = start + rows_per_process;

    // último processo pega linhas restantes
    if(rank == size - 1){
        end = n;
    }

    double x_old[n];

    // copia inicial
    for(int i = 0; i < n; i++){
        x_old[i] = x[i];
    }

    // vetor local
    double local_x[end - start];

    for(int k = 0; k < max_iter; k++){

        double local_error = 0.0;

        // =========================
        // cálculo local
        // =========================

        for(int i = start; i < end; i++){

            double soma = 0.0;

            for(int j = 0; j < n; j++){

                if(i != j){
                    soma += A[i][j] * x_old[j];
                }
            }

            local_x[i - start] =
                (b[i] - soma) / A[i][i];

            double diferenca =
                local_x[i - start] - x_old[i];

            local_error +=
                diferenca * diferenca;
        }

        // =========================
        // compartilhar resultados
        // =========================

        /*
            TODO:
            usar MPI_Allgather
            para montar o vetor x completo
        */

        // =========================
        // erro global
        // =========================

        double global_error = 0.0;

        /*
            TODO:
            usar MPI_Allreduce
            para somar os erros locais
        */

        global_error = sqrt(global_error);

        *final_error = global_error;

        // verifica convergência
        if(global_error < tol){
            return k + 1;
        }

        // atualiza x_old
        for(int i = 0; i < n; i++){
            x_old[i] = x[i];
        }
    }

    return max_iter;
}