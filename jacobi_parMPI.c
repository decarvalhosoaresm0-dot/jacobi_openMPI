#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>

#define NUM_CHECKPOINTS 5

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

    // quantidade real de linhas deste processo
    int local_n = end - start;

    /*
        Vetores auxiliares para o MPI_Allgatherv.

        recvcounts[p] indica quantos valores o processo p envia.
        displs[p] indica em qual posição do vetor x esses valores começam.
    */
    int recvcounts[size];
    int displs[size];

    for(int p = 0; p < size; p++){

        int p_start = p * rows_per_process;
        int p_end = p_start + rows_per_process;

        if(p == size - 1){
            p_end = n;
        }

        recvcounts[p] = p_end - p_start;
        displs[p] = p_start;
    }

    double x_old[n];

    // copia inicial
    for(int i = 0; i < n; i++){
        x_old[i] = x[i];
    }

    // vetor local --> local_n = end - start;
    double local_x[local_n > 0 ? local_n : 1];

    /*
        Vetor apenas para guardar os erros exigidos no relatório.
        O template pede a norma do erro nas iterações:
        0, 50, 100, 200 e 300.
    */
    int checkpoints[5] = {0, 50, 100, 200, 300};
    double checkpoint_errors[5];

    for(int i = 0; i < 5; i++){
        checkpoint_errors[i] = -1.0;
    }

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

            local_x[i - start] = (b[i] - soma) / A[i][i];

            double diferenca = local_x[i - start] - x_old[i];

            local_error += diferenca * diferenca;
        }

        // =========================
        // compartilhar resultados
        // =========================

        /*
            Cada processo calculou apenas uma parte de x.

            Agora precisamos montar o vetor x completo em todos os processos,
            pois a próxima iteração do Jacobi precisa de todos os valores de x_old.

            Foi usado MPI_Allgatherv em vez de MPI_Allgather porque o último
            processo pode ter uma quantidade diferente de elementos.
        */
        MPI_Allgatherv(
            local_x,
            local_n,
            MPI_DOUBLE,
            x,
            recvcounts,
            displs,
            MPI_DOUBLE,
            MPI_COMM_WORLD
        );

        // =========================
        // erro global
        // =========================

        double global_error = 0.0;

        /*
            Cada processo calculou apenas o erro da sua parte do vetor.

            Agora somamos os erros locais de todos os processos.
            O resultado fica disponível em todos os processos.
        */
        MPI_Allreduce(
            &local_error,
            &global_error,
            1,
            MPI_DOUBLE,
            MPI_SUM,
            MPI_COMM_WORLD
        );

        global_error = sqrt(global_error);

        *final_error = global_error;

        // guarda os erros nas iterações pedidas pelo template do relatório
        for(int c = 0; c < 5; c++){
            if(k == checkpoints[c]){
                checkpoint_errors[c] = global_error;
            }
        }

        // verifica convergência
        if(global_error < tol){

            if(rank == 0){
                printf("\n=== MPI Convergence Table ===\n");
                printf("Iteration\tError norm\n");

                for(int c = 0; c < 5; c++){
                    if(checkpoint_errors[c] >= 0.0){
                        printf("%d\t\t%e\n", checkpoints[c], checkpoint_errors[c]);
                    }
                    else{
                        printf("%d\t\tNot reached\n", checkpoints[c]);
                    }
                }

                printf("\n");
            }
            return k + 1;
        }

        // atualiza x_old
        for(int i = 0; i < n; i++){
            x_old[i] = x[i];
        }
    }

        /*
            Se chegou aqui, significa que o método não convergiu antes de max_iter.

            Mesmo assim, imprimimos a tabela de convergência com os valores que foram
            registrados nas iterações 0, 50, 100, 200 e 300.
        */
        if(rank == 0){
        printf("\n=== MPI Convergence Table ===\n");
        printf("Iteration\tError norm\n");

        for(int c = 0; c < 5; c++){
            if(checkpoint_errors[c] >= 0.0){
                printf("%d\t\t%e\n", checkpoints[c], checkpoint_errors[c]);
            }
            else{
                printf("%d\t\tNot reached\n", checkpoints[c]);
            }
        }
        printf("\n");
    }

    return max_iter;
}