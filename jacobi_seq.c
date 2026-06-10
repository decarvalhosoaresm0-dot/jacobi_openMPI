#include <math.h>

int jacobi_sequential(
    int n,
    double a[n][n],
    double b[n],
    double x[n],
    int max_iter,
    double tol,
    double *final_error
) {

    // x ==> x^(k+1)              
    // x_old ==> x^(k)            

    double x_old[n];

    // Inicialização do x^(0)... optando sempre por inicializar todas as posições = 0    
    for(int i = 0; i < n; i++){
        x_old[i] = x[i];
    }


    // Laço externo das iterações => NÃO pode ser PARALELIZADO, porque a iteração k+1 depende de k
    for(int k = 0; k < max_iter; k++){
        
        double erro = 0.0;

        // Laço interno que calcula cada componente do novo vetor x
        for(int i = 0; i < n; i++){
            
            double soma = 0.0;

            // Cálculo da soma => Σa_ij x_j (usando o x_old)
            for (int j = 0; j < n; j++) {
                if(i != j){
                    soma += a[i][j] * x_old[j];
                } 
            }

             // Aplicação do Método de Jacobi
            x[i] = (b[i] - soma) / a[i][i];

            // Diferença do módulo entre o valor novo e antigo na pos. i
            double diferenca = x[i] - x_old[i];
        
            erro += diferenca * diferenca;

        }

        erro = sqrt(erro);

        *final_error = erro;

        // Retorna o número de iterações realizadas
        if(erro < tol){
            return k + 1;
        }

        // Caso não tenha convergido para um erro menor que tol, então x^(k+1) = x^(k)
        for(int i = 0; i < n; i++){
            x_old[i] = x[i];
        }

    }
    
    return max_iter;

}
