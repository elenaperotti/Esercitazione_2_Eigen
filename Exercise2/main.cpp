#include <iostream>
#include "Eigen/Eigen"

using namespace std;
using namespace Eigen;

// dichiarazioni:

VectorXd SolvePALU(const MatrixXd& A,
                   const VectorXd& b);

VectorXd SolveQR(const MatrixXd& A,
                 const VectorXd& b);

void Errore_relativo(const MatrixXd& A,
                     const VectorXd& b,
                     const VectorXd& solution,
                     double& errRelPALU,
                     double& errRelQR);

bool CheckSV(const MatrixXd& A);



int main()
{
    // scrivo il vettore della soluzione esatta
    Vector2d soluzione(-1.0e+0, -1.0e+00);

    // Sistema 1
    Matrix2d A1{{5.547001962252291e-01, -3.770900990025203e-02}, { 8.320502943378437e-01,
                                                                  -9.992887623566787e-01}};
    Vector2d b1(-5.169911863249772e-01, 1.672384680188350e-01);

    // Sistema 2
    Matrix2d A2{{5.547001962252291e-01, -5.540607316466765e-01}, { 8.320502943378437e-01,
                                                                  -8.324762492991313e-01}};
    Vector2d b2(-6.394645785530173e-04, 4.259549612877223e-04);

    // Sistema 3
    Matrix2d A3{{5.547001962252291e-01, -5.547001955851905e-01}, { 8.320502943378437e-01,
                                                                  -8.320502947645361e-01}};
    Vector2d b3(-6.400391328043042e-10, 4.266924591433963e-10);

    //Controllo preliminare sulla possibilità di applicare i metodi numerici: la matrice ha determinante nullo?

    if (!CheckSV(A1))
    {
        cout << "Il primo sistema non è risolvibile, la matrice e' singolare ." << endl;
        return 1;
    }

    if (!CheckSV(A2))
    {
        cout << "Il secondo sistema non è risolvibile, la matrice e' singolare ." << endl;
        return 1;
    }

    if (!CheckSV(A3))
    {
        cout << "Il terzo sistema non è risolvibile, la matrice e' singolare ." << endl;
        return 1;
    }

    // Ora stampo i risultati degli errori relativi ottenuti
    double errore_relativo_PALU_1 = 0;
    double errore_relativo_QR_1 = 0;
    Errore_relativo(A1, b1, soluzione, errore_relativo_PALU_1, errore_relativo_QR_1);
    cout << "\n L'errore relativo utilizzando la fattorizzazione PALU della matrice A1 e' " << errore_relativo_PALU_1<<endl;
    cout << "\n L'errore relativo utilizzando la fattorizzazione QR della matrice A1 e' " << errore_relativo_QR_1<<endl;

    double errore_relativo_PALU_2 = 0;
    double errore_relativo_QR_2 = 0;
    Errore_relativo(A2, b2, soluzione, errore_relativo_PALU_2, errore_relativo_QR_2);
    cout << "\n\n L'errore relativo utilizzando la fattorizzazione PALU della matrice A2 e' " << errore_relativo_PALU_2<<endl;
    cout << "\n L'errore relativo utilizzando la fattorizzazione QR della matrice A2 e' " << errore_relativo_QR_2<<endl;

    double errore_relativo_PALU_3 = 0;
    double errore_relativo_QR_3 = 0;
    Errore_relativo(A3, b3, soluzione, errore_relativo_PALU_3, errore_relativo_QR_3);
    cout << "\n\n L'errore relativo utilizzando la fattorizzazione PALU della matrice A3 e' " << errore_relativo_PALU_3<<endl;
    cout << "\n L'errore relativo utilizzando la fattorizzazione QR della matrice A3 e' " << errore_relativo_QR_3 <<endl;



    return 0;
}

// definizione delle funzioni per risolvere i sistemi:

// 1 METODO: FATTORIZZAZIONE PALU
VectorXd SolvePALU(const MatrixXd& A,
                   const VectorXd& b)
{
    VectorXd solutionPALU = A.fullPivLu().solve(b);

    return solutionPALU;
}

// 2 METODO: FATTORIZZAZIONE QR
VectorXd SolveQR(const MatrixXd& A,
                 const VectorXd& b)
{
    VectorXd solutionQR = A.colPivHouseholderQr().solve(b);

    return solutionQR;
}

// funzione che stampa l'errore relativo
void Errore_relativo(const MatrixXd& A,
                     const VectorXd& b,
                     const VectorXd& soluzione,
                     double& errRelPALU,
                     double& errRelQR )
{
    errRelPALU = (SolvePALU(A,b)-soluzione).norm()/soluzione.norm();
    errRelQR = (SolveQR(A,b)-soluzione).norm()/soluzione.norm();
}

// funzione che controlla se il sistema è risolvibile
bool CheckSV(const MatrixXd& A)
{
    JacobiSVD<MatrixXd> svd(A);
    VectorXd singularValuesA = svd.singularValues();

    if( singularValuesA.minCoeff() < 1e-16) // precisione di macchina
    {
        return false;
    }

    return true;
}
