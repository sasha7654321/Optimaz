#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <clocale>

using namespace std;

//Варианта 4
double computeF(double x1, double x2) {
    return 2.0 * x1 * x1 + 3.0 * x2 * x2 - 2.0 * x1 * x2 - 8.0 * x1 - x2;
}

void computeGrad(double x1, double x2, double& g1, double& g2) {
    g1 = 4.0 * x1 - 2.0 * x2 - 8.0;
    g2 = 6.0 * x2 - 2.0 * x1 - 1.0;
}

double sgn(double v) {
    return (v > 0) ? 1.0 : ((v < 0) ? -1.0 : 0.0);
}

void getDirection(double g1, double g2, const string& metric, double& d1, double& d2) {
    d1 = 0.0;
    d2 = 0.0;

    if (metric == "sphere") {
        double norm_g = sqrt(g1 * g1 + g2 * g2);
        if (norm_g > 0) {
            d1 = -g1 / norm_g;
            d2 = -g2 / norm_g;
        }
    }
    else if (metric == "cube") {
        d1 = -sgn(g1);
        d2 = -sgn(g2);
    }
    else if (metric == "octa") {
        if (abs(g1) >= abs(g2)) {
            d1 = -sgn(g1);
        }
        else {
            d2 = -sgn(g2);
        }
    }
}

void runOptimization(const string& metricName, double h0, double eps) {
    const double X1_EXACT = 2.5;
    const double X2_EXACT = 1.0;
    const double H_MIN = 1e-10;
    const int MAX_ITER = 1000;

    double x1 = -4.0; //x0
    double x2 = -4.0;
    double h = h0;

    cout << "=====================================================================================================================\n";
    cout << fixed << setprecision(2);
    cout << "Метрика: " << metricName << " | Начальный шаг h0 = " << h0 << " | eps = " << eps << "\n";
    cout << "=====================================================================================================================\n";
    cout << fixed << setprecision(6);
    cout << setw(4) << "k"
        << setw(12) << "x1"
        << setw(12) << "x2"
        << setw(13) << "F(x_k)"
        << setw(11) << "g1"
        << setw(11) << "g2"
        << setw(11) << "||g||_2"
        << setw(10) << "d1"
        << setw(10) << "d2"
        << setw(10) << "h"
        << setw(7) << " дроб."
        << setw(13) << "F(x_k+1)" << "\n";
    cout << string(124, '-') << "\n";

    // Шаг 1
    double Fx = computeF(x1, x2);
    int nF = 1;
    int nG = 0;

    bool converged = false;
    bool stuck = false;
    int k = 0;

    for (; k < MAX_ITER; ++k) {
        // Шаг 2
        double g1, g2;
        computeGrad(x1, x2, g1, g2);
        nG++;
        double norm_g = sqrt(g1 * g1 + g2 * g2);

        if (norm_g <= eps) {
            cout << setw(4) << k
                << setw(12) << x1
                << setw(12) << x2
                << setw(13) << Fx
                << setw(11) << g1
                << setw(11) << g2
                << setw(11) << norm_g
                << setw(10) << "-"
                << setw(10) << "-"
                << setw(10) << "-"
                << setw(7) << "-"
                << setw(13) << "-" << "\n";
            converged = true;
            break;
        }

        // Шаг 3
        double d1, d2;
        getDirection(g1, g2, metricName, d1, d2);

        // Шаги 4-5
        int halvings = 0;
        bool accepted = false;
        double x1_new = x1, x2_new = x2, F_new = Fx;

        while (true) {
            x1_new = x1 + h * d1;
            x2_new = x2 + h * d2;
            F_new = computeF(x1_new, x2_new);
            nF++;

            if (F_new < Fx) {
                accepted = true;
                break;
            }

            h /= 2.0;
            halvings++;
            if (h < H_MIN) {
                break;
            }
        }

        if (!accepted) {
            cout << setw(4) << k
                << setw(12) << x1
                << setw(12) << x2
                << setw(13) << Fx
                << setw(11) << g1
                << setw(11) << g2
                << setw(11) << norm_g
                << setw(10) << d1
                << setw(10) << d2
                << setw(10) << h
                << setw(7) << halvings
                << setw(13) << "-" << "\n";
            stuck = true;
            break;
        }

        cout << setw(4) << k
            << setw(12) << x1
            << setw(12) << x2
            << setw(13) << Fx
            << setw(11) << g1
            << setw(11) << g2
            << setw(11) << norm_g
            << setw(10) << d1
            << setw(10) << d2
            << setw(10) << h
            << setw(7) << halvings
            << setw(13) << F_new << "\n";

        // Шаг 6
        x1 = x1_new;
        x2 = x2_new;
        Fx = F_new;
    }

    cout << string(124, '-') << "\n";
    if (converged) {
        cout << "-> Остановка: ||g||_2 <= " << eps << " на итерации " << k << "\n";
    }
    else if (stuck) {
        cout << "-> Предупреждение: шаг раздробился до h_min, метод остановлен.\n";
    }
    else {
        cout << "-> Предупреждение: превышен лимит итераций (" << MAX_ITER << ").\n";
    }

    double dist = sqrt((x1 - X1_EXACT) * (x1 - X1_EXACT) + (x2 - X2_EXACT) * (x2 - X2_EXACT));
    cout << "x* = (" << x1 << "; " << x2 << "), F(x*) = " << Fx << "\n";
    cout << scientific << setprecision(2)
        << "||x* - x_точн||_2 = " << dist << fixed << setprecision(6)
        << "  (x_точн = (2.5; 1), F = -10.5)\n";
    cout << "Итог: n = " << k << ", N_F = " << nF << ", N_grad = " << nG
        << ", всего = " << nF + nG << "\n\n";
}

int main() {
    setlocale(LC_ALL, "Russian");

    // Вариант 4
    double eps = 0.01;

    runOptimization("sphere", 1.0, eps);
    runOptimization("cube", 1.0, eps);
    runOptimization("octa", 1.0, eps);

    runOptimization("sphere", 0.7, eps);
    runOptimization("cube", 0.7, eps);
    runOptimization("octa", 0.7, eps);

    return 0;
}