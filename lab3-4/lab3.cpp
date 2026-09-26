#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>

using namespace std;

// Целевая функция
double f(double x) {
    return -(4.0 * x + 4.0) * exp(-x) + cos(x) + x;
}

int main() {
    setlocale(LC_ALL, "Russian");
    double a = -1.25;
    double b = 0.75;
    double eps = 0.05;

    const double tau = (sqrt(5.0) - 1.0) / 2.0;

    double x1 = a + (1.0 - tau) * (b - a);
    double x2 = a + tau * (b - a);

    double f1 = f(x1);
    double f2 = f(x2);

    int f_evals = 2;
    int k = 1;

    cout << fixed << setprecision(5);
    cout << "k   | a_k      | b_k      | x1       | x2       | f(x1)    | f(x2)    | L_k      | Решение\n";
    cout << string(95, '-') << "\n";

    while ((b - a) > eps) {
        double L_k = b - a;
        string decision;

        if (f1 <= f2) {
            decision = "f1 <= f2 -> b=x2, успадковано x2";
            b = x2;

            x2 = x1;
            f2 = f1;

            x1 = a + (1.0 - tau) * (b - a);
            f1 = f(x1);
        }
        else {
            decision = "f1 > f2 -> a=x1, успадковано x1";
            a = x1;

            x1 = x2;
            f1 = f2;

            x2 = a + tau * (b - a);
            f2 = f(x2);
        }

        f_evals++;

        cout << setw(3) << k << " | "
            << setw(8) << a << " | "
            << setw(8) << b << " | "
            << setw(8) << x1 << " | "
            << setw(8) << x2 << " | "
            << setw(8) << f1 << " | "
            << setw(8) << f2 << " | "
            << setw(8) << L_k << " | "
            << decision << "\n";
        k++;
    }

    double x_star = (a + b) / 2.0;
    double f_star = f(x_star);

    cout << string(95, '=') << "\n";
    cout << "Результат:\n";
    cout << "x* = " << x_star << "\n";
    cout << "f(x*) = " << f_star << "\n";
    cout << "Количество итераций (k): " << k - 1 << "\n";
    cout << "Количество вычислений функции N_f: " << f_evals << "\n";

    return 0;
}