#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
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
    double delta = eps / 10.0;

    double L0 = b - a;
    double R = L0 / eps;

    vector<long long> F = { 0, 1, 1 };
    while (F.back() < R) {
        F.push_back(F.back() + F[F.size() - 2]);
    }

    int N = F.size() - 1;

    cout << "Расчет N:\n";
    cout << "L0 = " << L0 << ", eps = " << eps << ", R = " << R << "\n";
    cout << "Выбрано N = " << N << " (F[" << N << "] = " << F[N] << " >= " << R << ")\n";
    cout << "Всего шагов (N-2): " << N - 2 << "\n";
    cout << "Запланировано вычислений функции: " << N - 1 << "\n\n";

    double x1 = 0.0, x2 = 0.0;
    double f1 = 0.0, f2 = 0.0;
    int f_evals = 0;

    int m = N;
    x1 = a + ((double)F[m - 2] / F[m]) * (b - a);
    x2 = a + ((double)F[m - 1] / F[m]) * (b - a);
    f1 = f(x1);
    f2 = f(x2);
    f_evals += 2;

    cout << fixed << setprecision(5);
    cout << "k   | m  | F[m-2]/F[m] | F[m-1]/F[m] | a_k      | b_k      | x1       | x2       | f(x1)    | f(x2)    | L_k      | Решение\n";
    cout << string(130, '-') << "\n";

    for (int k = 0; k <= N - 3; k++) {
        m = N - k;
        double L_k = b - a;
        string decision;

        double frac1 = (double)F[m - 2] / F[m];
        double frac2 = (double)F[m - 1] / F[m];

        if (f1 <= f2) {
            decision = "f1 <= f2 -> b=x2";
            b = x2;

            x2 = x1;
            f2 = f1;

            if (m - 1 == 3) {
                x1 = a + (b - a) / 2.0;
                x2 = x1 + delta;
                f1 = f(x1);
                f2 = f(x2);
                f_evals++;
            }
            else {
                x1 = a + ((double)F[m - 3] / F[m - 1]) * (b - a);
                f1 = f(x1);
                f_evals++;
            }
        }
        else {
            decision = "f1 > f2 -> a=x1";
            a = x1;

            x1 = x2;
            f1 = f2;

            if (m - 1 == 3) {
                x1 = a + (b - a) / 2.0;
                x2 = x1 + delta;
                f1 = f(x1);
                f2 = f(x2);
                f_evals++;
            }
            else {
                x2 = a + ((double)F[m - 2] / F[m - 1]) * (b - a);
                f2 = f(x2);
                f_evals++;
            }
        }

        cout << setw(3) << k << " | "
            << setw(2) << m << " | "
            << setw(11) << frac1 << " | "
            << setw(11) << frac2 << " | "
            << setw(8) << a << " | "
            << setw(8) << b << " | "
            << setw(8) << x1 << " | "
            << setw(8) << x2 << " | "
            << setw(8) << f1 << " | "
            << setw(8) << f2 << " | "
            << setw(8) << L_k << " | "
            << decision << "\n";
    }

    double x_star = (a + b) / 2.0;
    double f_star = f(x_star);

    cout << string(130, '=') << "\n";
    cout << "Результат метода Фибоначчи:\n";
    cout << "x* = " << x_star << "\n";
    cout << "f(x*) = " << f_star << "\n";
    cout << "Фактическая конечная длина L_кин = " << (b - a) << "\n";
    cout << "Теоретическая гарантия L0 / F[N] = " << L0 / F[N] << "\n";
    cout << "Количество вычислений функции N_f: " << f_evals << "\n";

    return 0;
}