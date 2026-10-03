#include <iostream>
#include <cmath>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

// Целевая функция
double f(double x) {
    return -(4.0 * x + 4.0) * exp(-x) + cos(x) + x;
}

int main() {
    setlocale(LC_ALL, "Russian");

    double x1_start = -1.25;
    double dx = 0.1;
    double sigma_x = 0.001;
    double sigma_f = 0.01;

    cout << "=== Метод Пауэлла (квадратичная аппроксимация) ===\n";
    cout << "Входные параметры:\n";
    cout << "x1_0 = " << x1_start << ", dx = " << dx
        << ", sigma_x = " << sigma_x << ", sigma_f = " << sigma_f << "\n\n";

    double x1 = x1_start;
    double x2 = x1 + dx;
    double f1 = f(x1);
    double f2 = f(x2);

    double x3;
    if (f1 > f2) {
        x3 = x1 + 2.0 * dx;
    }
    else {
        x3 = x1 - dx;
    }
    double f3 = f(x3);

    int f_evals = 3;

    cout << fixed << setprecision(6);
    cout << setw(3) << "k" << " | "
        << setw(9) << "x1" << " | "
        << setw(9) << "x2" << " | "
        << setw(9) << "x3" << " | "
        << setw(9) << "f(x1)" << " | "
        << setw(9) << "f(x2)" << " | "
        << setw(9) << "f(x3)" << " | "
        << setw(10) << "a1" << " | "
        << setw(10) << "a2" << " | "
        << setw(9) << "x_tilde" << " | "
        << setw(9) << "f(x_t)" << " | "
        << setw(9) << "|df|" << " | "
        << setw(9) << "|dx|" << " | "
        << "Действие\n";
    cout << string(155, '-') << "\n";

    int k = 1;
    double x_tilde = 0.0;
    double f_tilde = 0.0;
    bool converged = false;

    while (k <= 50) {
        vector<double> xs = { x1, x2, x3 };
        vector<double> fs = { f1, f2, f3 };

        int i_min = 0, i_max = 0;
        for (int i = 1; i < 3; ++i) {
            if (fs[i] < fs[i_min]) i_min = i;
            if (fs[i] > fs[i_max]) i_max = i;
        }

        double x_min = xs[i_min];
        double f_min = fs[i_min];
        double x_max = xs[i_max];

        if (abs(x2 - x1) < 1e-12 || abs(x3 - x1) < 1e-12 || abs(x3 - x2) < 1e-12) {
            cout << "Предупреждение: Точки слишком близко друг к другу!\n";
            break;
        }

        double a1 = (f2 - f1) / (x2 - x1);
        double a2 = ((f3 - f1) / (x3 - x1) - a1) / (x3 - x2);

        string note = "";
        if (a2 > 0) {
            x_tilde = -a1 / (2.0 * a2) + (x1 + x2) / 2.0;
        }
        else {
            x_tilde = 2.0 * x_min - x_max;
            note = " [a2<=0: отскок]";
        }

        f_tilde = f(x_tilde);
        f_evals++;

        double diff_f = abs(f_min - f_tilde);
        double diff_x = abs(x_min - x_tilde);

        if (diff_f <= sigma_f && diff_x <= sigma_x) {
            converged = true;
            note = "СТОП (достигнута точность)" + note;
        }
        else {
            note = "x_max -> x_tilde" + note;
        }

        cout << setw(3) << k << " | "
            << setw(9) << x1 << " | "
            << setw(9) << x2 << " | "
            << setw(9) << x3 << " | "
            << setw(9) << f1 << " | "
            << setw(9) << f2 << " | "
            << setw(9) << f3 << " | "
            << setw(10) << a1 << " | "
            << setw(10) << a2 << " | "
            << setw(9) << x_tilde << " | "
            << setw(9) << f_tilde << " | "
            << setw(9) << diff_f << " | "
            << setw(9) << diff_x << " | "
            << note << "\n";

        if (converged) {
            break;
        }

        if (i_max == 0) { x1 = x_tilde; f1 = f_tilde; }
        else if (i_max == 1) { x2 = x_tilde; f2 = f_tilde; }
        else { x3 = x_tilde; f3 = f_tilde; }

        k++;
    }

    double x_star = x_tilde;
    double f_star = f_tilde;

    cout << string(155, '=') << "\n";
    cout << "Результат метода Пауэлла:\n";
    cout << "x* = " << x_star << "\n";
    cout << "f(x*) = " << f_star << "\n";
    cout << "Количество итераций n: " << k << "\n";
    cout << "Всего вычислений функции N_f: " << f_evals << " (N_f = n + 3)\n";

    return 0;
}