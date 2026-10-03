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

// Первая производная f'(x)
double df(double x) {
    return 4.0 * x * exp(-x) - sin(x) + 1.0;
}

// Вторая производная f''(x)
double d2f(double x) {
    return (4.0 - 4.0 * x) * exp(-x) - cos(x);
}

int main() {
    setlocale(LC_ALL, "Russian");

    double x0 = -1.25;
    double eps = 0.001;
    double alpha = 0.5;

    cout << "=== ЛАБОРАТОРНАЯ РАБОТА № 6 ===\n";
    cout << "Входные параметры:\n";
    cout << "x0 = " << x0 << ", eps = " << eps << ", alpha = " << alpha << "\n\n";

    // ЧАСТЬ А: Классический метод Ньютона

    cout << "=== ЧАСТЬ А: Классический метод Ньютона ===\n";
    cout << fixed << setprecision(6);
    cout << setw(3)  << "k" << " | "
         << setw(9)  << "x_k" << " | "
         << setw(9)  << "f(x_k)" << " | "
         << setw(10) << "f'(x_k)" << " | "
         << setw(9)  << "|f'|" << " | "
         << setw(9)  << "f''(x_k)" << " | "
         << "Действие\n";
    cout << string(85, '-') << "\n";

    double x_a = x0;
    int k_a = 0;
    int f_evals_a = 0, df_evals_a = 0, d2f_evals_a = 0;

    while (k_a <= 50) {
        double fx = f(x_a);         f_evals_a++;
        double d1 = df(x_a);        df_evals_a++;
        double d2 = d2f(x_a);       d2f_evals_a++;

        double abs_d1 = abs(d1);
        string note = "";

        if (abs_d1 <= eps) {
            note = "СТОП (|f'| <= eps)";
            cout << setw(3)  << k_a << " | "
                 << setw(9)  << x_a << " | "
                 << setw(9)  << fx << " | "
                 << setw(10) << d1 << " | "
                 << setw(9)  << abs_d1 << " | "
                 << setw(9)  << d2 << " | "
                 << note << "\n";
            break;
        }

        if (d2 <= 0) {
            cout << "Предупреждение: f''(x) <= 0! Шаг может вести к максимуму.\n";
        }

        note = "x_(k+1) = x_k - f'/f''";
        cout << setw(3)  << k_a << " | "
             << setw(9)  << x_a << " | "
             << setw(9)  << fx << " | "
             << setw(10) << d1 << " | "
             << setw(9)  << abs_d1 << " | "
             << setw(9)  << d2 << " | "
             << note << "\n";

        x_a = x_a - d1 / d2;
        k_a++;
    }

    cout << "\nРезультат (Классический Ньютон):\n";
    cout << "x* = " << x_a << ", f(x*) = " << f(x_a) << "\n";
    cout << "Количество итераций n: " << k_a << "\n";
    cout << "Вычислений: f(x) = " << f_evals_a + 1
         << ", f'(x) = " << df_evals_a
         << ", f''(x) = " << d2f_evals_a << "\n\n";

    // ЧАСТЬ Б: Метод Ньютона-Рафсона с регулированием шага

    cout << string(125, '=') << "\n";
    cout << "=== ЧАСТЬ Б: Метод Ньютона-Рафсона с регулированием шага ===\n";
    cout << setw(3)  << "k" << " | "
         << setw(9)  << "x_k" << " | "
         << setw(9)  << "f(x_k)" << " | "
         << setw(10) << "f'(x_k)" << " | "
         << setw(9)  << "f''(x_k)" << " | "
         << setw(6)  << "t" << " | "
         << setw(9)  << "x_try" << " | "
         << setw(9)  << "f(x_try)" << " | "
         << setw(10) << "f'(x_try)" << " | "
         << "Условие (3.10)\n";
    cout << string(125, '-') << "\n";

    double x_b = x0;
    int k_b = 0;
    int f_evals_b = 0, df_evals_b = 0, d2f_evals_b = 0;

    while (k_b <= 50) {
        double fx = f(x_b);         f_evals_b++;
        double d1 = df(x_b);        df_evals_b++;
        double d2 = d2f(x_b);       d2f_evals_b++;

        if (abs(d1) <= eps) {
            cout << setw(3)  << k_b << " | "
                 << setw(9)  << x_b << " | "
                 << setw(9)  << fx << " | "
                 << setw(10) << d1 << " | "
                 << setw(9)  << d2 << " | "
                 << setw(6)  << "-" << " | "
                 << setw(9)  << "-" << " | "
                 << setw(9)  << "-" << " | "
                 << setw(10) << "-" << " | "
                 << "СТОП (|f'| <= eps)\n";
            break;
        }

        if (d2 <= 0) {
            cout << "Ошибка: f''(x) <= 0! Выполнение остановлено.\n";
            break;
        }

        double t = 1.0;
        bool step_accepted = false;
        double d1_accepted = 0.0;


        while (t >= 1e-8) {
            double x_try = x_b - t * d1 / d2;
            double f_try = f(x_try);     f_evals_b++;
            double d1_try = df(x_try);   df_evals_b++;

            double lhs = f_try - fx;
            double rhs = -alpha * t * (d1 * d1) / d2;
            bool armijo = (lhs <= rhs);

            cout << setw(3)  << k_b << " | "
                 << setw(9)  << x_b << " | "
                 << setw(9)  << fx << " | "
                 << setw(10) << d1 << " | "
                 << setw(9)  << d2 << " | "
                 << setw(6)  << t << " | "
                 << setw(9)  << x_try << " | "
                 << setw(9)  << f_try << " | "
                 << setw(10) << d1_try << " | "
                 << (armijo ? "Выполнено (шаг принят)" : "Не выполнено (t = t/2)") << "\n";

            if (abs(d1_try) <= eps) {
                x_b = x_try;
                d1_accepted = d1_try;
                step_accepted = true;
                break;
            }

            if (armijo) {
                x_b = x_try;
                d1_accepted = d1_try;
                step_accepted = true;
                break;
            }

            t /= 2.0;
        }

        if (!step_accepted) {
            cout << "Предупреждение: Достигнут минимальный шаг t!\n";
            break;
        }

        if (abs(d1_accepted) <= eps) break;

        k_b++;
    }

    cout << string(125, '=') << "\n";
    cout << "Результат (Ньютон-Рафсон с регулированием шага):\n";
    cout << "x* = " << x_b << "\n";
    cout << "f(x*) = " << f(x_b) << "\n";
    cout << "Количество внешних итераций n: " << k_b << "\n";
    cout << "Вычислений функций: f(x) = " << f_evals_b + 1
         << ", f'(x) = " << df_evals_b
         << ", f''(x) = " << d2f_evals_b << "\n";

    return 0;
}
