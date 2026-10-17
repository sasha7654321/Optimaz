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

double df(double x) {
    return 4.0 * x * exp(-x) - sin(x) + 1.0;
}

int main() {
    setlocale(LC_ALL, "Russian");

    double x0 = -1.25;
    double x1 = 0.75;
    double eps = 0.001;

    cout << "ЛАБОРАТОРНАЯ РАБОТА № 7 \n";
    cout << "Метод секущих (вторая модификация метода Ньютона)\n";
    cout << "Входные параметры:\n";
    cout << "x0 = " << x0 << ", x1 = " << x1 << ", eps = " << eps << "\n\n";

    double df0 = df(x0);
    double df1 = df(x1);

    if (df0 * df1 >= 0) {
        cout << "Ошибка: Производная должна иметь разные знаки на концах интервала! "
            << "f'(x0) = " << df0 << ", f'(x1) = " << df1 << "\n";
        return 1;
    }

    // ЧАСТЬ 1

    cout << "ЧАСТЬ 1: Метод секущих\n";
    cout << fixed << setprecision(6);
    cout << setw(3) << "k" << " | "
        << setw(9) << "x_(k-1)" << " | "
        << setw(10) << "f'(x_k-1)" << " | "
        << setw(9) << "x_k" << " | "
        << setw(10) << "f'(x_k)" << " | "
        << setw(9) << "x_(k+1)" << " | "
        << setw(10) << "f'(x_k+1)" << " | "
        << setw(9) << "f(x_k+1)" << " | "
        << "Действие\n";
    cout << string(110, '-') << "\n";

    double xm = x0;
    double xk = x1;
    double pm = df0;
    double pk = df1;

    int k1 = 1;
    int df_evals_1 = 2;
    double x_star_1 = 0.0;
    bool converged_1 = false;

    while (k1 <= 50) {
        if (abs(pk - pm) < 1e-12) {
            cout << "Предупреждение: f'(x_k) и f'(x_k-1) равны! Деление на ноль.\n";
            break;
        }

        double xn = (pk * xm - pm * xk) / (pk - pm);
        double pn = df(xn);
        df_evals_1++;
        double fn = f(xn);

        double abs_pn = abs(pn);
        string note = "";

        if (abs_pn <= eps) {
            note = "СТОП (|f'| <= eps)";
            cout << setw(3) << k1 << " | "
                << setw(9) << xm << " | "
                << setw(10) << pm << " | "
                << setw(9) << xk << " | "
                << setw(10) << pk << " | "
                << setw(9) << xn << " | "
                << setw(10) << pn << " | "
                << setw(9) << fn << " | "
                << note << "\n";
            x_star_1 = xn;
            converged_1 = true;
            break;
        }

        if (pn > 0) {
            note = "x_k <- x_(k+1) (замена правой)";
            cout << setw(3) << k1 << " | "
                << setw(9) << xm << " | "
                << setw(10) << pm << " | "
                << setw(9) << xk << " | "
                << setw(10) << pk << " | "
                << setw(9) << xn << " | "
                << setw(10) << pn << " | "
                << setw(9) << fn << " | "
                << note << "\n";
            xk = xn;
            pk = pn;
        }
        else {
            note = "x_(k-1) <- x_(k+1) (замена левой)";
            cout << setw(3) << k1 << " | "
                << setw(9) << xm << " | "
                << setw(10) << pm << " | "
                << setw(9) << xk << " | "
                << setw(10) << pk << " | "
                << setw(9) << xn << " | "
                << setw(10) << pn << " | "
                << setw(9) << fn << " | "
                << note << "\n";
            xm = xn;
            pm = pn;
        }

        k1++;
    }

    cout << "\nРезультат: \n";
    if (converged_1) {
        cout << "x* = " << x_star_1 << ", f(x*) = " << f(x_star_1) << "\n";
        cout << "Количество итераций n: " << k1 << "\n";
        cout << "Всего вычислений первой производной N_f': " << df_evals_1 << " (N_f' = n + 2)\n\n";
    }
    else {
        cout << "Метод не сошёлся (деление на ноль или превышен лимит итераций): x* не найден.\n";
        cout << "Выполнено итераций: " << k1 - 1
            << ", вычислений первой производной N_f': " << df_evals_1 << "\n\n";
    }


    // ЧАСТЬ 2 Чистые» секущие

    cout << string(110, '=') << "\n";
    cout << "ЧАСТЬ 2: Эксперимент — Чистые» секущие\n";
    cout << setw(3) << "k" << " | "
        << setw(9) << "x_(k-1)" << " | "
        << setw(10) << "f'(x_k-1)" << " | "
        << setw(9) << "x_k" << " | "
        << setw(10) << "f'(x_k)" << " | "
        << setw(9) << "x_(k+1)" << " | "
        << setw(10) << "f'(x_k+1)" << " | "
        << setw(9) << "f(x_k+1)" << " | "
        << "Действие\n";
    cout << string(110, '-') << "\n";

    xm = x0;
    xk = x1;
    pm = df0;
    pk = df1;

    int k2 = 1;
    int df_evals_2 = 2;
    double x_star_2 = 0.0;
    bool converged_2 = false;

    while (k2 <= 50) {
        if (abs(pk - pm) < 1e-12) {
            cout << "Предупреждение: f'(x_k) и f'(x_k-1) равны!\n";
            break;
        }

        double xn = (pk * xm - pm * xk) / (pk - pm);
        double pn = df(xn);
        df_evals_2++;
        double fn = f(xn);

        string note = "";
        if (abs(pn) <= eps) {
            note = "СТОП (|f'| <= eps)";
            cout << setw(3) << k2 << " | "
                << setw(9) << xm << " | "
                << setw(10) << pm << " | "
                << setw(9) << xk << " | "
                << setw(10) << pk << " | "
                << setw(9) << xn << " | "
                << setw(10) << pn << " | "
                << setw(9) << fn << " | "
                << note << "\n";
            x_star_2 = xn;
            converged_2 = true;
            break;
        }

        note = "Сдвиг пары (xm<-xk, xk<-xn)";
        cout << setw(3) << k2 << " | "
            << setw(9) << xm << " | "
            << setw(10) << pm << " | "
            << setw(9) << xk << " | "
            << setw(10) << pk << " | "
            << setw(9) << xn << " | "
            << setw(10) << pn << " | "
            << setw(9) << fn << " | "
            << note << "\n";

        xm = xk;
        pm = pk;
        xk = xn;
        pk = pn;

        k2++;
    }

    cout << string(110, '=') << "\n";
    cout << "Результат («Чистые» секущие):\n";
    if (converged_2) {
        cout << "x* = " << x_star_2 << ", f(x*) = " << f(x_star_2) << "\n";
        cout << "Количество итераций n: " << k2 << "\n";
        cout << "Всего вычислений первой производной N_f': " << df_evals_2 << "\n";
    }
    else {
        cout << "Метод не сошёлся (деление на ноль или превышен лимит итераций): x* не найден.\n";
        cout << "Выполнено итераций: " << k2 - 1
            << ", вычислений первой производной N_f': " << df_evals_2 << "\n";
    }

    return 0;
}