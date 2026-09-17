#include <iostream>
#include <cmath>
#include <iomanip>
#include <algorithm>

using namespace std;

// Целевая функция f(x)
double f(double x) {
    return -(4.0 * x + 4.0) * exp(-x) + cos(x) + x;
}


// Метод Свенна
void svennMethod(double x0, double delta0, double& a_out, double& b_out) {
    cout << "                   Результаты поиска методом Свенна\n";
    cout << left
        << setw(16) << "№ итер., k"
        << setw(12) << "Delta_0"
        << setw(12) << "x_k"
        << setw(16) << "f(x_k)"
        << setw(26) << "f(x_k) < f(x_{k-1})?"
        << setw(20) << "[a_0, b_0]" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------------------------\n";

    int k = 0;
    int p = 0;
    double delta = delta0;
    double x_k = x0;
    double f_k = f(x_k);
    double x_prev = x_k; 

    cout << left
        << setw(16) << k
        << setw(12) << fixed << setprecision(2) << delta
        << setw(12) << setprecision(2) << x_k
        << setw(16) << setprecision(4) << f_k
        << setw(26) << "-"
        << setw(20) << "-" << endl;

    bool finished = false;
    while (!finished) {
        double x_next = x_k + delta;
        double f_next = f(x_next);

        if (f_next < f_k) {
            x_prev = x_k;
            x_k = x_next;
            f_k = f_next;
            k++;
            delta *= 2.0;

            cout << left
                << setw(16) << k
                << setw(12) << fixed << setprecision(2) << delta
                << setw(12) << setprecision(2) << x_k
                << setw(16) << setprecision(4) << f_k
                << setw(26) << "Да"
                << setw(20) << "-" << endl;
        }
        else {
            if (p == 0 && k == 0) {
                delta = -delta;
                p = 1;
            }
            else if (p == 1 && k == 0) {
                a_out = x0 - fabs(delta0);
                b_out = x0 + fabs(delta0);
                k++;

                cout << left
                    << setw(16) << k
                    << setw(12) << fixed << setprecision(2) << delta
                    << setw(12) << setprecision(2) << x_next
                    << setw(16) << setprecision(4) << f_next
                    << setw(26) << "Нет"
                    << "[" << setprecision(2) << a_out << ", " << b_out << "]" << endl;

                finished = true;
            }
            else {
                a_out = min(x_prev, x_next);
                b_out = max(x_prev, x_next);
                k++;

                cout << left
                    << setw(16) << k
                    << setw(12) << fixed << setprecision(2) << delta
                    << setw(12) << setprecision(2) << x_next
                    << setw(16) << setprecision(4) << f_next
                    << setw(26) << "Нет"
                    << "[" << setprecision(2) << a_out << ", " << b_out << "]" << endl;

                finished = true;
            }
        }
    }
    cout << "----------------------------------------------------------------------------------------------------------------------------------\n";
    cout << "Результат (интервал Свенна): [" << a_out << ", " << b_out << "]\n\n\n";
}


// Метод Дихотомии

void dichotomyMethod(double a0, double b0, double sigma, double eps) {
    cout << "                  Результаты поиска методом Дихотомии\n";
    cout << left
        << setw(16) << "№ итер., k"
        << setw(14) << "x1"
        << setw(14) << "x2"
        << setw(16) << "f(x1)"
        << setw(16) << "f(x2)"
        << setw(24) << "[a_k, b_k]"
        << setw(12) << "L_k" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------------------------\n";

    double ak = a0;
    double bk = b0;
    int k = 0;

    while (true) {
        double Lk = bk - ak;
        double mid = (ak + bk) / 2.0;
        double x1 = mid - eps;
        double x2 = mid + eps;
        double f1 = f(x1);
        double f2 = f(x2);

        cout << left
            << setw(16) << k
            << setw(14) << fixed << setprecision(4) << x1
            << setw(14) << setprecision(4) << x2
            << setw(16) << setprecision(4) << f1
            << setw(16) << setprecision(4) << f2
            << "[" << setw(8) << setprecision(4) << ak << ", "
            << setw(8) << setprecision(4) << bk << "]    "
            << setw(12) << setprecision(4) << Lk << endl;

        if (Lk <= sigma) {
            double x_star = (ak + bk) / 2.0;
            cout << "----------------------------------------------------------------------------------------------------------------------------------\n";
            cout << "Результат: x* = " << setprecision(4) << x_star
                << ", f(x*) = " << f(x_star) << "\n\n\n";
            break;
        }

        if (f1 < f2) {
            bk = x2;
        }
        else {
            ak = x1;
        }
        k++;
    }
}


// Метод Половинного Деления

void intervalHalvingMethod(double a0, double b0, double sigma) {
    cout << "            Результаты поиска методом Половинного Деления\n";
    cout << left
        << setw(14) << "№ итер., k"
        << setw(12) << "x1"
        << setw(12) << "x_m"
        << setw(12) << "x2"
        << setw(14) << "f(x1)"
        << setw(14) << "f(x_m)"
        << setw(14) << "f(x2)"
        << setw(24) << "[a_k, b_k]"
        << setw(12) << "L_k" << endl;
    cout << "----------------------------------------------------------------------------------------------------------------------------------\n";
    double ak = a0;
    double bk = b0;
    int k = 0;

    while (true) {
        double Lk = bk - ak;
        double xm = (ak + bk) / 2.0;
        double x1 = ak + Lk / 4.0;
        double x2 = bk - Lk / 4.0;

        double f1 = f(x1);
        double fm = f(xm);
        double f2 = f(x2);

        cout << left
            << setw(14) << k
            << setw(12) << fixed << setprecision(4) << x1
            << setw(12) << setprecision(4) << xm
            << setw(12) << setprecision(4) << x2
            << setw(14) << setprecision(4) << f1
            << setw(14) << setprecision(4) << fm
            << setw(14) << setprecision(4) << f2
            << "[" << setw(8) << setprecision(4) << ak << ", "
            << setw(8) << setprecision(4) << bk << "]  "
            << setw(12) << setprecision(4) << Lk << endl;

        if (Lk <= sigma) {
            double x_star = xm;
            cout << "----------------------------------------------------------------------------------------------------------------------------------\n";
            cout << "Результат: x* = " << setprecision(4) << x_star
                << ", f(x*) = " << f(x_star) << "\n\n";
            break;
        }

        if (f1 < fm) {
            bk = xm;
        }
        else if (f2 < fm) {
            ak = xm;
        }
        else {
            ak = x1;
            bk = x2;
        }
        k++;
    }
}

int main() {
    setlocale(LC_ALL, "Russian");
    double x0 = -0.25;
    double delta0 = 1.0;
    double sigma = 0.05;
    double eps = 0.01;

    double a0 = 0.0, b0 = 0.0;

    svennMethod(x0, delta0, a0, b0);
    dichotomyMethod(a0, b0, sigma, eps);
    intervalHalvingMethod(a0, b0, sigma);

    return 0;
}