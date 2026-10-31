#include <iostream>
#include <cmath>
#include <iomanip>
#include <string>
#include <clocale>
#include <algorithm>

using namespace std;

typedef double (*Func)(double, double);
typedef void (*Grad)(double, double, double&, double&);

// Вариант 4 
double computeF(double x1, double x2) {
    return 2.0 * x1 * x1 + 3.0 * x2 * x2 - 2.0 * x1 * x2 - 8.0 * x1 - x2;
}

void computeGrad(double x1, double x2, double& g1, double& g2) {
    g1 = 4.0 * x1 - 2.0 * x2 - 8.0;
    g2 = 6.0 * x2 - 2.0 * x1 - 1.0;
}

void getHessian(double H[2][2]) {
    H[0][0] = 4.0;  H[0][1] = -2.0;
    H[1][0] = -2.0; H[1][1] = 6.0;
}

// Функция Розенброка
double rosenbrockF(double x1, double x2) {
    return (1.0 - x1) * (1.0 - x1) + 100.0 * (x2 - x1 * x1) * (x2 - x1 * x1);
}

void rosenbrockGrad(double x1, double x2, double& g1, double& g2) {
    g1 = -2.0 * (1.0 - x1) - 400.0 * x1 * (x2 - x1 * x1);
    g2 = 200.0 * (x2 - x1 * x1);
}

//Одномерный поиск: Свенн + золотое сечение
double phi(Func F, double x1, double x2, double d1, double d2, double t) {
    return F(x1 + t * d1, x2 + t * d2);
}

// Локализация минимума
void svenn(Func F, double x1, double x2, double d1, double d2, double delta0,
    double& a, double& b, int& nF) {
    double f0 = phi(F, x1, x2, d1, d2, 0.0);
    double fp = phi(F, x1, x2, d1, d2, delta0);
    nF += 2;

    double delta = delta0;
    double t_cur = delta0, f_cur = fp;

    if (fp >= f0) {
        double fm = phi(F, x1, x2, d1, d2, -delta0);
        nF++;
        if (fm >= f0) {
            a = -delta0;
            b = delta0;
            return;
        }
        delta = -delta0;
        t_cur = -delta0;
        f_cur = fm;
    }

    double t_prev = 0.0;
    while (true) {
        delta *= 2.0;
        double t_next = t_cur + delta;
        double f_next = phi(F, x1, x2, d1, d2, t_next);
        nF++;

        if (f_next >= f_cur) {
            a = min(t_prev, t_next);
            b = max(t_prev, t_next);
            return;
        }
        t_prev = t_cur;
        t_cur = t_next;
        f_cur = f_next;
    }
}

double goldenSection(Func F, double x1, double x2, double d1, double d2,
    double a, double b, double eps, int& nF) {
    const double golden = (1.0 + sqrt(5.0)) / 2.0;
    double x_l = b - (b - a) / golden;
    double x_r = a + (b - a) / golden;
    double f_l = phi(F, x1, x2, d1, d2, x_l);
    double f_r = phi(F, x1, x2, d1, d2, x_r);
    nF += 2;

    while ((b - a) > eps) {
        if (f_l < f_r) {
            b = x_r;
            x_r = x_l;
            f_r = f_l;
            x_l = b - (b - a) / golden;
            f_l = phi(F, x1, x2, d1, d2, x_l);
        }
        else {
            a = x_l;
            x_l = x_r;
            f_l = f_r;
            x_r = a + (b - a) / golden;
            f_r = phi(F, x1, x2, d1, d2, x_r);
        }
        nF++;
    }
    return (a + b) / 2.0;
}

double exactLineSearch(Func F, double x1, double x2, double d1, double d2,
    double delta0, int& nF) {
    double a, b;
    svenn(F, x1, x2, d1, d2, delta0, a, b, nF);
    return goldenSection(F, x1, x2, d1, d2, a, b, 1e-6, nF);
}

// Метод Флетчера-Ривса
struct Result {
    double x1, x2, F;
    int n, nF, nG;
};

Result fletcherReeves(Func F, Grad grad, double x1, double x2, double eps, double delta0,
    bool restart, int maxIter, bool verbose) {
    const int n = 2;
    int nF = 0, nG = 0;

    double g1, g2;
    grad(x1, x2, g1, g2);
    nG++;
    double norm_g = sqrt(g1 * g1 + g2 * g2);
    double d1 = -g1, d2 = -g2;
    double Fx = F(x1, x2);

    int k = 0;
    for (; k < maxIter; ++k) {
        if (norm_g <= eps) {
            if (verbose) {
                cout << setw(4) << k << setw(13) << x1 << setw(13) << x2 << setw(13) << Fx
                    << setw(12) << g1 << setw(12) << g2 << setw(12) << norm_g
                    << setw(12) << "-" << setw(12) << "-" << setw(12) << "-"
                    << setw(11) << "-" << setw(14) << "-" << "\n";
            }
            break;
        }

        double h = exactLineSearch(F, x1, x2, d1, d2, delta0, nF);

        double x1_new = x1 + h * d1;
        double x2_new = x2 + h * d2;
        double F_new = F(x1_new, x2_new);

        double g1_new, g2_new;
        grad(x1_new, x2_new, g1_new, g2_new);
        nG++;
        double norm_g_new = sqrt(g1_new * g1_new + g2_new * g2_new);

        double beta = (norm_g_new * norm_g_new) / (norm_g * norm_g);
        if (restart && (k + 1) % n == 0) {
            beta = 0.0;
        }

        if (verbose) {
            cout << setw(4) << k << setw(13) << x1 << setw(13) << x2 << setw(13) << Fx
                << setw(12) << g1 << setw(12) << g2 << setw(12) << norm_g
                << setw(12) << d1 << setw(12) << d2 << setw(12) << h
                << setw(11) << beta << setw(14) << F_new << "\n";
        }

        double d1_old = d1, d2_old = d2;
        d1 = -g1_new + beta * d1;
        d2 = -g2_new + beta * d2;

        if (verbose && k == 0) {
            double H[2][2];
            getHessian(H);
            double orth = g1_new * d1_old + g2_new * d2_old;
            double conj = d1 * (H[0][0] * d1_old + H[0][1] * d2_old)
                + d2 * (H[1][0] * d1_old + H[1][1] * d2_old);
            double descent = g1_new * d1 + g2_new * d2;
            cout << scientific << setprecision(3)
                << "   проверка 1: <g(1), d(0)>       = " << orth << "  (должно быть ~0)\n"
                << "   проверка 2: d(1)^T * H * d(0)  = " << conj << "  (должно быть ~0)\n"
                << "   проверка 3: <g(1), d(1)>       = " << descent << "  (должно быть < 0)\n"
                << fixed << setprecision(6);
        }

        x1 = x1_new;
        x2 = x2_new;
        Fx = F_new;
        g1 = g1_new;
        g2 = g2_new;
        norm_g = norm_g_new;
    }

    Result r = { x1, x2, Fx, k, nF, nG };
    return r;
}

void runVariant4() {
    const double X1_EXACT = 2.5;
    const double X2_EXACT = 1.0;
    const double eps = 0.01;

    cout << "=====================================================================================================================\n";
    cout << "Метод Флетчера-Ривса с перезапуском (n = 2) | Вариант 4 | Старт: (-4; -4) | eps = " << eps << "\n";
    cout << "=====================================================================================================================\n";
    cout << fixed << setprecision(6);
    cout << setw(4) << "k" << setw(13) << "x1" << setw(13) << "x2" << setw(13) << "F(x_k)"
        << setw(12) << "g1" << setw(12) << "g2" << setw(12) << "||g||_2"
        << setw(12) << "d1" << setw(12) << "d2" << setw(12) << "h_k"
        << setw(11) << "beta_k+1" << setw(14) << "F(x_k+1)" << "\n";
    cout << string(150, '-') << "\n";

    Result r = fletcherReeves(computeF, computeGrad, -4.0, -4.0, eps, 0.01, true, 100, true);

    cout << string(150, '-') << "\n";
    cout << "x* = (" << r.x1 << "; " << r.x2 << "), F(x*) = " << r.F << "\n";
    double dist = sqrt((r.x1 - X1_EXACT) * (r.x1 - X1_EXACT) + (r.x2 - X2_EXACT) * (r.x2 - X2_EXACT));
    cout << scientific << setprecision(2) << "||x* - x_точн||_2 = " << dist
        << fixed << setprecision(6) << "  (x_точн = (2.5; 1), F = -10.5)\n";
    cout << "Итог: n = " << r.n << ", N_F = " << r.nF << ", N_grad = " << r.nG
        << ", всего = " << r.nF + r.nG << "\n\n";
}

void runRosenbrock(bool restart) {
    Result r = fletcherReeves(rosenbrockF, rosenbrockGrad, -1.2, 1.0, 0.01, 0.01, restart, 5000, false);
    cout << "Розенброк, старт (-1.2; 1), перезапуск " << (restart ? "вкл" : "выкл") << ":\n";
    cout << "  n = " << r.n << ", N_F = " << r.nF << ", N_grad = " << r.nG
        << ", всего = " << r.nF + r.nG << "\n";
    cout << "  x* = (" << r.x1 << "; " << r.x2 << "), F(x*) = " << scientific << setprecision(2) << r.F
        << fixed << setprecision(6) << "\n\n";
}

int main() {
    setlocale(LC_ALL, "Russian");

    runVariant4();
    runRosenbrock(true);
    runRosenbrock(false);

    return 0;
}
