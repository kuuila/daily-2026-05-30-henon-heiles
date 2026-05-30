/**
 * Hénon-Heiles Potential Surface Chaos Demonstrator
 * ==================================================
 * A C++17 + SFML 2.6 simulation of the classic Hénon-Heiles (1964)
 * non-integrable Hamiltonian system exhibiting chaotic dynamics.
 *
 * The potential: V(x,y) = x²·y + y³/3
 * Hamiltonian:    H = (px² + py²)/2 + V(x,y)
 *
 * Physics:
 *   dx/dt = px
 *   dy/dt = py
 *   dpx/dt = -∂V/∂x = -2xy
 *   dpy/dt = -∂V/∂y = -(x² + y²)
 *
 * Compile: g++ -std=c++17 -O2 -o henon_heiles main.cpp -lsfml-graphics -lsfml-window -lsfml-system
 */

#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>
#include <cmath>
#include <vector>
#include <array>
#include <random>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>

// ─── Constants ────────────────────────────────────────────────────────────────
constexpr double DT = 0.002;           // time step for RK4
constexpr int SKIP = 8;                // draw every N steps
constexpr int MAX_STEPS = 30000;       // max integration steps per orbit
constexpr int N_ORBITS = 12;           // number of orbits
constexpr double BOUND = 2.5;          // coordinate clamp ±BOUND
constexpr double PI = 3.14159265358979;

// Hénon-Heiles potential: V = x²y + y³/3
inline double potential(double x, double y) {
    return x * x * y + y * y * y / 3.0;
}

// Hamiltonian (kinetic + potential)
inline double hamiltonian(double x, double y, double px, double py) {
    return (px * px + py * py) * 0.5 + potential(x, y);
}

// ─── State Derivative ─────────────────────────────────────────────────────────
// state = {x, y, px, py}
// dstate/dt = f(state)
inline std::array<double, 4> derivative(const std::array<double, 4>& s) {
    double x = s[0], y = s[1], px = s[2], py = s[3];
    return { px, py, -2.0 * x * y, -(x * x + y * y) };
}

// ─── RK4 Integrator ───────────────────────────────────────────────────────────
inline std::array<double, 4> rk4(std::array<double, 4> s, double dt) {
    auto k1 = derivative(s);
    std::array<double, 4> s2 = { s[0] + dt/2*k1[0], s[1] + dt/2*k1[1],
                                  s[2] + dt/2*k1[2], s[3] + dt/2*k1[3] };
    auto k2 = derivative(s2);
    std::array<double, 4> s3 = { s[0] + dt/2*k2[0], s[1] + dt/2*k2[1],
                                  s[2] + dt/2*k2[2], s[3] + dt/2*k2[3] };
    auto k3 = derivative(s3);
    std::array<double, 4> s4 = { s[0] + dt*k3[0], s[1] + dt*k3[1],
                                  s[2] + dt*k3[2], s[3] + dt*k3[3] };
    auto k4 = derivative(s4);
    return { s[0] + dt/6*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
             s[1] + dt/6*(k1[1]+2*k2[1]+2*k3[1]+k4[1]),
             s[2] + dt/6*(k1[2]+2*k2[2]+2*k3[2]+k4[2]),
             s[3] + dt/6*(k1[3]+2*k2[3]+2*k3[3]+k4[3]) };
}

// ─── Color mapping ─────────────────────────────────────────────────────────────
sf::Color rainbow(double t) {
    t = std::fmax(0.0, std::fmin(1.0, t));
    double r = std::fmax(0.0, std::fmin(1.0, 1.5 - std::abs(6*t - 3.0)));
    double g = std::fmax(0.0, std::fmin(1.0, 1.5 - std::abs(6*t - 2.0)));
    double b = std::fmax(0.0, std::fmin(1.0, 1.5 - std::abs(6*t - 4.0)));
    return sf::Color(static_cast<sf::Uint8>(r * 220 + 35),
                     static_cast<sf::Uint8>(g * 220 + 35),
                     static_cast<sf::Uint8>(b * 220 + 35));
}

// ─── Coordinate transform ───────────────────────────────────────────────────────
// world [-BOUND, BOUND] x [-BOUND, BOUND] → pixel [0, WIN] x [0, WIN]
inline sf::Vector2f worldToPixel(double wx, double wy, unsigned winSize) {
    float f = winSize / (2.0f * BOUND);
    return sf::Vector2f( float(winSize/2) + float(wx) * f,
                         float(winSize/2) - float(wy) * f );
}

// ─── Initial conditions ────────────────────────────────────────────────────────
// Various initial conditions to probe chaos in different energy regimes
std::array<double, 4> makeIC(int idx, double E) {
    switch (idx) {
        case  0: return { 0.2,  0.0, 0.0, std::sqrt(2.0 * E) };
        case  1: return { 0.5,  0.0, 0.0, std::sqrt(2.0 * E) };
        case  2: return { 0.8,  0.0, 0.0, std::sqrt(2.0 * E) };
        case  3: return { 1.1,  0.0, 0.0, std::sqrt(2.0 * E) };
        case  4: return { 0.3,  0.3,  0.2, -0.2 };
        case  5: return { 0.6,  0.2, -0.3,  0.1 };
        case  6: return { 0.9,  0.15, 0.1, -0.5 };
        case  7: return { 0.4, -0.4,  0.2,  0.4 };
        case  8: return { 1.3,  0.0, 0.0, std::sqrt(2.0 * E - potential(1.3,0.0)) };
        case  9: return { 0.0,  0.6,  0.0, std::sqrt(2.0 * E) };
        case 10: return { 0.7,  0.5, -0.1, -0.3 };
        case 11: return { 1.0, -0.3,  0.2,  0.3 };
        default: return { 0.2,  0.0, 0.0, std::sqrt(2.0 * E) };
    }
}

// ─── Terminal mode output ──────────────────────────────────────────────────────
void runTerminalMode() {
    std::cout << "\n";
    std::cout << "══════════════════════════════════════════════════════\n";
    std::cout << "    Hénon-Heiles 势能面混沌演示器  ·  终端模式\n";
    std::cout << "══════════════════════════════════════════════════════\n\n";
    std::cout << "  势能函数: V(x,y) = x²y + y³/3\n";
    std::cout << "  Hamiltonian:  H = (px²+py²)/2 + V(x,y)\n";
    std::cout << "  积分器: 4阶 Runge-Kutta (RK4)\n";
    std::cout << "  时间步长: dt = " << DT << "\n\n";

    std::vector<double> energies = {0.08, 0.10, 0.12, 0.14, 0.16, 0.18};
    std::vector<std::array<double, 4>> orbitICs = {
        {0.2, 0.0, 0.0, 0.0}, {0.5, 0.0, 0.0, 0.0},
        {0.8, 0.0, 0.0, 0.0}, {1.1, 0.0, 0.0, 0.0},
    };

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // ── Draw ASCII potential surface ─────────────────────────────────────────
    std::cout << "  等势能线 ASCII 图 (V = x²y + y³/3)\n";
    std::cout << "  范围: x,y ∈ [-2, 2]\n\n";

    const int W = 60, H = 30;
    double xMin = -2.0, xMax = 2.0, yMin = -2.0, yMax = 2.0;
    double dx = (xMax - xMin) / W, dy = (yMax - yMin) / H;

    for (int j = 0; j < H; ++j) {
        double y = yMax - j * dy;
        std::string line;
        for (int i = 0; i < W; ++i) {
            double x = xMin + i * dx;
            double V = potential(x, y);
            char c = '.';
            if (V >  0.15) c = '#';
            else if (V >  0.05) c = '=';
            else if (V >  0.0)  c = '-';
            else if (V > -0.05) c = '.';
            else if (V > -0.15) c = '+';
            else if (V > -0.3)  c = '*';
            else                c = '@';
            line += c;
        }
        printf("  %2.1f %s\n", y, line.c_str());
    }
    std::cout << "       -2.0                      +2.0 (x)\n\n";

    // ── Integrate orbits ─────────────────────────────────────────────────────
    std::cout << "  轨道积分测试 (dt=" << DT << ", max steps=" << MAX_STEPS << ")\n\n";

    for (int i = 0; i < 8; ++i) {
        auto ic = makeIC(i, 0.10);
        double x = ic[0], y = ic[1], px = ic[2], py = ic[3];
        double E0 = hamiltonian(x, y, px, py);
        double x_min = x, x_max = x, y_min = y, y_max = y;

        for (int step = 0; step < MAX_STEPS; ++step) {
            std::array<double, 4> s = {x, y, px, py};
            auto sn = rk4(s, DT);
            x = sn[0]; y = sn[1]; px = sn[2]; py = sn[3];
            // escape / NaN guard
            if (!std::isfinite(x) || !std::isfinite(y)) break;
            if (std::abs(x) > BOUND * 3 || std::abs(y) > BOUND * 3) break;
            if (x < x_min) x_min = x;
            if (x > x_max) x_max = x;
            if (y < y_min) y_min = y;
            if (y > y_max) y_max = y;
        }
        double E1 = hamiltonian(x, y, px, py);
        printf("  IC[%d]  E0=%.6f  E_final=%.6f  ΔE/E=%.2e  range: x[%.3f,%.3f] y[%.3f,%.3f]\n",
               i, E0, E1, (E1-E0)/(E0==0?1:E0),
               x_min, x_max, y_min, y_max);
    }

    std::cout << "\n  终端模式完成。运行 ./henon_heiles 启动 SFML 图形模式。\n";
    std::cout << "══════════════════════════════════════════════════════\n";
}

// ─── Main ──────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {

    // --terminal flag
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--terminal" || std::string(argv[i]) == "-t") {
            runTerminalMode();
            return 0;
        }
    }

    const unsigned WIN = 900;
    sf::RenderWindow window(sf::VideoMode(WIN, WIN),
                            "Hénon-Heiles Chaos · butterfly effect in mechanics",
                            sf::Style::Close);

    // ─── Font (use built-in if available, else minimal) ───────────────────────
    sf::Font font;
    bool hasFont = font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
                || font.loadFromFile("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf")
                || font.loadFromFile("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf");

    auto makeText = [&](const std::string& str, unsigned sz, sf::Color col, float x, float y) {
        sf::Text t;
        if (hasFont) t.setFont(font);
        t.setCharacterSize(sz);
        t.setFillColor(col);
        t.setString(str);
        t.setPosition(x, y);
        return t;
    };

    // ─── Prepare orbit vertex arrays ───────────────────────────────────────
    std::vector<sf::VertexArray> orbitLines;
    std::vector<sf::Color>       orbitColors;
    std::vector<double>          orbitEnergies;
    std::vector<int>             orbitPoincareCounts;

    for (int i = 0; i < N_ORBITS; ++i) {
        orbitLines.emplace_back(sf::LinesStrip, MAX_STEPS / SKIP);
        orbitColors.push_back(rainbow(double(i) / N_ORBITS));
        orbitEnergies.push_back(0.0);
        orbitPoincareCounts.push_back(0);
    }

    // Poincaré section: y ≈ 0, py > 0  (crossing plane in yellow)
    std::vector<sf::VertexArray> poincareSections;
    for (int i = 0; i < N_ORBITS; ++i)
        poincareSections.emplace_back(sf::Points, 2000);

    // ─── Integrate all orbits ───────────────────────────────────────────────
    std::vector<std::array<double, 4>> states(N_ORBITS);
    for (int i = 0; i < N_ORBITS; ++i) {
        states[i] = makeIC(i, 0.12);
        orbitEnergies[i] = hamiltonian(states[i][0], states[i][1], states[i][2], states[i][3]);
    }

    // Pre-compute poincare y threshold
    const double POINCARE_Y_THRESH = 0.02;

    for (int step = 0; step < MAX_STEPS; ++step) {
        for (int i = 0; i < N_ORBITS; ++i) {
            states[i] = rk4(states[i], DT);

            double x = states[i][0], y = states[i][1];
            double px = states[i][2], py = states[i][3];

            // bounding / escape
            if (!std::isfinite(x) || !std::isfinite(y)) break;
            if (std::abs(x) > BOUND * 2.5 || std::abs(y) > BOUND * 2.5) break;

            if (step % SKIP == 0) {
                auto idx = (step / SKIP);
                if (idx < int(orbitLines[i].getVertexCount()))
                    orbitLines[i][idx] = sf::Vertex(worldToPixel(x, y, WIN), orbitColors[i]);
            }

            // Poincaré section: y crosses threshold from negative to positive
            if (step > 0 && py > 0 && std::abs(y) < POINCARE_Y_THRESH) {
                auto& ps = poincareSections[i];
                int cnt = orbitPoincareCounts[i];
                if (cnt < int(ps.getVertexCount())) {
                    sf::Color pc = orbitColors[i];
                    pc.a = 200;
                    ps[cnt] = sf::Vertex(worldToPixel(x, py, WIN), pc);
                    orbitPoincareCounts[i]++;
                }
            }
        }
    }

    // ─── Draw potential contours (precomputed) ───────────────────────────────
    sf::VertexArray contours(sf::Lines);
    const int CONTOUR_LEVELS = 12;
    const double V_MIN = -0.5, V_MAX = 0.5;
    const int GRID = 80;

    for (int i = 0; i < GRID; ++i) {
        for (int j = 0; j < GRID; ++j) {
            double x0 = -BOUND + 2.0 * BOUND * i / GRID;
            double y0 = -BOUND + 2.0 * BOUND * j / GRID;
            double x1 = -BOUND + 2.0 * BOUND * (i+1) / GRID;
            double y1 = -BOUND + 2.0 * BOUND * (j+1) / GRID;
            double V00 = potential(x0, y0);
            double V01 = potential(x0, y1);
            double V10 = potential(x1, y0);
            double V11 = potential(x1, y1);

            for (int lv = 0; lv < CONTOUR_LEVELS; ++lv) {
                double Vlv = V_MIN + (V_MAX - V_MIN) * lv / CONTOUR_LEVELS;
                // Marching squares at this level — simplified linear interp
                std::vector<std::pair<double,double>> pts;
                if ((V00 > Vlv) != (V10 > Vlv)) {
                    double t = (Vlv - V00) / (V10 - V00 + 1e-12);
                    pts.emplace_back(x0 + t*(x1-x0), y0);
                }
                if ((V10 > Vlv) != (V11 > Vlv)) {
                    double t = (Vlv - V10) / (V11 - V10 + 1e-12);
                    pts.emplace_back(x1, y0 + t*(y1-y0));
                }
                if ((V11 > Vlv) != (V01 > Vlv)) {
                    double t = (Vlv - V11) / (V01 - V11 + 1e-12);
                    pts.emplace_back(x1 - t*(x1-x0), y1);
                }
                if ((V01 > Vlv) != (V00 > Vlv)) {
                    double t = (Vlv - V01) / (V00 - V01 + 1e-12);
                    pts.emplace_back(x0, y1 - t*(y1-y0));
                }
                if (pts.size() == 2) {
                    contours.append(sf::Vertex(worldToPixel(pts[0].first,  pts[0].second, WIN),
                                              sf::Color(100, 100, 100, 60)));
                    contours.append(sf::Vertex(worldToPixel(pts[1].first,  pts[1].second, WIN),
                                              sf::Color(100, 100, 100, 60)));
                }
            }
        }
    }

    // ─── Main loop ───────────────────────────────────────────────────────────
    while (window.isOpen()) {
        sf::Event ev;
        while (window.pollEvent(ev)) {
            if (ev.type == sf::Event::Closed ||
               (ev.type == sf::Event::KeyPressed && ev.key.code == sf::Keyboard::Escape))
                window.close();
        }

        window.clear(sf::Color(12, 10, 20));

        // background grid
        sf::VertexArray grid(sf::Lines);
        for (int i = 0; i <= 10; ++i) {
            double v = -BOUND + 2.0 * BOUND * i / 10;
            grid.append(sf::Vertex(worldToPixel(v, -BOUND, WIN), sf::Color(40,40,55,80)));
            grid.append(sf::Vertex(worldToPixel(v,  BOUND, WIN), sf::Color(40,40,55,80)));
            grid.append(sf::Vertex(worldToPixel(-BOUND, v, WIN), sf::Color(40,40,55,80)));
            grid.append(sf::Vertex(worldToPixel( BOUND, v, WIN), sf::Color(40,40,55,80)));
        }
        window.draw(grid);

        // axes
        sf::VertexArray axes(sf::Lines, 2);
        axes[0] = sf::Vertex(worldToPixel(-BOUND, 0, WIN), sf::Color(80,80,100,120));
        axes[1] = sf::Vertex(worldToPixel( BOUND, 0, WIN), sf::Color(80,80,100,120));
        axes.append(sf::Vertex(worldToPixel(0, -BOUND, WIN), sf::Color(80,80,100,120)));
        axes.append(sf::Vertex(worldToPixel(0,  BOUND, WIN), sf::Color(80,80,100,120)));
        window.draw(axes);

        // potential contours
        window.draw(contours);

        // orbit trails
        for (int i = 0; i < N_ORBITS; ++i)
            window.draw(orbitLines[i]);

        // Poincaré sections (lower panel)
        sf::VertexArray poincareAll(sf::Points);
        for (int i = 0; i < N_ORBITS; ++i) {
            for (int j = 0; j < orbitPoincareCounts[i]; ++j)
                poincareAll.append(poincareSections[i][j]);
        }
        // Draw poincare in a separate strip at the bottom
        // We'll show it in a side panel by shifting y
        sf::VertexArray poincareShifted(sf::Points);
        for (int i = 0; i < N_ORBITS; ++i) {
            for (int j = 0; j < orbitPoincareCounts[i]; ++j) {
                auto v = poincareSections[i][j];
                // shift to bottom 200px strip
                sf::Vertex vs(v.position + sf::Vector2f(0, float(WIN - 200)),
                              v.color);
                poincareShifted.append(vs);
            }
        }
        window.draw(poincareShifted);

        // info overlay
        if (hasFont) {
            auto title = makeText("Hénon-Heiles Chaos · butterfly effect in Hamiltonian mechanics",
                                  16, sf::Color(200,200,230), 12, 12);
            window.draw(title);

            std::ostringstream oss;
            oss << std::fixed << std::setprecision(4);
            oss << "E[0]=" << orbitEnergies[0];
            auto eText = makeText(oss.str(), 13, sf::Color(150,200,150), 12, 34);
            window.draw(eText);

            auto legend = makeText("colors = different ICs  |  lines = orbits  |  dots = Poincaré section (y≈0, py>0)",
                                  12, sf::Color(120,120,150), 12, WIN - 24);
            window.draw(legend);
        }

        window.display();
        sf::sleep(sf::milliseconds(10));
    }

    return 0;
}
