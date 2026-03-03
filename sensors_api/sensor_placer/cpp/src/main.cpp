#include "mainwindow.h"

#include <QApplication>
#include <QPalette>
#include <QStyleFactory>
#include <QTimer>
#include <iostream>
#include <string>

namespace sl_clr {
    constexpr auto lime = "rgb(217,255,66)";      // #d9ff42  – company accent
    constexpr auto dark_lime = "rgb(187,225,36)"; // #bbe124  – hover accent
    constexpr auto soil = "rgb(25,25,25)";        // #191919  – deepest bg
    constexpr auto sl_black = "rgb(5,5,5)";       // #050505
    constexpr auto steel = "rgb(45,45,45)";       // #2d2d2d  – secondary bg
    constexpr auto charcoal = "rgb(60,60,60)";    // #3c3c3c  – widget bg
    constexpr auto ash = "rgb(137,137,137)";      // #898989  – disabled / muted
    constexpr auto iron = "rgb(194,194,194)";     // #c2c2c2  – borders
    constexpr auto pearl = "rgb(242,242,242)";    // #f2f2f2  – primary text
    constexpr auto white = "rgb(255,255,255)";    // #ffffff
    constexpr auto demo_red = "rgb(255,100,100)"; // #ff6464
} // namespace sl_clr

static void applyDarkTheme(QApplication& app) {
    app.setStyle(QStyleFactory::create("Fusion"));

    QPalette p;
    const QColor soil(25, 25, 25);
    const QColor steel(45, 45, 45);
    const QColor charcoal(60, 60, 60);
    const QColor ash(137, 137, 137);
    const QColor pearl(242, 242, 242);
    const QColor lime(217, 255, 66);

    p.setColor(QPalette::Window, soil);
    p.setColor(QPalette::WindowText, pearl);
    p.setColor(QPalette::Base, steel);
    p.setColor(QPalette::AlternateBase, charcoal);
    p.setColor(QPalette::ToolTipBase, charcoal);
    p.setColor(QPalette::ToolTipText, pearl);
    p.setColor(QPalette::Text, pearl);
    p.setColor(QPalette::Button, charcoal);
    p.setColor(QPalette::ButtonText, pearl);
    p.setColor(QPalette::BrightText, Qt::white);
    p.setColor(QPalette::Link, lime);
    p.setColor(QPalette::Highlight, lime);
    p.setColor(QPalette::HighlightedText, soil);
    p.setColor(QPalette::Disabled, QPalette::Text, ash);
    p.setColor(QPalette::Disabled, QPalette::ButtonText, ash);
    p.setColor(QPalette::Mid, ash); // used by spin-box arrows
    p.setColor(QPalette::Light, QColor(75, 75, 75));

    app.setPalette(p);

    // ── Global stylesheet ───────────────────
    app.setStyleSheet(
        /* ── Tooltips ── */
        "QToolTip { color: #f2f2f2; background-color: #3c3c3c; border: 1px solid #c2c2c2; }"

        /* ── Group boxes ── */
        "QGroupBox { border: 1px solid #3c3c3c; border-radius: 4px;"
        "  margin-top: 10px; padding-top: 10px; background-color: #191919; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px;"
        "  color: #c2c2c2; }"

        /* ── Buttons – lime outline ── */
        "QPushButton, QToolButton { color: #d9ff42; background-color: #191919;"
        "  border: 2px solid #d9ff42; border-radius: 4px; padding: 5px 14px; min-height: 20px; }"
        "QPushButton:hover, QToolButton:hover { background-color: #bbe124;"
        "  color: #191919; border-color: #bbe124; }"
        "QPushButton:pressed, QToolButton:pressed { background-color: #a8cc1e;"
        "  color: #191919; border-color: #a8cc1e; }"
        "QPushButton:disabled, QToolButton:disabled { color: #f2f2f2;"
        "  background-color: #191919; border: 1px solid #898989; }"

        /* ── Inputs ── */
        "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { color: #f2f2f2;"
        "  background-color: #191919; border: 1px solid #c2c2c2;"
        "  border-radius: 4px; padding: 3px 6px; min-height: 20px; }"

        /* ── Combo box drop-down ── */
        "QComboBox::drop-down { border: 0px; subcontrol-origin: padding;"
        "  subcontrol-position: top right; width: 20px; }"
        "QComboBox QAbstractItemView { color: #f2f2f2; background-color: #191919;"
        "  border: 1px solid #c2c2c2; selection-color: #d9ff42;"
        "  selection-background-color: #2d2d2d; }"

        /* ── Scroll bars ── */
        "QScrollBar:horizontal { background: #2d2d2d; height: 14px; border-radius: 7px; }"
        "QScrollBar::handle:horizontal { background: #898989; min-width: 20px; border-radius: 7px; }"
        "QScrollBar::handle:horizontal:hover { background: #d9ff42; }"
        "QScrollBar::add-line, QScrollBar::sub-line { width: 0; }"

        /* ── Slider – lime filled track ── */
        "QSlider::groove:horizontal { background: #898989; height: 3px; border-radius: 1px; }"
        "QSlider::handle:horizontal { background: #d9ff42; width: 14px;"
        "  margin: -6px 0; border-radius: 7px; }"
        "QSlider::handle:horizontal:hover { background: #bbe124; }"
        "QSlider::sub-page:horizontal { background: #d9ff42; }"
        "QSlider::add-page:horizontal { background: #898989; }"

        /* ── Check boxes ── */
        "QCheckBox { color: #f2f2f2; }"
        "QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #c2c2c2;"
        "  border-radius: 3px; background: #191919; }"
        "QCheckBox::indicator:checked { background: #d9ff42; border-color: #d9ff42; }"

        /* ── Dials ── */
        "QDial { background-color: #2d2d2d; }"

        /* ── Status bar ── */
        "QStatusBar { background: #2d2d2d; color: #898989; }"

        /* ── Labels ── */
        "QLabel { color: #f2f2f2; }"
    );
}

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    applyDarkTheme(app);

    MainWindow w;

    // Parse command-line args
    std::string configPath;
    std::string outputPath;
    bool doAutoDetect = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [sensors_config.json] [-o output.json] [--auto]\n"
                      << "  sensors_config.json   Load sensors from config file\n"
                      << "  -o, --output FILE     Output file path (default: <input>_placed.json)\n"
                      << "  --auto                Auto-detect all connected sensors\n";
            return 0;
        }
        if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--auto") {
            doAutoDetect = true;
        } else {
            configPath = arg;
        }
    }

    if (!outputPath.empty())
        w.setOutputPath(outputPath);

    w.show();

    // Defer sensor loading until after show() so initializeGL() has run
    if (!configPath.empty()) {
        QTimer::singleShot(0, &w, [&w, configPath]() {
            w.loadConfig(configPath);
        });
    } else if (doAutoDetect) {
        QTimer::singleShot(0, &w, [&w]() {
            w.autoDetect();
        });
    }

    return app.exec();
}
