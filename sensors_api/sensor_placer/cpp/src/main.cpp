#include "mainwindow.h"

#include <QApplication>
#include <QPalette>
#include <QStyleFactory>
#include <QTimer>
#include <iostream>
#include <string>

static void applyDarkTheme(QApplication& app) {
    app.setStyle(QStyleFactory::create("Fusion"));

    QPalette darkPalette;
    const QColor darkGray(45, 45, 48);
    const QColor gray(58, 58, 60);
    const QColor accent(0, 120, 215);
    const QColor white(220, 220, 225);

    darkPalette.setColor(QPalette::Window, darkGray);
    darkPalette.setColor(QPalette::WindowText, white);
    darkPalette.setColor(QPalette::Base, QColor(30, 30, 32));
    darkPalette.setColor(QPalette::AlternateBase, gray);
    darkPalette.setColor(QPalette::ToolTipBase, gray);
    darkPalette.setColor(QPalette::ToolTipText, white);
    darkPalette.setColor(QPalette::Text, white);
    darkPalette.setColor(QPalette::Button, gray);
    darkPalette.setColor(QPalette::ButtonText, white);
    darkPalette.setColor(QPalette::BrightText, Qt::red);
    darkPalette.setColor(QPalette::Link, accent);
    darkPalette.setColor(QPalette::Highlight, accent);
    darkPalette.setColor(QPalette::HighlightedText, Qt::white);
    darkPalette.setColor(QPalette::Disabled, QPalette::Text, QColor(100, 100, 100));
    darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, QColor(100, 100, 100));

    app.setPalette(darkPalette);

    app.setStyleSheet("QToolTip { color: #dcdce1; background-color: #3a3a3c; border: 1px solid #555; }"
                      "QGroupBox { border: 1px solid #555; border-radius: 4px; margin-top: 8px; padding-top: 8px; }"
                      "QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }"
                      "QPushButton, QToolButton { background-color: #3a3a3c; border: 1px solid #555; border-radius: 4px;"
                      "  padding: 5px 14px; min-height: 20px; }"
                      "QPushButton:hover, QToolButton:hover { background-color: #505053; }"
                      "QPushButton:pressed, QToolButton:pressed { background-color: #0078d7; }"
                      "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background-color: #1e1e20; border: 1px solid #555;"
                      "  border-radius: 3px; padding: 3px 6px; min-height: 20px; }"
                      "QComboBox::drop-down { border: none; }"
                      "QComboBox QAbstractItemView { background-color: #2d2d30; selection-background-color: #0078d7; }"
                      "QScrollBar:horizontal { background: #2d2d30; height: 14px; border-radius: 7px; }"
                      "QScrollBar::handle:horizontal { background: #555; min-width: 20px; border-radius: 7px; }"
                      "QScrollBar::handle:horizontal:hover { background: #0078d7; }"
                      "QScrollBar::add-line, QScrollBar::sub-line { width: 0; }"
                      "QSlider::groove:horizontal { background: #3a3a3c; height: 6px; border-radius: 3px; }"
                      "QSlider::handle:horizontal { background: #0078d7; width: 14px; margin: -4px 0; border-radius: 7px; }"
                      "QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #777; border-radius: 3px;"
                      "  background: #1e1e20; }"
                      "QCheckBox::indicator:checked { background: #0078d7; border-color: #0078d7; }"
                      "QStatusBar { background: #2d2d30; color: #a0a0a5; }"
                      "QLabel { color: #dcdce1; }");
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
