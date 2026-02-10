#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTimer>
#include <QColor>

#include "SensorWorker.h"

#include <map>
#include <vector>
#include <random>

QT_BEGIN_NAMESPACE
namespace Ui {
    class MainWindow;
}
QT_END_NAMESPACE

struct SensorData {
    SensorWorker worker;
    sl::Transform pose;
    QColor color;
    SensorType type = SensorType::ZED;
    bool poseLoaded = false; // true when pose was read from config JSON

    // JSON-preserved fields
    int serial = 0;
    std::string ip;
    std::string svo; // SVO2 path for ZED
    std::string osf; // OSF path for LiDAR
    int fps = 0;
    std::vector<int> resolution;
};

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(QWidget* parent = nullptr);
    ~MainWindow() override;

    // Load / auto-detect sensors
    bool loadConfig(const std::string& path);
    void autoDetect();
    void setOutputPath(const std::string& path) {
        outputPath_ = path;
    }

protected:
    void closeEvent(QCloseEvent* event) override;

private slots:
    void updateUI();

    // Sensor selection
    void on_combo_select_currentIndexChanged(int index);
    void on_push_remove_released();
    void on_check_hide_toggled(bool checked);

    // Add sensors
    void on_button_serial_clicked();
    void on_button_ip_clicked();
    void on_button_svo_clicked();
    void on_button_autodetect_clicked();

    // Rotation dials
    void on_dial_x_valueChanged(int value);
    void on_dial_y_valueChanged(int value);
    void on_dial_z_valueChanged(int value);

    // Rotation spin boxes
    void on_spin_rx_valueChanged(double v);
    void on_spin_ry_valueChanged(double v);
    void on_spin_rz_valueChanged(double v);

    // Translation scrollbars
    void on_scroll_tx_valueChanged(int v);
    void on_scroll_ty_valueChanged(int v);
    void on_scroll_tz_valueChanged(int v);

    // Translation spin boxes
    void on_spin_tx_valueChanged(double v);
    void on_spin_ty_valueChanged(double v);
    void on_spin_tz_valueChanged(double v);

    // Rendering options
    void on_check_edges_stateChanged(int state);
    void on_check_clr_stateChanged(int state);
    void on_slider_ptsize_valueChanged(int value);

    // Actions
    void on_button_findPlane_clicked();
    void on_button_imuPose_clicked();
    void on_button_export_clicked();
    void on_button_load_clicked();

private:
    Ui::MainWindow* ui;
    QTimer* uiTimer_;

    std::map<int, SensorData> sensors_;
    std::map<int, int> comboMap_; // combo index → sensor id
    std::map<int, bool> sensorHidden_;
    int currentId_ = 0;
    int nextId_ = 0; // monotonic sensor id counter

    void connectToViewer(int id);
    int getComboIdx(int sensorId);

    // Palette
    QColor generateColor();
    std::vector<QColor> palette_;
    std::mt19937 rng_;
    std::uniform_int_distribution<uint32_t> dist100_;

    // Config paths
    std::string configPath_;
    std::string outputPath_;

    // JSON I/O
    void saveConfig(const std::string& path);
};

#endif // MAINWINDOW_H
