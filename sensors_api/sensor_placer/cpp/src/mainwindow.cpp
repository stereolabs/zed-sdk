#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "GlViewer.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <unordered_set>

#include <QFileDialog>
#include <QMessageBox>
#include <QBrush>
#include <QPalette>
#include <QCloseEvent>

// Inline JSON parser (nlohmann/json single-header)
// We use a minimal approach: parse manually with Qt's JSON or a small lib.
// For simplicity, use the same json.hpp already in the project.
#include <nlohmann/json.hpp>
using json = nlohmann::json;

static const float ROT_SCALE = 0.1f;     // dial step → degrees
static const float TRANS_SCALE = 0.005f; // scrollbar step → meters

// ─── Color palette ───────────────────────────────────────────────────────────

static double colorDeltaE(QColor c1, QColor c2) {
    auto toXYZ = [](QColor c) -> sl::float3 {
        float r = c.redF(), g = c.greenF(), b = c.blueF();
        auto gamma = [](float v) {
            return v > 0.04045f ? powf((v + 0.055f) / 1.055f, 2.4f) : v / 12.92f;
        };
        r = gamma(r) * 100;
        g = gamma(g) * 100;
        b = gamma(b) * 100;
        return {r * 0.4124f + g * 0.3576f + b * 0.1805f, r * 0.2126f + g * 0.7152f + b * 0.0722f, r * 0.0193f + g * 0.1192f + b * 0.9505f};
    };
    auto toLab = [](sl::float3 c) -> sl::float3 {
        float x = c.x / 95.047f, y = c.y / 100.f, z = c.z / 108.883f;
        auto f = [](float v) {
            return v > 0.008856f ? powf(v, 1.f / 3.f) : 7.787f * v + 16.f / 116.f;
        };
        x = f(x);
        y = f(y);
        z = f(z);
        return {116.f * y - 16.f, 500.f * (x - y), 200.f * (y - z)};
    };
    auto lab1 = toLab(toXYZ(c1));
    auto lab2 = toLab(toXYZ(c2));
    return sl::float3::distance(lab1, lab2);
}

// ─── MainWindow ──────────────────────────────────────────────────────────────

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow) {
    ui->setupUi(this);
    setWindowTitle("Sensor Placer");

    uiTimer_ = new QTimer(this);
    connect(uiTimer_, &QTimer::timeout, this, &MainWindow::updateUI);
    uiTimer_->start(25);

    ui->glWidget->setAvailable(true);
    on_slider_ptsize_valueChanged(25);

    std::random_device dev;
    rng_ = std::mt19937(dev());
    dist100_ = std::uniform_int_distribution<uint32_t>(0, 100);
}

MainWindow::~MainWindow() {
    uiTimer_->stop();
    for (auto& [id, s] : sensors_)
        s.worker.close();
    delete uiTimer_;
    delete ui;
}

void MainWindow::closeEvent(QCloseEvent* event) {
    // Auto-save when an output path was provided via -o
    if (!outputPath_.empty() && !sensors_.empty()) {
        saveConfig(outputPath_);
        std::cout << "Auto-saved to: " << outputPath_ << std::endl;
    }
    QMainWindow::closeEvent(event);
}

// ─── Update loop ─────────────────────────────────────────────────────────────

void MainWindow::updateUI() {
    for (auto& [id, s] : sensors_) {
        bool needsUp = s.worker.needsUpdate();
        if (s.worker.type() == SensorType::LIDAR && needsUp) {
            sl::Mat pc = s.worker.getLidarPC();
            if (pc.isInit())
                ui->glWidget->updateLidarCloud(id, pc);
        }
        ui->glWidget->updateSensor(id, s.pose, needsUp);
    }
    ui->glWidget->update();
    update();
}

// ─── Rotation dials ──────────────────────────────────────────────────────────

void MainWindow::on_dial_x_valueChanged(int value) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto rot = sensors_[currentId_].pose.getEulerAngles();
    rot.x = deg2rad(value * ROT_SCALE);
    sensors_[currentId_].pose.setEulerAngles(rot);
    ui->spin_rx->blockSignals(true);
    ui->spin_rx->setValue(rad2deg(rot.x));
    ui->spin_rx->blockSignals(false);
}

void MainWindow::on_dial_y_valueChanged(int value) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto rot = sensors_[currentId_].pose.getEulerAngles();
    rot.y = deg2rad(value * ROT_SCALE);
    sensors_[currentId_].pose.setEulerAngles(rot);
    ui->spin_ry->blockSignals(true);
    ui->spin_ry->setValue(rad2deg(rot.y));
    ui->spin_ry->blockSignals(false);
}

void MainWindow::on_dial_z_valueChanged(int value) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto rot = sensors_[currentId_].pose.getEulerAngles();
    rot.z = deg2rad(value * ROT_SCALE);
    sensors_[currentId_].pose.setEulerAngles(rot);
    ui->spin_rz->blockSignals(true);
    ui->spin_rz->setValue(rad2deg(rot.z));
    ui->spin_rz->blockSignals(false);
}

// ─── Rotation spin boxes ─────────────────────────────────────────────────────

void MainWindow::on_spin_rx_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto rot = sensors_[currentId_].pose.getEulerAngles();
    rot.x = deg2rad(v);
    sensors_[currentId_].pose.setEulerAngles(rot);
    ui->dial_x->blockSignals(true);
    ui->dial_x->setValue((int)(v / ROT_SCALE));
    ui->dial_x->blockSignals(false);
}

void MainWindow::on_spin_ry_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto rot = sensors_[currentId_].pose.getEulerAngles();
    rot.y = deg2rad(v);
    sensors_[currentId_].pose.setEulerAngles(rot);
    ui->dial_y->blockSignals(true);
    ui->dial_y->setValue((int)(v / ROT_SCALE));
    ui->dial_y->blockSignals(false);
}

void MainWindow::on_spin_rz_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto rot = sensors_[currentId_].pose.getEulerAngles();
    rot.z = deg2rad(v);
    sensors_[currentId_].pose.setEulerAngles(rot);
    ui->dial_z->blockSignals(true);
    ui->dial_z->setValue((int)(v / ROT_SCALE));
    ui->dial_z->blockSignals(false);
}

// ─── Translation scrollbars ─────────────────────────────────────────────────

void MainWindow::on_scroll_tx_valueChanged(int v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    sensors_[currentId_].pose.tx = v * TRANS_SCALE;
    ui->spin_tx->blockSignals(true);
    ui->spin_tx->setValue(v * TRANS_SCALE);
    ui->spin_tx->blockSignals(false);
}

void MainWindow::on_scroll_ty_valueChanged(int v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    sensors_[currentId_].pose.ty = v * TRANS_SCALE;
    ui->spin_ty->blockSignals(true);
    ui->spin_ty->setValue(v * TRANS_SCALE);
    ui->spin_ty->blockSignals(false);
}

void MainWindow::on_scroll_tz_valueChanged(int v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    sensors_[currentId_].pose.tz = v * TRANS_SCALE;
    ui->spin_tz->blockSignals(true);
    ui->spin_tz->setValue(v * TRANS_SCALE);
    ui->spin_tz->blockSignals(false);
}

// ─── Translation spin boxes ─────────────────────────────────────────────────

void MainWindow::on_spin_tx_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    sensors_[currentId_].pose.tx = v;
    ui->scroll_tx->blockSignals(true);
    ui->scroll_tx->setValue((int)(v / TRANS_SCALE));
    ui->scroll_tx->blockSignals(false);
}

void MainWindow::on_spin_ty_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    sensors_[currentId_].pose.ty = v;
    ui->scroll_ty->blockSignals(true);
    ui->scroll_ty->setValue((int)(v / TRANS_SCALE));
    ui->scroll_ty->blockSignals(false);
}

void MainWindow::on_spin_tz_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    sensors_[currentId_].pose.tz = v;
    ui->scroll_tz->blockSignals(true);
    ui->scroll_tz->setValue((int)(v / TRANS_SCALE));
    ui->scroll_tz->blockSignals(false);
}

// ─── Sensor selection ────────────────────────────────────────────────────────

void MainWindow::on_combo_select_currentIndexChanged(int index) {
    if (comboMap_.find(index) == comboMap_.end())
        return;
    currentId_ = comboMap_[index];

    if (sensorHidden_.count(index))
        ui->check_hide->setChecked(sensorHidden_[index]);

    auto& s = sensors_[currentId_];
    auto rot = s.pose.getEulerAngles();
    auto trans = s.pose.getTranslation();

    // Block signals to prevent feedback loops
    ui->dial_x->blockSignals(true);
    ui->dial_x->setValue((int)(rad2deg(rot.x) / ROT_SCALE));
    ui->dial_x->blockSignals(false);
    ui->dial_y->blockSignals(true);
    ui->dial_y->setValue((int)(rad2deg(rot.y) / ROT_SCALE));
    ui->dial_y->blockSignals(false);
    ui->dial_z->blockSignals(true);
    ui->dial_z->setValue((int)(rad2deg(rot.z) / ROT_SCALE));
    ui->dial_z->blockSignals(false);

    ui->spin_rx->blockSignals(true);
    ui->spin_rx->setValue(rad2deg(rot.x));
    ui->spin_rx->blockSignals(false);
    ui->spin_ry->blockSignals(true);
    ui->spin_ry->setValue(rad2deg(rot.y));
    ui->spin_ry->blockSignals(false);
    ui->spin_rz->blockSignals(true);
    ui->spin_rz->setValue(rad2deg(rot.z));
    ui->spin_rz->blockSignals(false);

    ui->scroll_tx->blockSignals(true);
    ui->scroll_tx->setValue((int)(trans.tx / TRANS_SCALE));
    ui->scroll_tx->blockSignals(false);
    ui->scroll_ty->blockSignals(true);
    ui->scroll_ty->setValue((int)(trans.ty / TRANS_SCALE));
    ui->scroll_ty->blockSignals(false);
    ui->scroll_tz->blockSignals(true);
    ui->scroll_tz->setValue((int)(trans.tz / TRANS_SCALE));
    ui->scroll_tz->blockSignals(false);

    ui->spin_tx->blockSignals(true);
    ui->spin_tx->setValue(trans.tx);
    ui->spin_tx->blockSignals(false);
    ui->spin_ty->blockSignals(true);
    ui->spin_ty->setValue(trans.ty);
    ui->spin_ty->blockSignals(false);
    ui->spin_tz->blockSignals(true);
    ui->spin_tz->setValue(trans.tz);
    ui->spin_tz->blockSignals(false);

    // Enable/disable ZED-only buttons
    bool isZed = (s.type == SensorType::ZED);
    ui->button_findPlane->setEnabled(isZed);
    ui->check_edges->setEnabled(isZed);
}

void MainWindow::on_push_remove_released() {
    int comboIdx = getComboIdx(currentId_);
    if (comboIdx < 0)
        return;

    sensors_[currentId_].worker.close();
    sensors_.erase(currentId_);
    ui->glWidget->removeSensor(currentId_);

    ui->combo_select->removeItem(comboIdx);
    if (ui->combo_select->count() > 0)
        ui->combo_select->setCurrentIndex(0);
}

void MainWindow::on_check_hide_toggled(bool checked) {
    int comboIdx = getComboIdx(currentId_);
    if (comboIdx >= 0)
        sensorHidden_[comboIdx] = checked;
    ui->glWidget->hideSensor(currentId_, checked);
}

// ─── Add sensors ─────────────────────────────────────────────────────────────

void MainWindow::on_button_serial_clicked() {
    int serial = ui->edit_serial->text().toInt();
    if (serial <= 0)
        return;
    int id = nextId_++;
    sensors_[id].type = SensorType::ZED;
    sensors_[id].serial = serial;
    if (sensors_[id].worker.openZedSerial(serial))
        connectToViewer(id);
    else {
        sensors_.erase(id);
        statusBar()->showMessage("Failed to open ZED S/N " + QString::number(serial), 5000);
    }
}

void MainWindow::on_button_ip_clicked() {
    std::string ip = ui->edit_ip->text().toStdString();
    if (ip.empty())
        return;
    int id = nextId_++;
    sensors_[id].type = SensorType::LIDAR;
    sensors_[id].ip = ip;
    if (sensors_[id].worker.openLidarIP(ip))
        connectToViewer(id);
    else {
        sensors_.erase(id);
        statusBar()->showMessage("Failed to open LiDAR at " + QString::fromStdString(ip), 5000);
    }
}

void MainWindow::on_button_svo_clicked() {
    std::string path = ui->edit_svo->text().toStdString();
    if (path.empty())
        return;
    int id = nextId_++;

    // Guess type from extension
    bool isOSF = (path.find(".osf") != std::string::npos);
    if (isOSF) {
        sensors_[id].type = SensorType::LIDAR;
        if (sensors_[id].worker.openLidarOSF(path))
            connectToViewer(id);
        else {
            sensors_.erase(id);
        }
    } else {
        sensors_[id].type = SensorType::ZED;
        if (sensors_[id].worker.openZedSVO(path))
            connectToViewer(id);
        else {
            sensors_.erase(id);
        }
    }
}

void MainWindow::on_button_autodetect_clicked() {
    statusBar()->showMessage("Detecting sensors…", 2000);
    autoDetect();
    if (sensors_.empty())
        statusBar()->showMessage("No sensors found", 5000);
    else
        statusBar()->showMessage(QString::number(sensors_.size()) + " sensor(s) detected", 5000);
}

// ─── Rendering options ──────────────────────────────────────────────────────

void MainWindow::on_check_edges_stateChanged(int state) {
    for (auto& [id, s] : sensors_) {
        if (s.type == SensorType::ZED)
            s.worker.setTextureOnly(state != 0);
    }
}

void MainWindow::on_check_clr_stateChanged(int state) {
    ui->glWidget->drawColors(state != 0);
}

void MainWindow::on_slider_ptsize_valueChanged(int value) {
    float v = value / 10.f;
    ui->glWidget->setPointSize(v);
    ui->label_ptval->setText(QString::number(v, 'f', 1));
}

// ─── Actions ─────────────────────────────────────────────────────────────────

void MainWindow::on_button_findPlane_clicked() {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto& s = sensors_[currentId_];
    if (s.type != SensorType::ZED)
        return;

    statusBar()->showMessage("Finding floor plane...");
    if (s.worker.findFloorPlane(s.pose)) {
        int comboIdx = getComboIdx(currentId_);
        if (comboIdx >= 0)
            on_combo_select_currentIndexChanged(comboIdx);
        statusBar()->showMessage("Floor plane found", 3000);
    } else {
        statusBar()->showMessage("Floor plane not found", 3000);
    }
}

void MainWindow::on_button_imuPose_clicked() {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto& s = sensors_[currentId_];
    if (s.type != SensorType::ZED)
        return;

    statusBar()->showMessage("Getting IMU gravity pose…");
    if (s.worker.getIMUPose(s.pose)) {
        int comboIdx = getComboIdx(currentId_);
        if (comboIdx >= 0)
            on_combo_select_currentIndexChanged(comboIdx);
        statusBar()->showMessage("IMU gravity applied (pitch/roll)", 3000);
    } else {
        statusBar()->showMessage("Failed to get IMU pose", 3000);
    }
}

void MainWindow::on_button_load_clicked() {
    auto fileName = QFileDialog::getOpenFileName(this, "Open Config File", "", "JSON Files (*.json)");
    if (fileName.isEmpty())
        return;
    loadConfig(fileName.toStdString());
}

void MainWindow::on_button_export_clicked() {
    QString defaultPath = QString::fromStdString(outputPath_.empty() ? "sensors_config.json" : outputPath_);
    auto fileName = QFileDialog::getSaveFileName(this, "Save Config", defaultPath, "JSON Files (*.json)");
    if (fileName.isEmpty())
        return;
    saveConfig(fileName.toStdString());
}

// ─── Connect sensor to viewer ────────────────────────────────────────────────

void MainWindow::connectToViewer(int id) {
    auto& s = sensors_[id];
    int comboIdx = ui->combo_select->count();
    comboMap_[comboIdx] = id;
    sensorHidden_[comboIdx] = false;

    QColor clr = generateColor();
    palette_.push_back(clr);
    s.color = clr;
    sl::float3 clrF(clr.red(), clr.green(), clr.blue());

    if (s.type == SensorType::ZED) {
        // Only set IMU orientation as default if no pose was loaded from config
        if (!s.poseLoaded)
            s.pose = s.worker.getOrientation();
        auto param = s.worker.getCamParams();
        auto pc = s.worker.getPC();
        auto stream = s.worker.getCUDAStream();
        ui->glWidget->addCamera(param, pc, stream, id, clrF);
        ui->combo_select->addItem("ZED: " + QString::number(s.worker.serial()));
    } else {
        if (!s.poseLoaded)
            s.pose.setIdentity();
        ui->glWidget->addLidar(id, clrF);
        ui->combo_select->addItem("LiDAR: " + QString::fromStdString(s.ip.empty() ? s.worker.ip() : s.ip));
    }

    // Set this item's color in the dropdown list
    ui->combo_select->setItemData(comboIdx, QBrush(clr), Qt::ForegroundRole);

    ui->combo_select->setCurrentIndex(comboIdx);
    statusBar()->showMessage("Sensor added: " + ui->combo_select->itemText(comboIdx), 3000);
}

int MainWindow::getComboIdx(int sensorId) {
    for (auto& [idx, id] : comboMap_)
        if (id == sensorId)
            return idx;
    return -1;
}

QColor MainWindow::generateColor() {
    QColor candidate;
    double bestDiff = 0;
    for (int attempt = 0; attempt < 100; attempt++) {
        float hue = dist100_(rng_) / 100.f;
        hue += 0.618033988749895f;
        hue = fmod(hue, 1.0f);
        candidate = QColor::fromHsvF(hue, 0.65, 0.9);
        if (palette_.empty())
            return candidate;

        double minDist = 1e9;
        for (auto& c : palette_)
            minDist = std::min(minDist, colorDeltaE(c, candidate));
        if (minDist > 50)
            return candidate;
        if (minDist > bestDiff)
            bestDiff = minDist;
    }
    return candidate;
}

// ─── JSON I/O ────────────────────────────────────────────────────────────────

static sl::Transform poseFromRotTrans(const std::vector<float>& rot, const std::vector<float>& trans) {
    sl::Transform p;
    p.setIdentity();
    if (rot.size() >= 3)
        p.setRotationVector(sl::float3(rot[0], rot[1], rot[2]));
    if (trans.size() >= 3)
        p.setTranslation(sl::float3(trans[0], trans[1], trans[2]));
    return p;
}

static sl::Transform poseFromMatrix16(const std::vector<float>& m) {
    sl::Transform p;
    p.setIdentity();
    if (m.size() >= 16)
        for (int i = 0; i < 16; i++)
            p.m[i] = m[i];
    return p;
}

bool MainWindow::loadConfig(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        QMessageBox::warning(this, "Error", "Cannot open: " + QString::fromStdString(path));
        return false;
    }

    json config;
    try {
        file >> config;
    } catch (const json::exception& e) {
        QMessageBox::warning(this, "Error", "JSON parse error: " + QString::fromStdString(e.what()));
        return false;
    }

    // Close and remove all existing sensors before loading new config
    for (auto& [id, s] : sensors_) {
        s.worker.close();
        ui->glWidget->removeSensor(id);
    }
    sensors_.clear();
    comboMap_.clear();
    sensorHidden_.clear();
    palette_.clear();
    ui->combo_select->clear();
    currentId_ = 0;
    nextId_ = 0;

    configPath_ = path;
    if (outputPath_.empty()) {
        auto dot = path.rfind('.');
        outputPath_ = (dot != std::string::npos) ? path.substr(0, dot) + "_placed" + path.substr(dot) : path + "_placed.json";
    }

    // Parse ZEDs
    if (config.contains("zeds") && config["zeds"].is_array()) {
        for (const auto& z : config["zeds"]) {
            int id = nextId_++;
            sensors_[id].type = SensorType::ZED;

            int serial = 0;
            if (z.contains("serial"))
                serial = z["serial"].get<int>();
            sensors_[id].serial = serial;
            if (z.contains("fps"))
                sensors_[id].fps = z["fps"].get<int>();
            if (z.contains("resolution") && z["resolution"].is_array())
                for (const auto& v : z["resolution"])
                    sensors_[id].resolution.push_back(v.get<int>());

            // Parse pose
            bool hasPose = false;
            if (z.contains("pose") && z["pose"].is_array() && z["pose"].size() == 16) {
                std::vector<float> m;
                for (const auto& v : z["pose"])
                    m.push_back(v.get<float>());
                sensors_[id].pose = poseFromMatrix16(m);
                hasPose = true;
            } else if (z.contains("rotation") || z.contains("translation")) {
                std::vector<float> rot = {0, 0, 0}, trans = {0, 0, 0};
                if (z.contains("rotation") && z["rotation"].is_array())
                    for (size_t i = 0; i < std::min(z["rotation"].size(), (size_t)3); i++)
                        rot[i] = z["rotation"][i].get<float>();
                if (z.contains("translation") && z["translation"].is_array())
                    for (size_t i = 0; i < std::min(z["translation"].size(), (size_t)3); i++)
                        trans[i] = z["translation"][i].get<float>();
                sensors_[id].pose = poseFromRotTrans(rot, trans);
                hasPose = true;
            }
            sensors_[id].poseLoaded = hasPose;

            // Read optional SVO path
            std::string svo;
            if (z.contains("svo"))
                svo = z["svo"].get<std::string>();
            sensors_[id].svo = svo;

            bool opened = false;
            if (serial > 0) {
                opened = sensors_[id].worker.openZedSerial(serial);
                if (!opened)
                    std::cerr << "Failed to open ZED " << serial << std::endl;
            } else if (!svo.empty()) {
                opened = sensors_[id].worker.openZedSVO(svo);
                if (!opened)
                    std::cerr << "Failed to open SVO " << svo << std::endl;
            }
            if (opened)
                connectToViewer(id);
            else
                sensors_.erase(id);
        }
    }

    // Parse LiDARs
    if (config.contains("lidars") && config["lidars"].is_array()) {
        for (const auto& l : config["lidars"]) {
            int id = nextId_++;
            sensors_[id].type = SensorType::LIDAR;

            std::string ip;
            if (l.contains("ip"))
                ip = l["ip"].get<std::string>();
            sensors_[id].ip = ip;

            // Parse pose
            bool hasPose = false;
            if (l.contains("pose") && l["pose"].is_array() && l["pose"].size() == 16) {
                std::vector<float> m;
                for (const auto& v : l["pose"])
                    m.push_back(v.get<float>());
                sensors_[id].pose = poseFromMatrix16(m);
                hasPose = true;
            } else if (l.contains("rotation") || l.contains("translation")) {
                std::vector<float> rot = {0, 0, 0}, trans = {0, 0, 0};
                if (l.contains("rotation") && l["rotation"].is_array())
                    for (size_t i = 0; i < std::min(l["rotation"].size(), (size_t)3); i++)
                        rot[i] = l["rotation"][i].get<float>();
                if (l.contains("translation") && l["translation"].is_array())
                    for (size_t i = 0; i < std::min(l["translation"].size(), (size_t)3); i++)
                        trans[i] = l["translation"][i].get<float>();
                sensors_[id].pose = poseFromRotTrans(rot, trans);
                hasPose = true;
            }
            sensors_[id].poseLoaded = hasPose;

            // Read optional OSF path
            std::string osf;
            if (l.contains("osf"))
                osf = l["osf"].get<std::string>();
            sensors_[id].osf = osf;

            bool opened = false;
            if (!ip.empty()) {
                opened = sensors_[id].worker.openLidarIP(ip);
                if (!opened)
                    std::cerr << "Failed to open LiDAR " << ip << std::endl;
            } else if (!osf.empty()) {
                opened = sensors_[id].worker.openLidarOSF(osf);
                if (!opened)
                    std::cerr << "Failed to open OSF " << osf << std::endl;
            }
            if (opened)
                connectToViewer(id);
            else
                sensors_.erase(id);
        }
    }

    return !sensors_.empty();
}

void MainWindow::autoDetect() {
    auto zeds = sl::Camera::getDeviceList();
    for (auto& z : zeds) {
        int id = nextId_++;
        sensors_[id].type = SensorType::ZED;
        sensors_[id].serial = z.serial_number;
        if (sensors_[id].worker.openZedSerial(z.serial_number))
            connectToViewer(id);
        else
            sensors_.erase(id);
    }

    auto lidars = sl::Lidar::getDeviceList();
    std::unordered_set<std::string> seenIPs;
    for (auto& l : lidars) {
        std::string ip(l.ip_address.c_str());
        if (seenIPs.count(ip))
            continue;
        seenIPs.insert(ip);

        int id = nextId_++;
        sensors_[id].type = SensorType::LIDAR;
        sensors_[id].ip = ip;
        if (sensors_[id].worker.openLidarIP(ip))
            connectToViewer(id);
        else
            sensors_.erase(id);
    }
}

void MainWindow::saveConfig(const std::string& path) {
    json config;
    json zeds = json::array();
    json lidars = json::array();

    for (auto& [id, s] : sensors_) {
        sl::float3 rotVec = s.pose.getRotationVector();
        sl::float3 trans = s.pose.getTranslation();

        if (s.type == SensorType::ZED) {
            json z;
            if (s.serial > 0)
                z["serial"] = s.serial;
            if (!s.svo.empty())
                z["svo"] = s.svo;
            if (s.fps > 0)
                z["fps"] = s.fps;
            if (!s.resolution.empty())
                z["resolution"] = s.resolution;
            z["rotation"] = {rotVec.x, rotVec.y, rotVec.z};
            z["translation"] = {trans.x, trans.y, trans.z};
            zeds.push_back(z);
        } else {
            json l;
            if (!s.ip.empty())
                l["ip"] = s.ip;
            if (!s.osf.empty())
                l["osf"] = s.osf;
            // 4×4 matrix (row-major)
            json pose_arr = json::array();
            for (int i = 0; i < 16; i++)
                pose_arr.push_back(s.pose.m[i]);
            l["pose"] = pose_arr;
            lidars.push_back(l);
        }
    }

    if (!zeds.empty())
        config["zeds"] = zeds;
    if (!lidars.empty())
        config["lidars"] = lidars;

    std::ofstream out(path);
    if (!out.is_open()) {
        QMessageBox::warning(this, "Error", "Cannot write: " + QString::fromStdString(path));
        return;
    }
    out << std::setw(4) << config << std::endl;
    statusBar()->showMessage("Config saved to: " + QString::fromStdString(path), 5000);
    std::cout << "Config saved to: " << path << std::endl;
}
