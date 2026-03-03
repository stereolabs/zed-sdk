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
#include <QStyledItemDelegate>
#include <QPainter>

// Delegate that paints each combo item using its Qt::ForegroundRole color,
// even when a stylesheet sets a global color on the QComboBox.
class ColorItemDelegate : public QStyledItemDelegate {
public:
    using QStyledItemDelegate::QStyledItemDelegate;
    void paint(QPainter* painter, const QStyleOptionViewItem& option, const QModelIndex& index) const override {
        QStyleOptionViewItem opt = option;
        initStyleOption(&opt, index);
        QVariant fg = index.data(Qt::ForegroundRole);
        if (fg.isValid())
            opt.palette.setColor(QPalette::Text, fg.value<QBrush>().color());
        QStyledItemDelegate::paint(painter, opt, index);
    }
};

#include <algorithm>

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

    // ── Style spin-box arrows  ──────────────────────
    const QString spinStyle = QStringLiteral("QDoubleSpinBox {"
                                             "  color: rgb(242,242,242);"
                                             "  background-color: rgb(25,25,25);"
                                             "  border: 1px solid rgb(194,194,194);"
                                             "  border-radius: 4px;"
                                             "  padding-right: 24px;"
                                             "  min-height: 24px;"
                                             "}"
                                             "QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {"
                                             "  width: 22px;"
                                             "  border: 0px;"
                                             "  background: transparent;"
                                             "}"
                                             "QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {"
                                             "  background: rgb(60,60,60);"
                                             "}"
                                             "QDoubleSpinBox::up-arrow {"
                                             "  image: none;"
                                             "  border-left: 5px solid transparent;"
                                             "  border-right: 5px solid transparent;"
                                             "  border-bottom: 6px solid rgb(194,194,194);"
                                             "}"
                                             "QDoubleSpinBox::up-arrow:hover {"
                                             "  border-bottom: 6px solid rgb(242,242,242);"
                                             "}"
                                             "QDoubleSpinBox::down-arrow {"
                                             "  image: none;"
                                             "  border-left: 5px solid transparent;"
                                             "  border-right: 5px solid transparent;"
                                             "  border-top: 6px solid rgb(194,194,194);"
                                             "}"
                                             "QDoubleSpinBox::down-arrow:hover {"
                                             "  border-top: 6px solid rgb(242,242,242);"
                                             "}");
    for (auto* sb : {ui->spin_rx, ui->spin_ry, ui->spin_rz, ui->spin_tx, ui->spin_ty, ui->spin_tz})
        sb->setStyleSheet(spinStyle);

    // Use a custom delegate so dropdown items keep their own ForegroundRole colors
    ui->combo_select->setItemDelegate(new ColorItemDelegate(ui->combo_select));

    // Connect point-pick signal from GlViewer
    connect(ui->glWidget, &GlViewer::pointPicked, this, &MainWindow::onPointPicked);

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
}

// ─── Rotation dials ──────────────────────────────────────────────────────────

void MainWindow::on_dial_x_valueChanged(int value) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    pushUndo();
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
    pushUndo();
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
    pushUndo();
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
    pushUndo();
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
    pushUndo();
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
    pushUndo();
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
    pushUndo();
    sensors_[currentId_].pose.tx = v * TRANS_SCALE;
    ui->spin_tx->blockSignals(true);
    ui->spin_tx->setValue(v * TRANS_SCALE);
    ui->spin_tx->blockSignals(false);
}

void MainWindow::on_scroll_ty_valueChanged(int v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    pushUndo();
    sensors_[currentId_].pose.ty = v * TRANS_SCALE;
    ui->spin_ty->blockSignals(true);
    ui->spin_ty->setValue(v * TRANS_SCALE);
    ui->spin_ty->blockSignals(false);
}

void MainWindow::on_scroll_tz_valueChanged(int v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    pushUndo();
    sensors_[currentId_].pose.tz = v * TRANS_SCALE;
    ui->spin_tz->blockSignals(true);
    ui->spin_tz->setValue(v * TRANS_SCALE);
    ui->spin_tz->blockSignals(false);
}

// ─── Translation spin boxes ─────────────────────────────────────────────────

void MainWindow::on_spin_tx_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    pushUndo();
    sensors_[currentId_].pose.tx = v;
    ui->scroll_tx->blockSignals(true);
    ui->scroll_tx->setValue((int)(v / TRANS_SCALE));
    ui->scroll_tx->blockSignals(false);
}

void MainWindow::on_spin_ty_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    pushUndo();
    sensors_[currentId_].pose.ty = v;
    ui->scroll_ty->blockSignals(true);
    ui->scroll_ty->setValue((int)(v / TRANS_SCALE));
    ui->scroll_ty->blockSignals(false);
}

void MainWindow::on_spin_tz_valueChanged(double v) {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    pushUndo();
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
    ui->button_imuPose->setEnabled(isZed);
    ui->check_edges->setEnabled(isZed);

    // Set the combo box field text to the selected sensor's color.
    // Dropdown items use their own ForegroundRole via ColorItemDelegate.
    QColor c = s.color;
    ui->combo_select->setStyleSheet(QString("QComboBox#combo_select { color: rgb(%1,%2,%3); }").arg(c.red()).arg(c.green()).arg(c.blue()));
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
    if (sensors_[id].worker.openZedSerial(serial)) {
        ui->edit_serial->clear();
        connectToViewer(id);
    } else {
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
    if (sensors_[id].worker.openLidarIP(ip)) {
        ui->edit_ip->clear();
        connectToViewer(id);
    } else {
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
        if (sensors_[id].worker.openLidarOSF(path)) {
            ui->edit_svo->clear();
            connectToViewer(id);
        } else {
            sensors_.erase(id);
        }
    } else {
        sensors_[id].type = SensorType::ZED;
        if (sensors_[id].worker.openZedSVO(path)) {
            ui->edit_svo->clear();
            connectToViewer(id);
        } else {
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

void MainWindow::on_slider_density_valueChanged(int value) {
    ui->glWidget->setDensity(value);
    int pct = 100 / value;
    ui->label_densityval->setText(QString("%1%").arg(pct));
}

// ─── Actions ─────────────────────────────────────────────────────────────────

// ─── Undo ─────────────────────────────────────────────────────────────────────

void MainWindow::pushUndo() {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    UndoEntry entry {currentId_, sensors_[currentId_].pose};
    // Avoid duplicates: skip if top of stack is same sensor + same pose
    if (!undoStack_.empty()) {
        auto& top = undoStack_.back();
        if (top.sensorId == entry.sensorId) {
            bool same = true;
            for (int i = 0; i < 16 && same; i++)
                same = (top.pose.m[i] == entry.pose.m[i]);
            if (same)
                return;
        }
    }
    undoStack_.push_back(entry);
    if ((int)undoStack_.size() > kMaxUndo)
        undoStack_.pop_front();
    ui->button_undo->setEnabled(true);
}

void MainWindow::on_button_undo_clicked() {
    if (undoStack_.empty())
        return;
    auto entry = undoStack_.back();
    undoStack_.pop_back();
    ui->button_undo->setEnabled(!undoStack_.empty());

    if (sensors_.find(entry.sensorId) == sensors_.end())
        return;
    sensors_[entry.sensorId].pose = entry.pose;

    // Switch to that sensor and refresh the UI controls
    int comboIdx = getComboIdx(entry.sensorId);
    if (comboIdx >= 0) {
        ui->combo_select->setCurrentIndex(comboIdx);
        on_combo_select_currentIndexChanged(comboIdx);
    }
    statusBar()->showMessage("Undo", 1500);
}

void MainWindow::on_button_reset_clicked() {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    pushUndo();
    sensors_[currentId_].pose = sensors_[currentId_].initialPose;

    int comboIdx = getComboIdx(currentId_);
    if (comboIdx >= 0) {
        ui->combo_select->setCurrentIndex(comboIdx);
        on_combo_select_currentIndexChanged(comboIdx);
    }
    statusBar()->showMessage("Pose reset to initial state", 2000);
}

// ─── Actions ─────────────────────────────────────────────────────────────────

void MainWindow::on_button_findPlane_clicked() {
    if (sensors_.find(currentId_) == sensors_.end())
        return;
    auto& s = sensors_[currentId_];
    if (s.type != SensorType::ZED)
        return;
    pushUndo();

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

    pushUndo();
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

    // Remember initial pose for Reset
    s.initialPose = s.pose;

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

// ════════════════════════════════════════════════════════════════════════
// ICP Point Matching
// ════════════════════════════════════════════════════════════════════════

void MainWindow::onPointPicked(int sensorId, float x, float y, float z) {
    sl::float3 pos(x, y, z);
    allPicks_.push_back({pos, sensorId});

    // Color by sensor: first sensor seen = red, second = green
    // Determine which is the "first" sensor we picked on
    int firstSensor = allPicks_[0].sensorId;
    sl::float3 color = (sensorId == firstSensor) ? sl::float3(1.0f, 0.0f, 0.0f)  // red
                                                 : sl::float3(0.0f, 1.0f, 0.0f); // green

    ui->glWidget->addPickMarker(pos, color, (int)allPicks_.size());

    std::cout << "Pick #" << allPicks_.size() << ": sensor=" << sensorId << " pos=(" << x << ", " << y << ", " << z << ")" << std::endl;
    statusBar()->showMessage(
        QString("Pick #%1 on sensor %2: (%3, %4, %5)")
            .arg(allPicks_.size())
            .arg(sensorId)
            .arg(x, 0, 'f', 3)
            .arg(y, 0, 'f', 3)
            .arg(z, 0, 'f', 3),
        3000
    );

    updatePickLabel();
}

void MainWindow::on_button_clearPicks_clicked() {
    allPicks_.clear();
    ui->glWidget->clearPickMarkers();
    updatePickLabel();
    statusBar()->showMessage("Picks cleared", 2000);
}

void MainWindow::updatePickLabel() {
    // Count picks per sensor
    std::map<int, int> perSensor;
    for (auto& p : allPicks_)
        perSensor[p.sensorId]++;

    int nSensors = (int)perSensor.size();
    int minPicks = 0;
    if (nSensors == 2) {
        auto it = perSensor.begin();
        int a = it->second;
        ++it;
        int b = it->second;
        minPicks = std::min(a, b);
    }

    QString txt = QString("Picks: %1 total").arg(allPicks_.size());
    if (nSensors == 2)
        txt += QString(" (%1 pairs — 1st↔1st, 2nd↔2nd …)").arg(minPicks);
    else if (nSensors < 2)
        txt += " — pick matching points on 2 sensors in same order";
    else
        txt += " — too many sensors (need exactly 2)";

    bool ready = (nSensors == 2 && minPicks >= 3);
    if (ready)
        txt += "  \u2714";
    ui->label_picks->setText(txt);
    ui->button_icpAlign->setEnabled(ready);
}

namespace {

    struct Mat3 {
        float m[3][3];
        Mat3() {
            memset(m, 0, sizeof(m));
        }
        static Mat3 identity() {
            Mat3 I;
            I.m[0][0] = I.m[1][1] = I.m[2][2] = 1;
            return I;
        }
    };

    // Compute yaw-only (Y-axis rotation) + translation from point pairs.
    // Solves for the optimal angle θ around Y and translation [tx, ty, tz]
    // that minimises  Σ ‖tgt_i − (Ry(θ) · src_i + t)‖².
    //
    // The yaw rotation Ry(θ) only mixes X and Z:
    //   x' =  cos θ · x + sin θ · z
    //   z' = -sin θ · x + cos θ · z
    //   y' = y
    //
    // We solve for (cos θ, sin θ) from the centered XZ cross-covariance,
    // then recover t from the centroids.
    void yawTranslationFromPairs(const std::vector<sl::float3>& src, const std::vector<sl::float3>& tgt, Mat3& R_out, sl::float3& t_out) {
        int n = (int)src.size();

        // Centroids
        sl::float3 sc(0, 0, 0), tc(0, 0, 0);
        for (int i = 0; i < n; i++) {
            sc.x += src[i].x;
            sc.y += src[i].y;
            sc.z += src[i].z;
            tc.x += tgt[i].x;
            tc.y += tgt[i].y;
            tc.z += tgt[i].z;
        }
        sc.x /= n;
        sc.y /= n;
        sc.z /= n;
        tc.x /= n;
        tc.y /= n;
        tc.z /= n;

        // Accumulate XZ cross-covariance terms
        // We want to maximise  Σ [ (tx_i)(cos θ · sx_i + sin θ · sz_i) +
        //                          (tz_i)(-sin θ · sx_i + cos θ · sz_i) ]
        // = cos θ · Σ(tx·sx + tz·sz) + sin θ · Σ(tx·sz - tz·sx)
        // = cos θ · A + sin θ · B
        // Maximised when  θ = atan2(B, A)
        float A = 0, B = 0;
        for (int i = 0; i < n; i++) {
            float sx = src[i].x - sc.x, sz = src[i].z - sc.z;
            float tx = tgt[i].x - tc.x, tz = tgt[i].z - tc.z;
            A += tx * sx + tz * sz; // cos coefficient
            B += tx * sz - tz * sx; // sin coefficient
        }

        float theta = atan2f(B, A);
        float cosT = cosf(theta), sinT = sinf(theta);

        // Ry(θ)
        R_out = Mat3::identity();
        R_out.m[0][0] = cosT;
        R_out.m[0][2] = sinT;
        R_out.m[2][0] = -sinT;
        R_out.m[2][2] = cosT;

        // t = tc - Ry · sc
        t_out.x = tc.x - (cosT * sc.x + sinT * sc.z);
        t_out.y = tc.y - sc.y;
        t_out.z = tc.z - (-sinT * sc.x + cosT * sc.z);
    }

} // anonymous namespace

void MainWindow::on_button_icpAlign_clicked() {
    // Split picks by sensor, preserving order of appearance
    std::map<int, std::vector<sl::float3>> bySensor;
    int firstSensorId = -1;
    for (auto& p : allPicks_) {
        if (firstSensorId < 0)
            firstSensorId = p.sensorId;
        bySensor[p.sensorId].push_back(p.pos);
    }

    if (bySensor.size() != 2) {
        statusBar()->showMessage("Need picks on exactly 2 sensors", 3000);
        return;
    }

    auto it = bySensor.begin();
    int idA = it->first;
    auto& ptsA = it->second;
    ++it;
    int idB = it->first;
    auto& ptsB = it->second;

    // Source = first sensor picked on (the one to move), target = the other
    int srcSensorId, tgtSensorId;
    std::vector<sl::float3>*srcPool, *tgtPool;
    if (idA == firstSensorId) {
        srcSensorId = idA;
        tgtSensorId = idB;
        srcPool = &ptsA;
        tgtPool = &ptsB;
    } else {
        srcSensorId = idB;
        tgtSensorId = idA;
        srcPool = &ptsB;
        tgtPool = &ptsA;
    }

    // Pair by order: 1st src ↔ 1st tgt, 2nd ↔ 2nd, etc.
    int nPairs = (int)std::min(srcPool->size(), tgtPool->size());
    if (nPairs < 3) {
        statusBar()->showMessage(QString("Need ≥ 3 pairs, have %1 src and %2 tgt picks").arg(srcPool->size()).arg(tgtPool->size()), 3000);
        return;
    }

    std::vector<sl::float3> srcPts, tgtPts;
    std::cout << "=== Align sensor " << srcSensorId << " -> sensor " << tgtSensorId << " (" << nPairs << " pairs) ===" << std::endl;
    for (int i = 0; i < nPairs; i++) {
        srcPts.push_back((*srcPool)[i]);
        tgtPts.push_back((*tgtPool)[i]);
        float d = sl::float3::distance(srcPts[i], tgtPts[i]);
        std::cout << "  pair " << i << ": src=(" << srcPts[i].x << ", " << srcPts[i].y << ", " << srcPts[i].z << ")  tgt=(" << tgtPts[i].x
                  << ", " << tgtPts[i].y << ", " << tgtPts[i].z << ")  dist=" << d << std::endl;
    }

    pushUndo();

    // Yaw (Y-rotation) + translation
    Mat3 R;
    sl::float3 t;
    yawTranslationFromPairs(srcPts, tgtPts, R, t);

    // Cap each translation component to ±0.5 m
    t.x = std::max(-0.5f, std::min(0.5f, t.x));
    t.y = std::max(-0.5f, std::min(0.5f, t.y));
    t.z = std::max(-0.5f, std::min(0.5f, t.z));

    float tLen = sqrtf(t.x * t.x + t.y * t.y + t.z * t.z);
    std::cout << "  t=(" << t.x << ", " << t.y << ", " << t.z << ") len=" << tLen << std::endl;

    // Build sl::Transform from R and t
    sl::Transform T;
    T.setIdentity();
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            T(i, j) = R.m[i][j];
    T(0, 3) = t.x;
    T(1, 3) = t.y;
    T(2, 3) = t.z;

    // Apply: new_pose = T * old_pose
    auto& srcSensor = sensors_[srcSensorId];
    sl::Transform oldPose = srcSensor.pose;
    sl::Transform newPose = T * oldPose;
    srcSensor.pose = newPose;

    // Refresh the UI controls
    int comboIdx = getComboIdx(srcSensorId);
    if (comboIdx >= 0) {
        ui->combo_select->setCurrentIndex(comboIdx);
        on_combo_select_currentIndexChanged(comboIdx);
    }

    // Compute residual error
    float totalErr = 0;
    for (int i = 0; i < nPairs; i++) {
        sl::float3& s = srcPts[i];
        sl::float3 tr;
        tr.x = R.m[0][0] * s.x + R.m[0][1] * s.y + R.m[0][2] * s.z + t.x;
        tr.y = R.m[1][0] * s.x + R.m[1][1] * s.y + R.m[1][2] * s.z + t.y;
        tr.z = R.m[2][0] * s.x + R.m[2][1] * s.y + R.m[2][2] * s.z + t.z;
        float dx = tr.x - tgtPts[i].x;
        float dy = tr.y - tgtPts[i].y;
        float dz = tr.z - tgtPts[i].z;
        totalErr += sqrtf(dx * dx + dy * dy + dz * dz);
    }
    float meanErr = totalErr / nPairs;
    float yawDeg = rad2deg(atan2f(R.m[0][2], R.m[0][0]));
    auto euler = newPose.getEulerAngles();
    auto trans = newPose.getTranslation();

    QString result = QString("Moved sensor %1 -> %2\n"
                             "%3 closest-point pairs\n"
                             "Mean residual: %4 m\n"
                             "Applied yaw: %5 deg  t=[%6, %7, %8] m\n"
                             "Final: t=[%9, %10, %11] r=[%12, %13, %14] deg")
                         .arg(srcSensorId)
                         .arg(tgtSensorId)
                         .arg(nPairs)
                         .arg(meanErr, 0, 'f', 4)
                         .arg(yawDeg, 0, 'f', 2)
                         .arg(t.x, 0, 'f', 4)
                         .arg(t.y, 0, 'f', 4)
                         .arg(t.z, 0, 'f', 4)
                         .arg(trans.x, 0, 'f', 4)
                         .arg(trans.y, 0, 'f', 4)
                         .arg(trans.z, 0, 'f', 4)
                         .arg(rad2deg(euler.x), 0, 'f', 2)
                         .arg(rad2deg(euler.y), 0, 'f', 2)
                         .arg(rad2deg(euler.z), 0, 'f', 2);
    statusBar()->showMessage(QString("Aligned %1 pairs — residual: %2 m").arg(nPairs).arg(meanErr, 0, 'f', 4), 5000);
    std::cout << result.toStdString() << std::endl;

    // Clear pick points and markers after alignment
    allPicks_.clear();
    ui->glWidget->clearPickMarkers();
    updatePickLabel();
}
