# task1.py
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap, QImage
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QSlider,
    QLineEdit,
    QPushButton,
    QFileDialog,      
    QMessageBox,      
)
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class Task1Window(QWidget):
    """功能1：直方图 + 阈值分割"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = parent  # ImageViewer

        # ---- 状态变量（一定要在 _build_ui 之前定义）----
        self._orig_pixmap: QPixmap | None = None      # 原图
        self._result_pixmap: QPixmap | None = None    # 阈值后的图
        self._gray: np.ndarray | None = None          # 灰度图数组
        self._current_thresh: int = 128

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # ---- 上面：左右图像 ----
        img_layout = QHBoxLayout()

        self.label_orig = JLabelWithBg("原图")
        self.label_result = JLabelWithBg("阈值结果")

        img_layout.addWidget(self.label_orig, stretch=1)
        img_layout.addWidget(self.label_result, stretch=1)

        # ---- 中间：阈值控制 ----
        controls_layout = QHBoxLayout()

        lbl_manual = QLabel("手动阈值:")
        self.slider_thresh = QSlider(Qt.Horizontal)
        self.slider_thresh.setRange(0, 255)
        self.slider_thresh.setValue(self._current_thresh)
        self.slider_thresh.setTickInterval(10)
        self.slider_thresh.valueChanged.connect(self._on_slider_changed)

        self.edit_thresh = QLineEdit(str(self._current_thresh))
        self.edit_thresh.setFixedWidth(60)
        self.edit_thresh.setAlignment(Qt.AlignCenter)
        self.edit_thresh.editingFinished.connect(self._on_edit_finished)

        btn_apply = QPushButton("应用阈值")
        btn_apply.clicked.connect(self._apply_manual_threshold)

        btn_otsu = QPushButton("Otsu 自动阈值")
        btn_otsu.clicked.connect(self._apply_otsu)

        btn_entropy = QPushButton("Entropy 自动阈值")
        btn_entropy.clicked.connect(self._apply_entropy)

        btn_save = QPushButton("保存结果")
        btn_save.clicked.connect(self._save_result)

        controls_layout.addWidget(lbl_manual)
        controls_layout.addWidget(self.slider_thresh, stretch=1)
        controls_layout.addWidget(self.edit_thresh)
        controls_layout.addWidget(btn_apply)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(btn_otsu)
        controls_layout.addWidget(btn_entropy)
        controls_layout.addSpacing(10)         
        controls_layout.addWidget(btn_save)     
        controls_layout.addStretch()

        # ---- 下方：直方图 ----
        self.fig = Figure(figsize=(4, 2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")

        # 嵌入到 Qt 的画布
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setMinimumHeight(150)

        main_layout.addLayout(img_layout, stretch=3)
        main_layout.addLayout(controls_layout, stretch=0)
        main_layout.addWidget(self.canvas, stretch=2)

    # ---------------- 对外接口：由主窗口调用 ----------------
    def set_image(self, pm: QPixmap | None):
        """从 ImageViewer 传入当前原始图像"""
        self._orig_pixmap = pm
        self._result_pixmap = None

        if pm is None:
            self._gray = None
            self.label_orig.setText("原图：无图像")
            self.label_orig.setPixmap(QPixmap())
            self.label_result.setText("阈值结果：无图像")
            self.label_result.setPixmap(QPixmap())
            # self.label_hist.setText("直方图：无图像")
            # self.label_hist.setPixmap(QPixmap())
            return

        self.label_orig.setText("")

        # 转为灰度 numpy 数组（兼容 PySide6 / QtPy）
        qimg = pm.toImage().convertToFormat(QImage.Format_Grayscale8)
        w, h = qimg.width(), qimg.height()
        bytes_per_line = qimg.bytesPerLine()

        ptr = qimg.bits()      # memoryview
        buf = ptr.tobytes()
        arr = np.frombuffer(buf, np.uint8).reshape((h, bytes_per_line))[:, :w]
        self._gray = arr.copy()

        # 更新图像显示和直方图
        self._update_histogram()
        self._apply_threshold(self._current_thresh, sync_controls=True)
        self._refresh_previews()

    def _save_result(self):
        """保存当前阈值处理后的图像"""
        if self._result_pixmap is None:
            QMessageBox.information(self, "保存结果", "当前没有可保存的结果，请先应用阈值。")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;所有文件 (*)",
        )
        if not fname:
            return

        ok = self._result_pixmap.save(fname)
        if not ok:
            QMessageBox.warning(self, "保存失败", f"无法保存到：\n{fname}")
        else:
            QMessageBox.information(self, "保存成功", f"已保存到：\n{fname}")


    # ---------------- 直方图（灰 + 红线） ----------------
    def _update_histogram(self):
        """用 matplotlib 绘制直方图 + 当前阈值红线 + 坐标轴"""
        self.ax.clear()  # 清空上次内容

        if self._gray is None:
            # 没有图像就显示文字
            self.ax.set_facecolor("black")
            self.ax.text(
                0.5, 0.5, "直方图：无图像",
                color="white", ha="center", va="center", transform=self.ax.transAxes
            )
            self.ax.set_xticks([])
            self.ax.set_yticks([])
            self.canvas.draw()
            return

        # 计算直方图
        hist, _ = np.histogram(self._gray, bins=256, range=(0, 256))

        # 画灰色的柱状图
        x = np.arange(256)
        self.ax.bar(x, hist, width=1.0, color="0.7", edgecolor="0.7")

        # 画当前阈值的红色竖线
        t = int(np.clip(self._current_thresh, 0, 255))
        self.ax.axvline(t, color="red", linewidth=2)

        # 设置坐标轴和标题
        self.ax.set_xlim(0, 255)
        self.ax.set_xlabel("Gray level")
        self.ax.set_ylabel("Pixel count")
        self.ax.set_title("Histogram")

        # 让背景更像“数据图表”的风格
        self.ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

        self.canvas.draw()


    # ---------------- 阈值应用 ----------------
    def _apply_threshold(self, thresh: int, sync_controls: bool = False):
        if self._gray is None:
            return

        t = int(np.clip(thresh, 0, 255))
        self._current_thresh = t

        if sync_controls:
            self.slider_thresh.blockSignals(True)
            self.slider_thresh.setValue(t)
            self.slider_thresh.blockSignals(False)

            self.edit_thresh.blockSignals(True)
            self.edit_thresh.setText(str(t))
            self.edit_thresh.blockSignals(False)

        # 阈值化
        binary = (self._gray >= t).astype(np.uint8) * 255
        h, w = binary.shape
        qimg = QImage(binary.data, w, h, w, QImage.Format_Grayscale8).copy()
        pm = QPixmap.fromImage(qimg)
        self._result_pixmap = pm
        self.label_result.setText(f"阈值结果 (T={t})")

        self._update_histogram()
        self._refresh_previews()

    # ---------------- 控件回调 ----------------
    def _on_slider_changed(self, value: int):
        self._current_thresh = int(value)
        self.edit_thresh.setText(str(self._current_thresh))
        self._apply_threshold(self._current_thresh, sync_controls=False)

    def _on_edit_finished(self):
        try:
            t = int(self.edit_thresh.text())
        except ValueError:
            t = self._current_thresh
        t = max(0, min(255, t))
        self._current_thresh = t
        self.slider_thresh.setValue(t)
        self._apply_threshold(t, sync_controls=False)

    def _apply_manual_threshold(self):
        self._on_edit_finished()

    # ---------------- 自动阈值：Otsu & Entropy ----------------
    def _apply_otsu(self):
        if self._gray is None:
            return
        hist, _ = np.histogram(self._gray, bins=256, range=(0, 256))
        total = self._gray.size
        sum_total = np.dot(np.arange(256), hist)
        sumB = 0.0
        wB = 0.0
        max_var = 0.0
        threshold = 0
        for t in range(256):
            wB += hist[t]
            if wB == 0:
                continue
            wF = total - wB
            if wF == 0:
                break
            sumB += t * hist[t]
            mB = sumB / wB
            mF = (sum_total - sumB) / wF
            var_between = wB * wF * (mB - mF) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = t
        self._apply_threshold(threshold, sync_controls=True)

    def _apply_entropy(self):
        """Kapur 熵阈值法"""
        if self._gray is None:
            return
    
        # 直方图 & 概率
        hist, _ = np.histogram(self._gray, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        total = hist.sum()
        if total == 0:
            return
    
        p = hist / total  # p[i] 是灰度 i 的概率
    
        eps = 1e-12
        best_t = 0
        best_score = -1e9
    
        # t 只取到 254，避免后一类为空
        for t in range(255):
            w0 = p[: t + 1].sum()      # 类0权重
            w1 = p[t + 1 :].sum()      # 类1权重
            if w0 < eps or w1 < eps:
                continue
            
            # 条件概率分布
            p0 = p[: t + 1] / w0
            p1 = p[t + 1 :] / w1
    
            # 熵：-sum(p*log p)，只对 >0 的项求
            H0 = -(p0[p0 > 0] * np.log(p0[p0 > 0])).sum()
            H1 = -(p1[p1 > 0] * np.log(p1[p1 > 0])).sum()
    
            score = H0 + H1
            if score > best_score:
                best_score = score
                best_t = t
    
        self._apply_threshold(best_t, sync_controls=True)


    # ---------------- 预览刷新 + 自适应大小但保持比例 ----------------
    def _refresh_previews(self):
        # 左：原图
        if self._orig_pixmap is not None:
            target_size = self.label_orig.size()
            if target_size.width() > 0 and target_size.height() > 0:
                scaled = self._orig_pixmap.scaled(
                    target_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.label_orig.setPixmap(scaled)
        else:
            self.label_orig.setPixmap(QPixmap())

        # 右：阈值结果
        if self._result_pixmap is not None:
            target_size = self.label_result.size()
            if target_size.width() > 0 and target_size.height() > 0:
                scaled = self._result_pixmap.scaled(
                    target_size,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.label_result.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh_previews()


class JLabelWithBg(QLabel):
    """带统一背景样式的 QLabel，方便左右两个图像区外观一致"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background: #202020; color: white;")
        self.setScaledContents(False)
