# task2.py
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap, QImage
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QSpinBox,
    QFileDialog,      
    QMessageBox,
)
import numpy as np


class JLabelWithBg(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background: #202020; color: white;")
        self.setScaledContents(False)


class Task2Window(QWidget):
    """
    Project 2: Convolution and Image Filters
    - Roberts / Prewitt / Sobel 边缘检测
    - Gaussian / Median 滤波（可指定核大小 k×k）
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = parent

        self._orig_pixmap: QPixmap | None = None
        self._gray: np.ndarray | None = None
        self._filtered_pixmap: QPixmap | None = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        main = QVBoxLayout(self)

        # 上：左右图像
        img_layout = QHBoxLayout()
        self.label_orig = JLabelWithBg("原图（灰度）")
        self.label_result = JLabelWithBg("滤波 / 边缘结果")
        img_layout.addWidget(self.label_orig, 1)
        img_layout.addWidget(self.label_result, 1)

        # 中：滤波器 + 核大小 + 按钮
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("滤波器:"))

        self.combo_filter = QComboBox()
        self.combo_filter.addItem("Roberts 边缘", "roberts")
        self.combo_filter.addItem("Prewitt 边缘", "prewitt")
        self.combo_filter.addItem("Sobel 边缘", "sobel")
        self.combo_filter.addItem("Gaussian 滤波 (k×k)", "gaussian")
        self.combo_filter.addItem("Median 滤波 (k×k)", "median")

        ctrl_layout.addWidget(self.combo_filter)

        # 核大小选择（奇数）
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("核大小 k:"))

        self.spin_kernel = QSpinBox()
        self.spin_kernel.setRange(3, 15)     # 3~15
        self.spin_kernel.setSingleStep(2)    # 只走奇数：3,5,7,...
        self.spin_kernel.setValue(3)
        ctrl_layout.addWidget(self.spin_kernel)

        btn_apply = QPushButton("应用滤波")
        btn_apply.clicked.connect(self.apply_selected_filter)
        
        btn_save = QPushButton("保存结果")
        btn_save.clicked.connect(self._save_result)
        
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(btn_apply)
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(btn_save)
        ctrl_layout.addStretch()

        # 下：提示信息
        self.label_info = QLabel("提示：请先在主界面打开图片，再切换到 功能2。")
        self.label_info.setAlignment(Qt.AlignLeft)

        main.addLayout(img_layout, 3)
        main.addLayout(ctrl_layout, 0)
        main.addWidget(self.label_info, 0)

    # ---------------- 给 ImageViewer 用的接口 ----------------
    def set_image(self, pm: QPixmap | None):
        """从 ImageViewer 接收当前图像（转为灰度）。"""
        self._orig_pixmap = None
        self._filtered_pixmap = None
        self._gray = None

        if pm is None:
            self.label_orig.setText("原图：无图像")
            self.label_orig.setPixmap(QPixmap())
            self.label_result.setText("结果：无图像")
            self.label_result.setPixmap(QPixmap())
            self.label_info.setText("提示：请先在主界面打开图片。")
            return

        # 转灰度 QImage -> numpy
        qimg = pm.toImage().convertToFormat(QImage.Format_Grayscale8)
        w, h = qimg.width(), qimg.height()
        bpl = qimg.bytesPerLine()
        ptr = qimg.bits()
        buf = ptr.tobytes()
        arr = np.frombuffer(buf, np.uint8).reshape((h, bpl))[:, :w]
        self._gray = arr.astype(np.float64)

        # 灰度版原图（注意 bytesPerLine 用 w）
        qimg_gray = QImage(
            arr.astype(np.uint8).data,
            w,
            h,
            w,
            QImage.Format_Grayscale8,
        ).copy()
        self._orig_pixmap = QPixmap.fromImage(qimg_gray)

        self.label_info.setText("已载入灰度图像，选择滤波器和核大小后点击『应用滤波』。")
        self._filtered_pixmap = None
        self._refresh_previews()

    def _save_result(self):
        """保存当前滤波/边缘处理后的图像"""
        if self._filtered_pixmap is None:
            QMessageBox.information(self, "保存结果", "当前没有可保存的结果，请先应用滤波。")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "保存处理结果",
            "",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg *.jpeg);;所有文件 (*)",
        )
        if not fname:
            return

        ok = self._filtered_pixmap.save(fname)
        if not ok:
            QMessageBox.warning(self, "保存失败", f"无法保存到：\n{fname}")
        else:
            QMessageBox.information(self, "保存成功", f"已保存到：\n{fname}")


    # ---------------- 按钮：应用当前选择的滤波器 ----------------
    def apply_selected_filter(self):
        if self._gray is None:
            self.label_info.setText("当前无图像，请先在主界面打开图片。")
            return

        mode = self.combo_filter.currentData()
        img = self._gray

        # 核大小（只对 Gaussian / Median 有意义）
        k = self.spin_kernel.value()
        # 防御：确保是奇数
        if k % 2 == 0:
            k += 1
            self.spin_kernel.setValue(k)

        if mode == "roberts":
            result = self._edge_roberts(img)
            text = "Roberts 边缘检测（固定 2×2 核）"
        elif mode == "prewitt":
            result = self._edge_prewitt(img)
            text = "Prewitt 边缘检测（固定 3×3 核）"
        elif mode == "sobel":
            result = self._edge_sobel(img)
            text = "Sobel 边缘检测（固定 3×3 核）"
        elif mode == "gaussian":
            result = self._gaussian_blur(img, ksize=k)
            text = f"Gaussian 滤波 ({k}×{k})"
        elif mode == "median":
            result = self._median_filter(img, ksize=k)
            text = f"Median 滤波 ({k}×{k})"
        else:
            return

        # 归一化到 0~255
        result = np.clip(result, 0, None)
        maxv = result.max()
        if maxv > 0:
            result = result / maxv * 255.0
        result_u8 = result.astype(np.uint8)

        h, w = result_u8.shape
        qimg = QImage(result_u8.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._filtered_pixmap = QPixmap.fromImage(qimg)
        self.label_result.setText(text)
        self.label_info.setText(f"{text} 完成。")
        self._refresh_previews()

    # ---------------- 卷积和滤波实现 ----------------
    def _conv2d(self, img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """简单 2D 卷积，边界用 edge 填充。"""
        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
        out = np.zeros_like(img, dtype=np.float64)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+kh, j:j+kw]
                out[i, j] = np.sum(region * kernel)
        return out

    # --- 边缘检测：固定核 ---
    def _edge_roberts(self, img):
        kx = np.array([[1, 0],
                       [0, -1]], float)
        ky = np.array([[0, 1],
                       [-1, 0]], float)
        gx = self._conv2d(img, kx)
        gy = self._conv2d(img, ky)
        return np.sqrt(gx**2 + gy**2)

    def _edge_prewitt(self, img):
        kx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], float)
        ky = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]], float)
        gx = self._conv2d(img, kx)
        gy = self._conv2d(img, ky)
        return np.sqrt(gx**2 + gy**2)

    def _edge_sobel(self, img):
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], float)
        ky = np.array([[1, 2, 1],
                       [0, 0, 0],
                       [-1, -2, -1]], float)
        gx = self._conv2d(img, kx)
        gy = self._conv2d(img, ky)
        return np.sqrt(gx**2 + gy**2)

    # --- Gaussian / Median 支持任意奇数核大小 ---
    def _gaussian_blur(self, img, ksize: int = 3):
        """ksize×ksize 高斯滤波，ksize 为奇数。"""
        half = ksize // 2
        # 简单高斯核：sigma 取 ksize/3
        sigma = ksize / 3.0
        ax = np.arange(-half, half + 1)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel /= kernel.sum()
        return self._conv2d(img, kernel)

    def _median_filter(self, img, ksize: int = 3):
        """ksize×ksize 中值滤波，ksize 为奇数。"""
        kh = kw = ksize
        ph = pw = ksize // 2
        padded = np.pad(img, ((ph, ph), (pw, pw)), mode="edge")
        out = np.zeros_like(img, float)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = padded[i:i+kh, j:j+kw]
                out[i, j] = np.median(region)
        return out

    # ---------------- 预览刷新 & 自适应大小 ----------------
    def _refresh_previews(self):
        if self._orig_pixmap is not None:
            target = self.label_orig.size()
            if target.width() > 0 and target.height() > 0:
                scaled = self._orig_pixmap.scaled(
                    target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.label_orig.setPixmap(scaled)
                self.label_orig.setText("")
        else:
            self.label_orig.setPixmap(QPixmap())

        if self._filtered_pixmap is not None:
            target = self.label_result.size()
            if target.width() > 0 and target.height() > 0:
                scaled = self._filtered_pixmap.scaled(
                    target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.label_result.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh_previews()
