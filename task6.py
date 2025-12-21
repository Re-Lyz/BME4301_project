# task6.py
"""
实现以下功能：
- Morphological edge detection（形态学边缘检测，基于二值图）
- Morphological reconstruction：
    * Conditional dilation in binary image（二值条件膨胀重建）
    * Gray scale reconstruction（灰度形态学重建）
- Morphological gradient（形态学梯度）
"""

import cv2 as cv
import numpy as np

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

from task3 import JLabelWithBg


class Task6Window(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = parent

        # 状态
        self._orig_pixmap: QPixmap | None = None   # 原始图（灰度显示）
        self._gray: np.ndarray | None = None       # 灰度图 (H, W) uint8
        self._binary: np.ndarray | None = None     # 二值图 (H, W) uint8, 0/255
        self._result: np.ndarray | None = None     # 当前结果 (H, W) uint8
        self._result_pixmap: QPixmap | None = None # 显示结果

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        main = QVBoxLayout(self)

        # 顶部：三张图 —— 灰度原图 / 二值图 / 结果
        top = QHBoxLayout()
        self.label_gray = JLabelWithBg("灰度原图")
        self.label_bin = JLabelWithBg("二值图（用于二值运算）")
        self.label_result = JLabelWithBg("结果图")
        top.addWidget(self.label_gray, 1)
        top.addWidget(self.label_bin, 1)
        top.addWidget(self.label_result, 1)

        # 中部：控制区
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("操作:"))

        self.combo_op = QComboBox()
        self.combo_op.addItem("形态学边缘检测（Morphological edge, binary）", "edge")
        self.combo_op.addItem("二值重建（Binary reconstruction, conditional dilation）", "bin_recon")
        self.combo_op.addItem("灰度重建（Gray-scale reconstruction）", "gray_recon")
        self.combo_op.addItem("形态学梯度（Morphological gradient, gray）", "grad")
        ctrl.addWidget(self.combo_op)

        # 结构元素大小
        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("核大小:"))
        self.spin_ksize = QSpinBox()
        self.spin_ksize.setRange(3, 31)
        self.spin_ksize.setSingleStep(2)  # 3,5,7,...
        self.spin_ksize.setValue(3)
        ctrl.addWidget(self.spin_ksize)

        # 最大迭代次数（用于重建）
        ctrl.addSpacing(20)
        ctrl.addWidget(QLabel("最大迭代次数:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 200)
        self.spin_iter.setValue(50)
        ctrl.addWidget(self.spin_iter)

        # 按钮
        ctrl.addSpacing(20)
        btn_apply = QPushButton("应用")
        btn_apply.clicked.connect(self._apply_operation)
        ctrl.addWidget(btn_apply)

        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self._reset_result)
        ctrl.addWidget(btn_reset)

        btn_save = QPushButton("保存结果")
        btn_save.clicked.connect(self._save_result)
        ctrl.addWidget(btn_save)

        ctrl.addStretch()

        # 底部：提示
        self.label_info = QLabel("提示：请在主窗口打开图片后切换到 功能6。")
        self.label_info.setAlignment(Qt.AlignLeft)

        main.addLayout(top, 3)
        main.addLayout(ctrl, 0)
        main.addWidget(self.label_info, 0)

    # ---------------- 对外接口：ImageViewer 调用 ----------------
    def set_image(self, pm: QPixmap | None):
        """
        从主窗口获取当前图像：
        - 转成灰度；自动做 Otsu 二值化
        - 更新三张预览图
        """
        self._orig_pixmap = None
        self._gray = None
        self._binary = None
        self._result = None
        self._result_pixmap = None

        if pm is None:
            self.label_gray.setPixmap(QPixmap())
            self.label_bin.setPixmap(QPixmap())
            self.label_result.setPixmap(QPixmap())
            self.label_gray.setText("灰度原图：无图像")
            self.label_bin.setText("二值图：无图像")
            self.label_result.setText("结果图：无图像")
            self.label_info.setText("提示：请先在主窗口打开图片。")
            return

        # 1. QPixmap -> 灰度 QImage
        qimg = pm.toImage().convertToFormat(QImage.Format_Grayscale8)
        w, h = qimg.width(), qimg.height()
        bpl = qimg.bytesPerLine()
        buf = qimg.bits().tobytes()

        # 2. 灰度 numpy 数组（保证 C-contiguous）
        arr = (
            np.frombuffer(buf, dtype=np.uint8)
            .reshape((h, bpl))[:, :w]
            .copy()
        )
        self._gray = arr

        # 3. Otsu 二值化，0/255
        _, binary = cv.threshold(arr, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        self._binary = binary

        # 4. 灰度 / 二值 QPixmap
        qimg_gray = QImage(self._gray.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._orig_pixmap = QPixmap.fromImage(qimg_gray)

        qimg_bin = QImage(self._binary.data, w, h, w, QImage.Format_Grayscale8).copy()
        bin_pixmap = QPixmap.fromImage(qimg_bin)

        # 初始结果为空（或直接显示灰度）
        self._result = None
        self._result_pixmap = None

        # 刷新预览
        self._refresh_previews(gray_pm=self._orig_pixmap, bin_pm=bin_pixmap, result_pm=None)
        self.label_info.setText("已载入图像：灰度与二值图已生成，选择操作后点击『应用』。")

    # ---------------- 形态学核心操作 ----------------
    def _get_kernel(self):
        k = self.spin_ksize.value()
        if k % 2 == 0:
            k += 1
            self.spin_ksize.setValue(k)
        return cv.getStructuringElement(cv.MORPH_RECT, (k, k)), k

    def _apply_operation(self):
        if self._gray is None:
            self.label_info.setText("当前没有图像，请先在主窗口打开图片。")
            return

        op = self.combo_op.currentData()
        kernel, k = self._get_kernel()
        max_iter = self.spin_iter.value()

        if op == "edge":
            # Morphological edge detection on binary image:
            # Edge = A - (A ⊖ B)
            if self._binary is None:
                self.label_info.setText("二值图不存在，无法进行二值操作。")
                return
            eroded = cv.erode(self._binary, kernel, iterations=1)
            out = cv.subtract(self._binary, eroded)

        elif op == "grad":
            # Morphological gradient on gray image:
            # Grad = (A ⊕ B) - (A ⊖ B)
            dil = cv.dilate(self._gray, kernel, iterations=1)
            ero = cv.erode(self._gray, kernel, iterations=1)
            out = cv.subtract(dil, ero)

        elif op == "bin_recon":
            # Binary morphological reconstruction by dilation:
            # iterate: f_{k} = (f_{k-1} ⊕ B) ∧ g, until stable
            if self._binary is None:
                self.label_info.setText("二值图不存在，无法进行二值重建。")
                return
            mask = self._binary
            # marker 取 mask 的一次腐蚀（保证是其子集）
            marker = cv.erode(mask, kernel, iterations=1)
            out = self._binary_reconstruction(marker, mask, kernel, max_iter)

        elif op == "gray_recon":
            # Gray-scale morphological reconstruction by dilation
            mask = self._gray
            marker = cv.erode(mask, kernel, iterations=1)
            out = self._gray_reconstruction(marker, mask, kernel, max_iter)

        else:
            out = self._gray

        self._result = np.clip(out, 0, 255).astype(np.uint8)
        self._update_result_pixmap()
        self._refresh_previews()
        text_map = {
            "edge": "Morphological edge (binary)",
            "grad": "Morphological gradient (gray)",
            "bin_recon": "Binary reconstruction (conditional dilation)",
            "gray_recon": "Gray-scale reconstruction",
        }
        self.label_info.setText(
            f"{text_map.get(op, '操作')} 已完成。核大小 {k}×{k}，最大迭代 {max_iter}。"
        )

    def _binary_reconstruction(self, marker, mask, kernel, max_iter):
        """Binary morphological reconstruction by dilation."""
        prev = marker.copy()
        for i in range(max_iter):
            dil = cv.dilate(prev, kernel, iterations=1)
            rec = cv.min(dil, mask)
            if np.array_equal(rec, prev):
                break
            prev = rec
        return prev

    def _gray_reconstruction(self, marker, mask, kernel, max_iter):
        """Gray-scale morphological reconstruction by dilation."""
        prev = marker.copy()
        for i in range(max_iter):
            dil = cv.dilate(prev, kernel, iterations=1)
            rec = cv.min(dil, mask)
            if np.array_equal(rec, prev):
                break
            prev = rec
        return prev

    # ---------------- 状态 / 显示 ----------------
    def _update_result_pixmap(self):
        if self._result is None:
            self._result_pixmap = None
            self.label_result.setPixmap(QPixmap())
            return
        img = self._result
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._result_pixmap = QPixmap.fromImage(qimg)

    def _refresh_previews(self, gray_pm=None, bin_pm=None, result_pm=None):
        """根据当前窗口大小重新缩放各个预览图。"""
        if gray_pm is None:
            gray_pm = self._orig_pixmap
        if bin_pm is None and self._binary is not None:
            h, w = self._binary.shape
            qimg_bin = QImage(self._binary.data, w, h, w, QImage.Format_Grayscale8).copy()
            bin_pm = QPixmap.fromImage(qimg_bin)
        if result_pm is None:
            result_pm = self._result_pixmap

        # 灰度
        if gray_pm is not None:
            target = self.label_gray.size()
            if target.width() > 0 and target.height() > 0:
                scaled = gray_pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_gray.setPixmap(scaled)
                self.label_gray.setText("")
        else:
            self.label_gray.setPixmap(QPixmap())

        # 二值
        if bin_pm is not None:
            target = self.label_bin.size()
            if target.width() > 0 and target.height() > 0:
                scaled = bin_pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_bin.setPixmap(scaled)
                self.label_bin.setText("")
        else:
            self.label_bin.setPixmap(QPixmap())

        # 结果
        if result_pm is not None:
            target = self.label_result.size()
            if target.width() > 0 and target.height() > 0:
                scaled = result_pm.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.label_result.setPixmap(scaled)
                self.label_result.setText("")
        else:
            # 如果没有结果，就先显示一份灰度图当占位
            if self._orig_pixmap is not None:
                target = self.label_result.size()
                if target.width() > 0 and target.height() > 0:
                    scaled = self._orig_pixmap.scaled(
                        target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                    self.label_result.setPixmap(scaled)
                    self.label_result.setText("")
            else:
                self.label_result.setPixmap(QPixmap())

    def _reset_result(self):
        """不改灰度 / 二值，只清空结果，方便重新做实验。"""
        self._result = None
        self._result_pixmap = None
        self._refresh_previews()
        self.label_info.setText("结果已清空，可以重新选择操作并应用。")

    def _save_result(self):
        if self._result_pixmap is None:
            QMessageBox.information(self, "保存结果", "当前没有可保存的结果，请先进行形态学操作。")
            return

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "保存任务6结果图像",
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

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh_previews()
