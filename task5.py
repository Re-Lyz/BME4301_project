# task5.py
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


class Task5Window(QWidget):
    """
    Project-5: 灰度形态学
    - grayscale dilation / erosion / opening / closing
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = parent

        # 状态
        self._orig_pixmap: QPixmap | None = None   # 原图（灰度）
        self._gray_orig: np.ndarray | None = None  # 原始灰度图 (H, W) uint8
        self._gray_curr: np.ndarray | None = None  # 当前灰度图 (H, W) uint8
        self._result_pixmap: QPixmap | None = None # 当前结果图

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        main = QVBoxLayout(self)

        # 上：左右两张图
        img_layout = QHBoxLayout()
        self.label_orig = JLabelWithBg("原图（灰度）")
        self.label_result = JLabelWithBg("形态学结果（灰度）")
        img_layout.addWidget(self.label_orig, 1)
        img_layout.addWidget(self.label_result, 1)

        # 中间：控制区
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("操作:"))

        self.combo_op = QComboBox()
        self.combo_op.addItem("膨胀 (dilation)", "dilate")
        self.combo_op.addItem("腐蚀 (erosion)", "erode")
        self.combo_op.addItem("开运算 (opening)", "open")
        self.combo_op.addItem("闭运算 (closing)", "close")
        ctrl_layout.addWidget(self.combo_op)

        # 迭代次数
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("迭代次数:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 20)
        self.spin_iter.setValue(1)
        ctrl_layout.addWidget(self.spin_iter)

        # 核大小（只允许奇数）
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("核大小:"))
        self.spin_ksize = QSpinBox()
        self.spin_ksize.setRange(3, 31)
        self.spin_ksize.setSingleStep(2)  # 3,5,7,...
        self.spin_ksize.setValue(3)
        ctrl_layout.addWidget(self.spin_ksize)

        # 按钮
        btn_apply = QPushButton("应用")
        btn_apply.clicked.connect(self._apply_operation)

        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self._reset_image)

        btn_save = QPushButton("保存结果")
        btn_save.clicked.connect(self._save_result)

        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(btn_apply)
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(btn_reset)
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(btn_save)
        ctrl_layout.addStretch()

        # 下：提示信息
        self.label_info = QLabel("提示：请先在主界面打开图片，然后切换到 功能5。")
        self.label_info.setAlignment(Qt.AlignLeft)

        main.addLayout(img_layout, 3)
        main.addLayout(ctrl_layout, 0)
        main.addWidget(self.label_info, 0)

    # ---------------- 对外接口：由 ImageViewer 调用 ----------------
    def set_image(self, pm: QPixmap | None):
        self._orig_pixmap = None
        self._gray_orig = None
        self._gray_curr = None
        self._result_pixmap = None

        if pm is None:
            self.label_orig.setText("原图：无图像")
            self.label_orig.setPixmap(QPixmap())
            self.label_result.setText("结果：无图像")
            self.label_result.setPixmap(QPixmap())
            self.label_info.setText("提示：请先在主界面打开图片。")
            return

        # 1. 先把 QPixmap 转成灰度 QImage
        qimg = pm.toImage().convertToFormat(QImage.Format_Grayscale8)
        w, h = qimg.width(), qimg.height()
        bpl = qimg.bytesPerLine()  # bytes per line

        # 2. 拷贝到 numpy，注意：切片后要 copy()，保证是 C-contiguous
        buf = qimg.bits().tobytes()
        arr = (
            np.frombuffer(buf, dtype=np.uint8)
            .reshape((h, bpl))[:, :w]   # 先裁剪多余的字节
            .copy()                     # ★ 关键：变成 C-contiguous
        )

        # 3. 保存为原始 / 当前灰度图
        self._gray_orig = arr.copy()
        self._gray_curr = arr.copy()

        # 4. 左边显示的灰度 pixmap：直接用 qimg 或用 arr 都可以
        #   这里用 arr 再构造一个 QImage 也没问题，因为现在是连续内存了
        qimg_gray = QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._orig_pixmap = QPixmap.fromImage(qimg_gray)

        # 5. 初始结果就是原图
        self._update_result_pixmap()
        self._refresh_previews()
        self.label_info.setText("已载入灰度图，选择操作和参数后点击『应用』。")


    # ---------------- 形态学操作 ----------------
    def _apply_operation(self):
        if self._gray_curr is None:
            self.label_info.setText("当前没有灰度图，请先打开图片。")
            return

        img = self._gray_curr.astype(np.uint8)
        iters = self.spin_iter.value()
        ksize = self.spin_ksize.value()
        # 保险起见，保证是奇数
        if ksize % 2 == 0:
            ksize += 1
            self.spin_ksize.setValue(ksize)

        kernel = cv.getStructuringElement(cv.MORPH_RECT, (ksize, ksize))
        op = self.combo_op.currentData()

        if op == "dilate":
            out = cv.dilate(img, kernel, iterations=iters)
        elif op == "erode":
            out = cv.erode(img, kernel, iterations=iters)
        elif op == "open":
            out = cv.morphologyEx(img, cv.MORPH_OPEN, kernel, iterations=iters)
        elif op == "close":
            out = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel, iterations=iters)
        else:
            out = img

        self._gray_curr = out
        self._update_result_pixmap()
        self._refresh_previews()

        op_text = self.combo_op.currentText()
        self.label_info.setText(
            f"{op_text} 已应用 {iters} 次，核大小 {ksize}×{ksize}。"
        )

    def _reset_image(self):
        """恢复到原始灰度图。"""
        if self._gray_orig is None:
            return
        self._gray_curr = self._gray_orig.copy()
        self._update_result_pixmap()
        self._refresh_previews()
        self.label_info.setText("已重置为原始灰度图。")

    # ---------------- 保存结果 ----------------
    def _save_result(self):
        if self._result_pixmap is None:
            QMessageBox.information(self, "保存结果", "当前没有可保存的结果，请先应用形态学操作。")
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

    # ---------------- 显示 / 刷新 ----------------
    def _update_result_pixmap(self):
        """把当前灰度图 _gray_curr 更新到右侧 label。"""
        if self._gray_curr is None:
            self.label_result.setPixmap(QPixmap())
            self._result_pixmap = None
            return

        img = np.clip(self._gray_curr, 0, 255).astype(np.uint8)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._result_pixmap = QPixmap.fromImage(qimg)
        self.label_result.setText("")

    def _refresh_previews(self):
        # 左：原图（灰度）
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

        # 右：当前结果
        if self._result_pixmap is not None:
            target = self.label_result.size()
            if target.width() > 0 and target.height() > 0:
                scaled = self._result_pixmap.scaled(
                    target, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.label_result.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._refresh_previews()
