# task3.py
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
    QGridLayout,
)
import numpy as np


class JLabelWithBg(QLabel):
    """统一风格的图像显示 QLabel。"""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setStyleSheet("background: #202020; color: white;")
        self.setScaledContents(False)


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
    QGridLayout,
    QSizePolicy,   # ⭐ 新增
)
...
class StructElemWidget(QWidget):
    """嵌在界面里的结构元素示意图（3×3 正方形，中心用红框标记）"""

    def __init__(self, se: np.ndarray, parent=None):
        super().__init__(parent)
        self._se = se
        self._build_ui()

    def _build_ui(self):
        grid = QGridLayout(self)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(1)

        h, w = self._se.shape
        ch, cw = h // 2, w // 2      # 中心位置
        cell_size = 20               # 每个格子 20x20

        for i in range(h):
            for j in range(w):
                cell = QLabel()
                cell.setFixedSize(cell_size, cell_size)
                cell.setAlignment(Qt.AlignCenter)

                if self._se[i, j]:
                    # 结构元素中为 1 的格子：白色
                    if i == ch and j == cw:
                        # 中心格子：红色粗边框
                        cell.setStyleSheet(
                            "background: white; border: 2px solid red;"
                        )
                    else:
                        cell.setStyleSheet(
                            "background: white; border: 1px solid gray;"
                        )
                else:
                    # 为 0 的位置：深灰色
                    cell.setStyleSheet(
                        "background: #404040; border: 1px solid gray;"
                    )

                grid.addWidget(cell, i, j)

        # 不让这个小控件被拉伸变形，保持正方形格子
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)



class Task3Window(QWidget):
    """
    Project-3: 二值形态学
    - binary dilation / erosion / opening / closing
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = parent

        # 状态
        self._orig_pixmap: QPixmap | None = None      # 原图（灰度）
        self._gray: np.ndarray | None = None          # 灰度图
        self._binary_orig: np.ndarray | None = None   # 初始二值图（0/1）
        self._binary_curr: np.ndarray | None = None   # 当前二值图（0/1）
        self._result_pixmap: QPixmap | None = None    # 当前结果图（0/255）

        # 结构元素（默认 3×3 全 1）
        self._se = np.ones((3, 3), dtype=bool)
        self.struct_widget: StructElemWidget | None = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        main = QVBoxLayout(self)

        # 上：左右图
        img_layout = QHBoxLayout()
        self.label_orig = JLabelWithBg("原图（灰度）")
        self.label_result = JLabelWithBg("形态学结果（二值）")
        img_layout.addWidget(self.label_orig, 1)
        img_layout.addWidget(self.label_result, 1)

        # 中：操作控制区
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("操作:"))

        self.combo_op = QComboBox()
        self.combo_op.addItem("膨胀 (dilation)", "dilate")
        self.combo_op.addItem("腐蚀 (erosion)", "erode")
        self.combo_op.addItem("开运算 (opening)", "open")
        self.combo_op.addItem("闭运算 (closing)", "close")
        ctrl_layout.addWidget(self.combo_op)

        # 迭代次数（反复操作）
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("迭代次数:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(1, 20)
        self.spin_iter.setValue(1)
        ctrl_layout.addWidget(self.spin_iter)

        # 按钮
        btn_apply = QPushButton("应用")
        btn_apply.clicked.connect(self._apply_operation)

        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self._reset_image)

        btn_show_se = QPushButton("结构元素")
        btn_show_se.setCheckable(True) 
        btn_show_se.clicked.connect(self._toggle_struct_elem)

        btn_save = QPushButton("保存结果")
        btn_save.clicked.connect(self._save_result)

        self.struct_widget = StructElemWidget(self._se, self)
        self.struct_widget.setVisible(False)

        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(btn_apply)
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(btn_reset)
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(btn_show_se)
        ctrl_layout.addSpacing(5)
        ctrl_layout.addWidget(self.struct_widget)   
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(btn_save)
        ctrl_layout.addStretch()

        # 下：提示信息
        self.label_info = QLabel("提示：请先在主界面打开图片，然后切换到 功能3。")
        self.label_info.setAlignment(Qt.AlignLeft)

        main.addLayout(img_layout, 3)
        main.addLayout(ctrl_layout, 0)
        main.addWidget(self.label_info, 0)

    # ---------------- 对外接口：由 ImageViewer 调用 ----------------
    def set_image(self, pm: QPixmap | None):
        """
        从 ImageViewer 传入当前图像：
        - 左侧显示灰度原图
        - 右侧显示固定阈值(128)的二值图，作为形态学的初始输入
        """
        self._orig_pixmap = None
        self._gray = None
        self._binary_orig = None
        self._binary_curr = None
        self._result_pixmap = None

        if pm is None:
            self.label_orig.setText("原图：无图像")
            self.label_orig.setPixmap(QPixmap())
            self.label_result.setText("结果：无图像")
            self.label_result.setPixmap(QPixmap())
            self.label_info.setText("提示：请先在主界面打开图片。")
            return

        # 转灰度
        qimg = pm.toImage().convertToFormat(QImage.Format_Grayscale8)
        w, h = qimg.width(), qimg.height()
        bpl = qimg.bytesPerLine()
        ptr = qimg.bits()
        buf = ptr.tobytes()
        arr = np.frombuffer(buf, np.uint8).reshape((h, bpl))[:, :w]
        self._gray = arr.astype(np.float64)

        # 灰度版原图 pixmap（注意 bytesPerLine 用 w）
        qimg_gray = QImage(
            arr.astype(np.uint8).data,
            w,
            h,
            w,
            QImage.Format_Grayscale8,
        ).copy()
        self._orig_pixmap = QPixmap.fromImage(qimg_gray)

        # 创建初始二值图（阈值固定 128，可按需要再扩展 UI）
        binary = (arr >= 128).astype(np.uint8)  # 0 / 1
        self._binary_orig = binary
        self._binary_curr = binary.copy()
        self._update_result_pixmap()

        self.label_info.setText("已生成初始二值图，选择形态学操作和迭代次数后点击『应用』。")
        self._refresh_previews()

    # ---------------- 形态学操作 ----------------
    def _apply_operation(self):
        if self._binary_curr is None:
            self.label_info.setText("当前没有二值图，请先打开图片。")
            return

        op = self.combo_op.currentData()
        iters = self.spin_iter.value()
        img = self._binary_curr

        for _ in range(iters):
            if op == "dilate":
                img = self._dilate(img, self._se)
            elif op == "erode":
                img = self._erode(img, self._se)
            elif op == "open":
                img = self._dilate(self._erode(img, self._se), self._se)
            elif op == "close":
                img = self._erode(self._dilate(img, self._se), self._se)

        self._binary_curr = img
        self._update_result_pixmap()
        self._refresh_previews()

        op_text = self.combo_op.currentText()
        self.label_info.setText(f"{op_text} 已应用 {iters} 次。")

    def _reset_image(self):
        """恢复到初始二值图。"""
        if self._binary_orig is None:
            return
        self._binary_curr = self._binary_orig.copy()
        self._update_result_pixmap()
        self._refresh_previews()
        self.label_info.setText("已重置为初始二值图。")

    # --- 基本形态学算子：dilate / erode ---
    def _dilate(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        img_bool = img.astype(bool)
        se_bool = se.astype(bool)
        h, w = img_bool.shape
        sh, sw = se_bool.shape
        ph, pw = sh // 2, sw // 2

        padded = np.pad(img_bool, ((ph, ph), (pw, pw)), mode="constant", constant_values=False)
        out = np.zeros_like(img_bool)

        coords = np.argwhere(se_bool)  # se 中为 1 的相对坐标
        for i in range(h):
            for j in range(w):
                region = padded[i : i + sh, j : j + sw]
                # 只看结构元素为 1 的位置
                out[i, j] = np.any(region[se_bool])

        return out.astype(np.uint8)

    def _erode(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        img_bool = img.astype(bool)
        se_bool = se.astype(bool)
        h, w = img_bool.shape
        sh, sw = se_bool.shape
        ph, pw = sh // 2, sw // 2

        padded = np.pad(img_bool, ((ph, ph), (pw, pw)), mode="constant", constant_values=False)
        out = np.zeros_like(img_bool)

        coords = np.argwhere(se_bool)
        for i in range(h):
            for j in range(w):
                region = padded[i : i + sh, j : j + sw]
                out[i, j] = np.all(region[se_bool])

        return out.astype(np.uint8)

    # ---------------- 显示结构元的小窗 ----------------
    def _toggle_struct_elem(self, checked: bool):
        """结构元素按钮的开关：显示 / 隐藏示意图"""
        if self.struct_widget is not None:
            self.struct_widget.setVisible(checked)

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
        """把当前二值图 _binary_curr 更新到右侧 label。"""
        if self._binary_curr is None:
            self.label_result.setPixmap(QPixmap())
            self._result_pixmap = None
            return

        img = (self._binary_curr * 255).astype(np.uint8)
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

        # 右：当前二值结果
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
