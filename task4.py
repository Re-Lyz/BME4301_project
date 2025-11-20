# task4.py
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap, QImage
from qtpy.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QSpinBox,
    QSizePolicy,
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


class Task4Window(QWidget):
    """
    Project-4:
      - Morphological distance transform
      - Morphological skeleton
      - Morphological skeleton restoration
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.viewer = parent

        # ===== 状态 =====
        self._orig_pixmap: QPixmap | None = None      # 灰度原图
        self._gray: np.ndarray | None = None          # 灰度数组
        self._binary: np.ndarray | None = None        # 初始二值图（0/1）

        self._result_pixmap: QPixmap | None = None    # 当前显示结果
        self._distance: np.ndarray | None = None      # 距离变换结果（float）
        self._skeleton: np.ndarray | None = None      # 骨架（二值 0/1）
        self._skel_layers: list[np.ndarray] | None = None  # 每一层 S_k
        self._restored: np.ndarray | None = None      # 骨架重建结果（二值 0/1）

        self._se_type: str = "square"
        self._se_size: int = 3
        self._se = np.ones((self._se_size, self._se_size), dtype=bool)

        self._build_ui()
        self._update_se_from_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        main = QVBoxLayout(self)

        # 上：左右图像
        img_layout = QHBoxLayout()
        self.label_orig = JLabelWithBg("原图（灰度）")
        self.label_result = JLabelWithBg("结果图")
        img_layout.addWidget(self.label_orig, 1)
        img_layout.addWidget(self.label_result, 1)

        # 中：操作区
        ctrl_layout = QHBoxLayout()
        ctrl_layout.addWidget(QLabel("操作:"))

        self.combo_op = QComboBox()
        self.combo_op.addItem("距离变换", "distance")
        self.combo_op.addItem("骨架提取", "skeleton")
        self.combo_op.addItem("骨架重建", "restore")
        ctrl_layout.addWidget(self.combo_op)

        # ===== 新增：结构元素设置 =====
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("结构元素:"))
    
        self.combo_se_type = QComboBox()
        self.combo_se_type.addItem("正方形", "square")
        self.combo_se_type.addItem("十字架", "cross")
        self.combo_se_type.addItem("圆形", "disk")
        ctrl_layout.addWidget(self.combo_se_type)
    
        ctrl_layout.addSpacing(10)
        ctrl_layout.addWidget(QLabel("SE大小:"))
    
        self.spin_se_size = QSpinBox()
        self.spin_se_size.setRange(1, 99)      # 只用奇数，下面回调里会修正
        self.spin_se_size.setSingleStep(2)     # 步长 2：3,5,7,...
        self.spin_se_size.setValue(3)
        self.spin_se_size.setToolTip("结构元素外接正方形的边长（建议使用奇数）")
        ctrl_layout.addWidget(self.spin_se_size)
    
        # 当用户修改 SE 类型或大小时，实时更新 self._se
        self.combo_se_type.currentIndexChanged.connect(self._update_se_from_ui)
        self.spin_se_size.valueChanged.connect(self._update_se_from_ui)

        # 可选：迭代上限（影响距离变换和骨架的最大迭代次数）
        ctrl_layout.addSpacing(20)
        ctrl_layout.addWidget(QLabel("最大迭代:"))
        self.spin_max_iter = QSpinBox()
        self.spin_max_iter.setRange(1, 200)
        self.spin_max_iter.setValue(100)
        self.spin_max_iter.setToolTip("限制最多迭代次数，避免过慢")
        ctrl_layout.addWidget(self.spin_max_iter)

        # 按钮们
        btn_apply = QPushButton("执行")
        btn_apply.clicked.connect(self._apply_operation)

        btn_reset = QPushButton("重置")
        btn_reset.clicked.connect(self._reset)

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
        self.label_info = QLabel("提示：请先在主界面打开图片，然后切换到 功能4。")
        self.label_info.setAlignment(Qt.AlignLeft)

        main.addLayout(img_layout, 3)
        main.addLayout(ctrl_layout, 0)
        main.addWidget(self.label_info, 0)

    def _update_se_from_ui(self):
        """从界面读取 SE 类型和大小，更新 self._se。"""
        if not hasattr(self, "combo_se_type"):
            # 构造过程中可能会提前被调用一下，防御一下
            return
    
        se_type = self.combo_se_type.currentData()
        size = self.spin_se_size.value()
    
        # 保证 size >= 1 且为奇数（形态学运算居中好处理）
        if size < 1:
            size = 1
        if size % 2 == 0:
            size += 1
            # 防止递归 signal
            self.spin_se_size.blockSignals(True)
            self.spin_se_size.setValue(size)
            self.spin_se_size.blockSignals(False)
    
        self._se_type = se_type
        self._se_size = size
        self._se = self._create_se(se_type, size)
    
        # 可以顺便在状态栏展示一下
        self.label_info.setText(
            f"当前结构元素: {se_type}, 尺寸 {size}×{size}"
        )
    
    def _create_se(self, se_type: str, size: int) -> np.ndarray:
        """根据类型和外接正方形边长生成结构元素（布尔数组）。"""
        if size <= 1:
            return np.ones((1, 1), dtype=bool)
    
        if se_type == "square":
            # 正方形：全 1
            return np.ones((size, size), dtype=bool)
    
        h = w = size
        se = np.zeros((h, w), dtype=bool)
        cy, cx = h // 2, w // 2
    
        if se_type == "cross":
            # 十字架：中心行 + 中心列
            se[cy, :] = True
            se[:, cx] = True
            return se
    
        if se_type == "disk":
            # 圆形：外接正方形内，(x-cx)^2 + (y-cy)^2 <= r^2
            r = size / 2.0
            for y in range(h):
                for x in range(w):
                    if (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2:
                        se[y, x] = True
            return se
    
        # 兜底：未知类型默认正方形
        return np.ones((size, size), dtype=bool)


    # ---------------- 对外接口：由 ImageViewer 调用 ----------------
    def set_image(self, pm: QPixmap | None):
        """从主窗口接收图像，生成灰度图 + 初始二值图。"""
        self._orig_pixmap = None
        self._gray = None
        self._binary = None
        self._result_pixmap = None
        self._distance = None
        self._skeleton = None
        self._skel_layers = None
        self._restored = None

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

        # 灰度 pixmap（注意 bytesPerLine 用 w）
        qimg_gray = QImage(
            arr.astype(np.uint8).data,
            w,
            h,
            w,
            QImage.Format_Grayscale8,
        ).copy()
        self._orig_pixmap = QPixmap.fromImage(qimg_gray)

        # 初始二值图（简单固定阈值 128，可以后扩展）
        binary = (arr >= 128).astype(np.uint8)  # 0/1
        self._binary = binary

        # 默认结果：显示二值图
        self._update_result_from_binary(binary)
        self.label_info.setText("已生成初始二值图，可选择操作执行距离变换 / 骨架 / 重建。")
        self._refresh_previews()

    # ---------------- 按钮逻辑 ----------------
    def _apply_operation(self):
        if self._binary is None:
            self.label_info.setText("当前没有二值图，请先打开图片。")
            return

        op = self.combo_op.currentData()
        max_iter = self.spin_max_iter.value()

        if op == "distance":
            self._compute_distance(max_iter)
            self._show_distance()
            self.label_info.setText("距离变换完成。")
        elif op == "skeleton":
            self._compute_skeleton(max_iter)
            self._show_skeleton()
            self.label_info.setText("骨架提取完成。")
        elif op == "restore":
            # 若之前没算过骨架，则先算一遍
            if self._skel_layers is None:
                self._compute_skeleton(max_iter)
            self._compute_restoration()
            self._show_restoration()
            self.label_info.setText("骨架重建完成。")

        self._refresh_previews()

    def _reset(self):
        """恢复到初始二值图。"""
        if self._binary is None:
            return
        self._distance = None
        self._skeleton = None
        self._skel_layers = None
        self._restored = None
        self._update_result_from_binary(self._binary)
        self._refresh_previews()
        self.label_info.setText("已重置为初始二值图。")

    def _save_result(self):
        if self._result_pixmap is None:
            QMessageBox.information(self, "保存结果", "当前没有可保存的结果。")
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

    # ---------------- 形态学基础算子 ----------------
    def _dilate(self, img: np.ndarray, se: np.ndarray) -> np.ndarray:
        img_bool = img.astype(bool)
        se_bool = se.astype(bool)
        h, w = img_bool.shape
        sh, sw = se_bool.shape
        ph, pw = sh // 2, sw // 2

        padded = np.pad(img_bool, ((ph, ph), (pw, pw)), mode="constant", constant_values=False)
        out = np.zeros_like(img_bool)

        for i in range(h):
            for j in range(w):
                region = padded[i : i + sh, j : j + sw]
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

        for i in range(h):
            for j in range(w):
                region = padded[i : i + sh, j : j + sw]
                out[i, j] = np.all(region[se_bool])

        return out.astype(np.uint8)

    def _open(self, img: np.ndarray) -> np.ndarray:
        return self._dilate(self._erode(img, self._se), self._se)

    # ---------------- 距离变换 ----------------
    def _compute_distance(self, max_iter: int):
        """
        形态学距离变换：
        对每个前景像素，计算其在多少次腐蚀后被“吃掉”。
        """
        binary = self._binary.astype(np.uint8)
        h, w = binary.shape
        dist = np.zeros((h, w), dtype=np.float64)

        current = binary.copy()
        k = 0
        while np.any(current) and k < max_iter:
            k += 1
            eroded = self._erode(current, self._se)
            # 本次被腐蚀掉的边界像素：current=1 且 eroded=0
            border = (current == 1) & (eroded == 0)
            dist[border] = k
            current = eroded

        # 对从未被标记的像素（最里层），赋最大 k
        dist[(binary == 1) & (dist == 0)] = k

        self._distance = dist

    def _show_distance(self):
        if self._distance is None:
            return
        dist = self._distance
        maxv = dist.max()
        if maxv <= 0:
            img = np.zeros_like(dist, dtype=np.uint8)
        else:
            img = (dist / maxv * 255.0).astype(np.uint8)

        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._result_pixmap = QPixmap.fromImage(qimg)
        self.label_result.setText("形态学距离变换")

    # ---------------- 骨架提取 ----------------
    def _compute_skeleton(self, max_iter: int):
        """
        经典形态学骨架：
          A_k = A ⊖ kB
          S_k = A_k \ (A_k ∘ B)
          Skeleton = ⋃_k S_k
        这里用迭代实现，顺便保存每一层 S_k（用于重建）。
        """
        A = self._binary.astype(np.uint8)
        skel_layers: list[np.ndarray] = []
        skel_union = np.zeros_like(A, dtype=np.uint8)

        current = A.copy()
        k = 0
        while np.any(current) and k < max_iter:
            k += 1
            eroded = self._erode(current, self._se)
            opened = self._open(eroded)
            S_k = (eroded == 1) & (opened == 0)
            S_k_u8 = S_k.astype(np.uint8)
            
            skel_layers.append(S_k_u8)
            skel_union = np.maximum(skel_union, S_k_u8)

            current = eroded

        self._skeleton = skel_union
        self._skel_layers = skel_layers

    def _show_skeleton(self):
        if self._skeleton is None:
            return
        img = (self._skeleton * 255).astype(np.uint8)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._result_pixmap = QPixmap.fromImage(qimg)
        self.label_result.setText("形态学骨架")

    # ---------------- 骨架重建 ----------------
    def _compute_restoration(self):
        """
        骨架重建：
          A = ⋃_k (S_k ⊕ kB)
        我们用迭代的方式：对第 k 层骨架 S_k 进行 k 次膨胀，然后取并集。
        """
        if self._skel_layers is None or len(self._skel_layers) == 0:
            return

        h, w = self._skel_layers[0].shape
        restored = np.zeros((h, w), dtype=np.uint8)

        for idx, S_k in enumerate(self._skel_layers, start=1):
            img = S_k.copy()
            # 对第 k 层执行 k 次膨胀
            for _ in range(idx):
                img = self._dilate(img, self._se)
            restored = np.maximum(restored, img)

        self._restored = restored

    def _show_restoration(self):
        if self._restored is None:
            return
        img = (self._restored * 255).astype(np.uint8)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._result_pixmap = QPixmap.fromImage(qimg)
        self.label_result.setText("骨架重建")

    # ---------------- 显示 / 刷新 ----------------
    def _update_result_from_binary(self, binary: np.ndarray):
        img = (binary * 255).astype(np.uint8)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
        self._result_pixmap = QPixmap.fromImage(qimg)
        self.label_result.setText("初始二值图")

    def _refresh_previews(self):
        # 左：原灰度图
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

        # 右：结果图
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
