from pathlib import Path

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QAction, QKeySequence, QPixmap, QImageReader
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QToolBar,
)

class ImageViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Viewer (PySide6)")
        self.resize(980, 720)
        self.setAcceptDrops(True)

        # --- 中心视图：可滚动的 QLabel 显示图像 ---
        self._pm_orig: QPixmap | None = None
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setBackgroundRole(self.image_label.backgroundRole())
        self.image_label.setScaledContents(False)  # 用缩放函数控制，避免失真

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(self.scroll_area)

        # --- 状态 ---
        self._pixmap: QPixmap | None = None
        self._scale = 1.0
        self._fit_to_window = False

        # --- 构建 UI ---
        self._create_actions()
        self._create_menus_and_toolbar()
        self._update_actions()
        self._update_status()

    # ---------------- UI 构建 ----------------
    def _create_actions(self) -> None:
        self.act_open = QAction("打开…", self, shortcut=QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.open_file)

        self.act_exit = QAction("退出", self, shortcut=QKeySequence.StandardKey.Quit)
        self.act_exit.triggered.connect(self.close)

        self.act_zoom_in = QAction("放大", self, shortcut=QKeySequence(Qt.CTRL | Qt.Key_Equal))
        self.act_zoom_in.triggered.connect(lambda: self.scale_image(1.25))

        self.act_zoom_out = QAction("缩小", self, shortcut=QKeySequence(Qt.CTRL | Qt.Key_Minus))
        self.act_zoom_out.triggered.connect(lambda: self.scale_image(0.8))

        self.act_zoom_reset = QAction("100%", self, shortcut=QKeySequence(Qt.CTRL | Qt.Key_0))
        self.act_zoom_reset.triggered.connect(self.reset_zoom)

        self.act_fit = QAction("适应窗口", self, checkable=True, shortcut=QKeySequence(Qt.CTRL | Qt.Key_F))
        self.act_fit.toggled.connect(self.set_fit_to_window)

    def _create_menus_and_toolbar(self) -> None:
        menu_file = self.menuBar().addMenu("文件(&F)")
        menu_file.addAction(self.act_open)
        menu_file.addSeparator()
        menu_file.addAction(self.act_exit)

        menu_view = self.menuBar().addMenu("视图(&V)")
        menu_view.addAction(self.act_fit)
        menu_view.addSeparator()
        menu_view.addAction(self.act_zoom_in)
        menu_view.addAction(self.act_zoom_out)
        menu_view.addAction(self.act_zoom_reset)

        tb = QToolBar("Main", self)
        tb.setIconSize(QSize(18, 18))
        self.addToolBar(tb)
        tb.addAction(self.act_open)
        tb.addSeparator()
        tb.addAction(self.act_fit)
        tb.addAction(self.act_zoom_in)
        tb.addAction(self.act_zoom_out)
        tb.addAction(self.act_zoom_reset)

        self.statusBar()

    # ---------------- 文件/加载 ----------------
    def open_file(self) -> None:
        caption = "选择图片文件"
        # 由 Qt 插件列出支持的格式
        fmts = [bytes(f).decode().lower() for f in QImageReader.supportedImageFormats()]
        pattern = " ".join(f"*.{ext}" for ext in sorted(set(fmts))) or "*"
        fname, _ = QFileDialog.getOpenFileName(self, caption, "", f"Images ({pattern})")
        if fname:
            self.load_path(Path(fname))

    def load_path(self, path: Path) -> None:
        reader = QImageReader(str(path))
        reader.setAutoTransform(True)
        img = reader.read()
        if img.isNull():
            QMessageBox.critical(self, "读取失败", f"无法读取文件：\n{path}\n\n错误：{reader.errorString()}")
            return
        self.set_pixmap(QPixmap.fromImage(img))
        self.setWindowTitle(f"{path.name} — Image Viewer (PySide6)")

    def set_pixmap(self, pm: QPixmap) -> None:
        self._pixmap = pm
        self._scale = 1.0
        self._fit_to_window = False
        self.act_fit.setChecked(False)
        self.image_label.setPixmap(pm)
        self.image_label.adjustSize()
        self._apply_scale(1.0)
        self._update_actions()
        self._update_status()

    # ---------------- 视图/缩放 ----------------
    def set_fit_to_window(self, enabled: bool) -> None:
        self._fit_to_window = enabled
        self.image_label.setScaledContents(False)
        if enabled:
            self._fit_image_to_scrollarea()
        else:
            self.reset_zoom()
        self._update_actions()
        self._update_status()

    def _fit_image_to_scrollarea(self) -> None:
        if not self._pm_orig:
            return
        area = self.scroll_area.viewport().size()
        pm_size = self._pm_orig.size()
        if pm_size.isEmpty():
            return
        sx = area.width() / pm_size.width()
        sy = area.height() / pm_size.height()
        self._apply_scale(min(sx, sy))

    def _adjust_scrollbars(self, factor: float) -> None:
        for sb in (self.scroll_area.horizontalScrollBar(), self.scroll_area.verticalScrollBar()):
            sb.setValue(int(factor * sb.value() + ((factor - 1) * sb.pageStep() / 2)))

    def scale_image(self, factor: float) -> None:
        if not self._pm_orig or self._fit_to_window:
            return
        new_scale = max(0.05, min(self._scale * factor, 40.0))
        if new_scale == self._scale:
            return
        ratio = new_scale / self._scale
        self._apply_scale(new_scale)
        self._adjust_scrollbars(ratio)

    def reset_zoom(self) -> None:
        if not self._pixmap:
            return
        self._apply_scale(1.0)

    def _apply_scale(self, new_scale: float) -> None:
        assert self._pm_orig is not None
        self._scale = new_scale
        orig_size = self._pm_orig.size()
        w = max(1, int(round(orig_size.width() * self._scale)))
        h = max(1, int(round(orig_size.height() * self._scale)))

        scaled = self._pm_orig.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())
        self._update_status()
        
        
    # ---------------- 交互增强：拖拽 & 滚轮缩放 ----------------
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()
        else:
            super().dragEnterEvent(e)

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            p = Path(urls[0].toLocalFile())
            if p.is_file():
                self.load_path(p)
        else:
            super().dropEvent(e)

    def wheelEvent(self, e):
        # Ctrl + 滚轮进行缩放；否则交给父类处理（滚动）
        if QApplication.keyboardModifiers() & Qt.ControlModifier and self._pixmap and not self._fit_to_window:
            angle = e.angleDelta().y()
            factor = 1.25 if angle > 0 else 0.8
            self.scale_image(factor)
            e.accept()
        else:
            super().wheelEvent(e)

    # ---------------- 辅助 ----------------
    def _update_actions(self) -> None:
        has_img = self._pixmap is not None
        self.act_fit.setEnabled(has_img)
        self.act_zoom_in.setEnabled(has_img and not self._fit_to_window)
        self.act_zoom_out.setEnabled(has_img and not self._fit_to_window)
        self.act_zoom_reset.setEnabled(has_img and not self._fit_to_window)

    def _update_status(self) -> None:
        if not self._pixmap:
            self.statusBar().showMessage("就绪")
            return
        pm_size = self._pixmap.size()
        percent = int(round(self._scale * 100))
        fit = "(适应窗口)" if self._fit_to_window else ""
        self.statusBar().showMessage(f"{pm_size.width()}×{pm_size.height()}  |  {percent}% {fit}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit_to_window:
            self._fit_image_to_scrollarea()
