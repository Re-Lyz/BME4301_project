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
    QWidget,
    QVBoxLayout,
    QStackedWidget
)
from task1 import Task1Window
from task2 import Task2Window
from task3 import Task3Window
from task4 import Task4Window
from task5 import Task5Window

class ImageViewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Image Viewer (PySide6)")
        self.resize(980, 720)
        self.setAcceptDrops(True)

        # --- 状态 ---
        self._pm_orig: QPixmap | None = None   # 原始图
        self._pixmap: QPixmap | None = None    # 当前（缩放后）图
        self._scale = 1.0
        self._fit_to_window = False

        # ====== 1. 创建 QStackedWidget 作为中心部件 ======
        self.stack = QStackedWidget(self)
        self.setCentralWidget(self.stack)

        # ====== 2. 第 0 页：原来的图像浏览界面 ======
        self.image_label = QLabel(alignment=Qt.AlignCenter)
        self.image_label.setBackgroundRole(self.image_label.backgroundRole())
        self.image_label.setScaledContents(False)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidget(self.image_label)
        self.scroll_area.setAlignment(Qt.AlignCenter)

        # 用一个 QWidget 包住 scroll_area，再放进 stack
        self.page_image = QWidget()
        layout0 = QVBoxLayout(self.page_image)
        layout0.setContentsMargins(0, 0, 0, 0)
        layout0.addWidget(self.scroll_area)

        self.stack.addWidget(self.page_image)   # index = 0

        # ====== 3. 预先创建几个功能页面（空壳，之后你再填内容） ======
        self.page_func1 = Task1Window(self)
        self.page_func2 = Task2Window(self)
        self.page_func3 = Task3Window(self)
        self.page_func4 = Task4Window(self)
        self.page_func5 = Task5Window(self)
       
        self.page_func6 = QWidget()
        l6 = QVBoxLayout(self.page_func6)
        l6.addWidget(QLabel("这里是 功能6 页面"))

        self.stack.addWidget(self.page_func1)   # index = 1
        self.stack.addWidget(self.page_func2)   # index = 2
        self.stack.addWidget(self.page_func3)   # index = 3
        self.stack.addWidget(self.page_func4)   # index = 4
        self.stack.addWidget(self.page_func5)   # index = 5
        self.stack.addWidget(self.page_func6)   # index = 6
        # 默认显示图像浏览页面
        self.stack.setCurrentWidget(self.page_image)

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

        # ------ 功能菜单相关动作 ------
        self.act_home = QAction("回到图像浏览", self)
        self.act_home.triggered.connect(self.show_image_page)

        self.act_func1 = QAction("功能一：灰度直方图与均衡化", self)
        self.act_func1.triggered.connect(self.show_func1_page)

        self.act_func2 = QAction("功能二：卷积与滤波", self)
        self.act_func2.triggered.connect(self.show_func2_page)

        self.act_func3 = QAction("功能三：二值形态学基础功能", self)
        self.act_func3.triggered.connect(self.show_func3_page)
        
        self.act_func4 = QAction("功能四：二值形态学进阶功能", self)
        self.act_func4.triggered.connect(self.show_func4_page)
        
        self.act_func5 = QAction("功能五：灰度形态学基础功能", self)
        self.act_func5.triggered.connect(self.show_func5_page)
        
        # self.act_func6 = QAction("功能六：任务6", self)
        # self.act_func6.triggered.connect(self.show_func6_page)

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

        # ------ 功能菜单 ------
        menu_func = self.menuBar().addMenu("功能(&G)")
        menu_func.addAction(self.act_home)
        menu_func.addSeparator()
        menu_func.addAction(self.act_func1)
        menu_func.addAction(self.act_func2)
        menu_func.addAction(self.act_func3)
        menu_func.addAction(self.act_func4)
        menu_func.addAction(self.act_func5)
        # menu_func.addAction(self.act_func6)

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
        fmts = [bytes(f).decode().lower() for f in QImageReader.supportedImageFormats()]
        pattern = " ".join(f"*.{ext}" for ext in sorted(set(fmts))) or "*"
        fname, _ = QFileDialog.getOpenFileName(self, caption, "", f"Images ({pattern})")
        if fname:
            self.load_path(Path(fname))

            # 根据当前所在页面决定要不要更新功能1
            current = self.stack.currentWidget()
            if current is self.page_func1:
                # 如果此时正在功能1界面，直接把新图送进去
                self.page_func1.set_image(self._pm_orig)
            # 如果当前在图像浏览页，就什么都不做（自然就是默认界面）
            elif current is self.page_func2:
                self.page_func2.set_image(self._pm_orig)
            elif current is self.page_func3:
                self.page_func3.set_image(self._pm_orig)
            elif current is self.page_func4:
                self.page_func4.set_image(self._pm_orig)
            elif current is self.page_func5:
                self.page_func5.set_image(self._pm_orig)
            # elif current is self.page_func6:
            #     self.page_func6.set_image(self._pm_orig)
                

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
        """设置新的图像并重置缩放状态"""
        self._pm_orig = pm          # 记录原始图
        self._pixmap = pm           # 先让当前图等于原始图
        self._scale = 1.0
        self._fit_to_window = False
        self.act_fit.setChecked(False)

        # 用统一的缩放逻辑来设置 QLabel
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
        # 只有在图像浏览页才允许缩放
        if self.stack.currentWidget() is not self.page_image:
            return
        if not self._pm_orig or self._fit_to_window:
            return
        new_scale = max(0.05, min(self._scale * factor, 40.0))
        if new_scale == self._scale:
            return
        ratio = new_scale / self._scale
        self._apply_scale(new_scale)
        self._adjust_scrollbars(ratio)

    def reset_zoom(self) -> None:
        if self.stack.currentWidget() is not self.page_image:
            return
        if not self._pixmap:
            return
        self._apply_scale(1.0)

    def _apply_scale(self, new_scale: float) -> None:
        """根据 new_scale 缩放原始图像并更新 QLabel"""
        if self._pm_orig is None:
            return  # 防御式写法，避免再次抛错

        self._scale = new_scale
        orig_size = self._pm_orig.size()
        w = max(1, int(round(orig_size.width() * self._scale)))
        h = max(1, int(round(orig_size.height() * self._scale)))

        # 以原图为基准生成缩放图
        scaled = self._pm_orig.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._pixmap = scaled  # 更新当前显示图
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
    
                current = self.stack.currentWidget()
                if current is self.page_func1:
                    self.page_func1.set_image(self._pm_orig)
                elif current is self.page_func2:
                    self.page_func2.set_image(self._pm_orig)
                elif current is self.page_func3:
                    self.page_func3.set_image(self._pm_orig)
                elif current is self.page_func4:
                    self.page_func4.set_image(self._pm_orig)
                elif current is self.page_func5:
                    self.page_func5.set_image(self._pm_orig)
                # elif current is self.page_func6:
                #     self.page_func6.set_image(self._pm_orig)
        else:
            super().dropEvent(e)

    def wheelEvent(self, e):
        # Ctrl + 滚轮进行缩放；仅在图像浏览页有效；否则交给父类处理（滚动）
        if (
            QApplication.keyboardModifiers() & Qt.ControlModifier
            and self._pixmap
            and not self._fit_to_window
            and self.stack.currentWidget() is self.page_image
        ):
            angle = e.angleDelta().y()
            factor = 1.25 if angle > 0 else 0.8
            self.scale_image(factor)
            e.accept()
        else:
            super().wheelEvent(e)

    # ---------------- stack 页面切换 ----------------
    def show_image_page(self):
        self.stack.setCurrentWidget(self.page_image)
        self._update_actions()
        self._update_status()

    def show_func1_page(self):
        # 先把当前原始图像传给功能1
        self.page_func1.set_image(self._pm_orig)
        # 再切换页面
        self.stack.setCurrentWidget(self.page_func1)
        self._update_actions()
        self._update_status()

    def show_func2_page(self):
        # 进入功能2时，把当前原始图像传进去
        self.page_func2.set_image(self._pm_orig)
        self.stack.setCurrentWidget(self.page_func2)
        self._update_actions()
        self._update_status()

    def show_func3_page(self):
        self.page_func3.set_image(self._pm_orig)
        self.stack.setCurrentWidget(self.page_func3)
        self._update_actions()
        self._update_status()
        
    def show_func4_page(self): 
        self.page_func4.set_image(self._pm_orig)
        self.stack.setCurrentWidget(self.page_func4)
        self._update_actions()
        self._update_status()
        
    def show_func5_page(self):
        self.page_func5.set_image(self._pm_orig)
        self.stack.setCurrentWidget(self.page_func5)
        self._update_actions()
        self._update_status()


    # ---------------- 辅助 ----------------
    def _update_actions(self) -> None:
        on_image_page = self.stack.currentWidget() is self.page_image
        has_img = self._pixmap is not None

        self.act_fit.setEnabled(on_image_page and has_img)
        self.act_zoom_in.setEnabled(on_image_page and has_img and not self._fit_to_window)
        self.act_zoom_out.setEnabled(on_image_page and has_img and not self._fit_to_window)
        self.act_zoom_reset.setEnabled(on_image_page and has_img and not self._fit_to_window)

    def _update_status(self) -> None:
        # 不同页面可以展示不同的状态信息
        current = self.stack.currentWidget()
        if current is not self.page_image:
            if current is self.page_func1:
                self.statusBar().showMessage("功能1 页面")
            elif current is self.page_func2:
                self.statusBar().showMessage("功能2 页面")
            elif current is self.page_func3:
                self.statusBar().showMessage("功能3 页面")
            elif current is self.page_func4:
                self.statusBar().showMessage("功能4 页面")
            elif current is self.page_func5:
                self.statusBar().showMessage("功能5 页面")
            elif current is self.page_func6:
                self.statusBar().showMessage("功能6 页面")
            else:
                self.statusBar().showMessage("就绪")
            return

        # 图像浏览页
        if not self._pixmap:
            self.statusBar().showMessage("就绪")
            return
        pm_size = self._pixmap.size()
        percent = int(round(self._scale * 100))
        fit = "(适应窗口)" if self._fit_to_window else ""
        self.statusBar().showMessage(f"{pm_size.width()}×{pm_size.height()}  |  {percent}% {fit}")

    def resizeEvent(self, e):
        super().resizeEvent(e)
        if self._fit_to_window and self.stack.currentWidget() is self.page_image:
            self._fit_image_to_scrollarea()




