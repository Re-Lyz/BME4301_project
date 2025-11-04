"""
image_viewer_pyside6.py — 一个最小但好用的图片读取/查看 GUI 框架
依赖：PySide6  (pip install PySide6)
运行：python image_viewer_pyside6.py

功能要点：
- 打开图片（菜单/工具栏/快捷键 Ctrl+O）
- 支持常见格式（由 Qt 插件决定）
- 滚轮+Ctrl 缩放；菜单缩放（Ctrl+= / Ctrl+- / Ctrl+0）
- 适应窗口（Ctrl+F）/ 原始大小切换
- 拖拽图片文件到窗口直接打开
- 状态栏显示分辨率 & 缩放比例
- 滚动查看大图（QScrollArea）

可作为你项目中图像标注/处理工具的骨架，后续可在此基础上新增：
- Recent Files、批量浏览、图像处理管线（OpenCV/Pillow）、ROI/绘制层、快捷键映射等。

直方图与均衡化
卷积操作
二值化形态学基础
二值形态学高级
灰度形态学基础
灰度形态学高级

"""
from __future__ import annotations
from PySide6.QtWidgets import QApplication
import sys
from imageViewer import ImageViewer
 

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Image Viewer")
    app.setOrganizationName("Example")

    viewer = ImageViewer()
    viewer.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
