"""
    -- coding: utf-8 --
    @Author: codeking
    @Data : 2022/5/1 17:49
    @File : main.py.py
"""
import ctypes
import sys
import time
from threading import Thread
from PySide6 import QtWidgets
from PySide6.QtCore import Slot, Signal
from PySide6.QtWidgets import QFileDialog, QMessageBox, QLabel, QApplication, QWidget

from Entry.ApiModel import ModelUse
from Signal import my_signal
from zh_appui import Ui_zh_data_app


class View(QWidget, Ui_zh_data_app):
    def closeEvent(self, event):
        reply = QMessageBox.question(self, '退出提示', '您确定要退出吗？', QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.mySignal = Signal()
        # 槽函数自定义绑定
        self.bind()
        my_signal.SetProgressBar.emit(self.train_progressBar, "程序未运行...", 0)
        my_signal.SetProgressBar.emit(self.predict_progressBar, "程序未运行...", 0)
        # 设置子线程
        self.train_worker = None
        self.predict_worker = None
        # 设置线程状态
        self.work_threads = {"develop_finished": False, "predict_finished": False}
        # 建模预测文件的名字
        self.develop_filename = ""
        self.predict_filename = ""
        self.control_button_stats(self.train_stop_button, False)
        self.control_button_stats(self.predict_stop_button, False)

    def control_button_stats(self, button, flag: bool):
        button.setEnabled(flag)

    def bind(self):
        # 和生成数字按钮绑定
        # self.GenerateNumsButton.clicked.connect(self.GenerateNumbers)
        my_signal.SetProgressBar.connect(self.set_progressBar)
        my_signal.PopMeg.connect(self.pop_meg)
        my_signal.SetLabelValue.connect(self.set_all_label_value)
        my_signal.PopWindow.connect(self.pop_window)

    def pop_window(self, value: str):
        QMessageBox.information(self, "成功！", f"{value}")

    # 设置label的内容
    def set_all_label_value(self, value: str, components: QLabel):
        self.components = components
        self.components.setText(value)

    # 定义弹窗的方法
    def pop_meg(self, value: str):
        QMessageBox.information(self, '提示', value)

    def set_progressBar(self, progress_obj, state: str, value: int):
        # 这个是显示进度条的方法
        progress_obj.setFormat(f'{state}')
        progress_obj.setValue(int(value))

    @Slot()
    def on_predict_button_clicked(self):
        self.work_start(0, self.predict_filename, ["predict_finished"], self.predict_progressBar, self.predict_button,
                        self.predict_stop_button, "predict")

    def work_start(self, is_develop, filename, flag_list, progress_bar, select_button, stop_button, work_thread,train_numbers=1):
        if filename != "":
            try:
                self.work_task(is_develop, filename, flag_list, progress_bar, select_button, stop_button, work_thread,train_numbers=train_numbers)
            except Exception as e:
                QMessageBox.information(self, '错误', f"{e}")
        else:
            QMessageBox.information(self, '提示', "请先选择文件！")

    def work_task(self, is_develop, filename, flag_list, progress_bar, select_button, stop_button, work_thread,train_numbers):
        """
        :param filename:
        :param is_develop: 1的话就是建模，其他的预测
        :param flag_list:  例如 ["predict_finished", "predict_work"]
        :param progress_bar: 进度条
        :param select_button:  选择的是预测还是选择
        :return:
        """

        def inner_work(is_develop, filename, finished_flag,train_numbers):
            """
            :param is_develop: 1的话就是建模，其他的预测
            :param filename:
            :param finished_flag:
            :return:
            """
            if is_develop == 1:
                ModelUse.develop_model(filename,train_numbers)
            else:
                ModelUse.predict_model(filename)
            self.work_threads[f"{finished_flag}"] = True
            self.control_button_stats(stop_button, False)

        self.work_threads[f"{work_thread}"] = Thread(target=inner_work, args=(is_develop, filename, flag_list[0],train_numbers))
        self.work_threads[f"{work_thread}"].start()

        # 创建子进程
        progress_worker = Thread(target=self.work_progress,
                                 args=(flag_list[0], progress_bar, select_button, work_thread))
        progress_worker.start()
        # 禁用按钮
        select_button.setEnabled(False)
        self.control_button_stats(stop_button, True)

    def kill_thread(self, inner_work_thread):
        if not inner_work_thread.is_alive():
            return
        # 终止线程
        async_exc = ctypes.py_object(SystemExit)
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(inner_work_thread.ident), async_exc)

    def work_progress(self, finished_flag, progress_bar, select_button, work_thread):
        # 设置一格一格动,一直循环
        i = 0
        # 让进度条一直循环
        while True:
            # 计算结束
            if self.work_threads[f"{finished_flag}"]:
                break
            i += 1
            if i > 100:
                i = 0
            my_signal.SetProgressBar.emit(progress_bar, "正在运行...", i)
            time.sleep(0.1)
        self.kill_thread(self.work_threads[f"{work_thread}"])
        my_signal.SetProgressBar.emit(progress_bar, "运行结束...", 100)
        select_button.setEnabled(True)
        self.work_threads[f"{finished_flag}"] = False
        my_signal.PopWindow.emit("程序运行结束！")

    # 进度条归位
    def back_zero(self, left_button, right_button, flag_list, progress_bar):
        """
        :param left_button:
        :param right_button:
        :param flag_list:  类似["develop_finished", "develop_work"]
        :param progress_bar:
        :return:
        """
        left_button.setEnabled(True)
        self.work_threads[f"{flag_list[0]}"] = True
        # self.work_threads[f"{flag_list[1]}"] = False
        # 进度条归0
        # my_signal.SetProgressBar.emit(progress_bar, "程序终止！", 0)
        right_button.setEnabled(False)

    @Slot()  # 提前终止这个线程
    def on_predict_stop_button_clicked(self):
        if not self.work_threads["predict_finished"]:
            self.back_zero(self.predict_button, self.predict_stop_button, ["predict_finished"],
                           self.predict_progressBar)

    @Slot()
    def on_train_button_clicked(self):
        self.work_start(1, self.develop_filename, ["develop_finished"], self.train_progressBar, self.train_button,
                        self.train_stop_button, "develop",self.train_number.value())

    @Slot()  # 提前终止这个线程
    def on_train_stop_button_clicked(self):
        if not self.work_threads["develop_finished"]:
            self.back_zero(self.train_button, self.train_stop_button, ["develop_finished"], self.train_progressBar)

    @Slot()  # 选择建模文件
    def on_load_modeling_files_clicked(self):
        # 选择excel文件
        file_path = QFileDialog.getOpenFileName(self, '选择建模文件', '',
                                                '选择Excel文件 (*.xlsx *.xls);;所有文件类型 (*)')
        if file_path[0] != '':
            modeling_path = file_path[0].split('/')[-1]
            my_signal.SetLabelValue.emit(modeling_path, self.modeling_file_path)
            self.develop_filename = file_path[0]

    # 选择其他训练模型
    @Slot()
    def on_load_modeling_exist_files_clicked(self):
        choice = QMessageBox.question(self, '确认', '当前已有模型，您确认要执行该操作？请慎重！')
        if choice == QMessageBox.Yes:
            file_path = QFileDialog.getOpenFileName(self, '选择其他训练模型', '', '(*)')
            if file_path[0] != '':
                modeling_path = file_path[0].split('/')[-1]
                # 去操作 modeling_file_path
                my_signal.SetLabelValue.emit(modeling_path, self.modeling_exist_file_path)
        else:
            pass

    # 选择预测文件
    @Slot()
    def on_load_predict_files_clicked(self):
        file_path = QFileDialog.getOpenFileName(self, '选择预测文件', '',
                                                '选择Excel文件 (*.xlsx *.xls);;所有文件类型 (*)')
        if file_path[0] != '':
            modeling_path = file_path[0].split('/')[-1]
            my_signal.SetLabelValue.emit(modeling_path, self.predict_file_path)
            self.predict_filename = file_path[0]


def mian():
    app = QApplication(sys.argv)
    # 设置风格样式 Fusion,windows,windowsvista
    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    view = View()
    view.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    mian()
