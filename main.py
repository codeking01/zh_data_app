"""
    -- coding: utf-8 --
    @Author: codeking
    @Data : 2022/5/1 17:49
    @File : main.py.py
"""
import sys
import time
from threading import Thread

import gevent as gevent
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
        self.work_threads = {"develop_work": True, "develop_finished": False, "predict_work": True,
                             "predict_finished": False}
        # 建模预测文件的名字
        self.develop_filename = ""
        self.predict_filename = ""

    def bind(self):
        # 和生成数字按钮绑定
        # self.GenerateNumsButton.clicked.connect(self.GenerateNumbers)
        my_signal.SetProgressBar.connect(self.set_progressBar)
        my_signal.PopMeg.connect(self.pop_meg)
        my_signal.SetLabelValue.connect(self.set_all_label_value)

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
        self.work_start(0, self.predict_filename, ["predict_finished", "predict_work"], self.predict_progressBar,
                        self.predict_button)

    def work_start(self, is_develop, filename, flag_list, progress_bar, select_button):
        if filename != "":
            try:
                self.work_task(is_develop, filename, flag_list, progress_bar, select_button)
            except Exception as e:
                QMessageBox.information(self, '错误', f"{e}")
        else:
            QMessageBox.information(self, '提示', "请先选择文件！")

    def work_task(self, is_develop, filename, flag_list, progress_bar, select_button):
        """
        :param filename:
        :param is_develop: 1的话就是建模，其他的预测
        :param flag_list:  例如 ["predict_finished", "predict_work"]
        :param progress_bar: 进度条
        :param select_button:  选择的是预测还是选择
        :return:
        """
        # 调用预测模型
        # 创建子进程
        progress_worker = Thread(target=self.work_progress,
                                 args=(flag_list[0], flag_list[1], progress_bar, select_button))
        progress_worker.start()
        # 禁用按钮
        select_button.setEnabled(False)

        def inner_work(is_develop, filename, finished_flag):
            """
            :param is_develop: 1的话就是建模，其他的预测
            :param filename:
            :param finished_flag:
            :return:
            """
            if is_develop == 1:
                ModelUse.develop_model(filename)
            else:
                ModelUse.predict_model(filename)
            self.work_threads[f"{finished_flag}"] = True

        # inner_work(is_develop, filename, flag_list[0])
        inner_work_thread = Thread(target=inner_work, args=(is_develop, filename, flag_list[0]))
        inner_work_thread.start()

    def work_progress(self, finished_flag, work_flag, progress_bar, select_button):
        # 设置一格一格动,一直循环
        i = 0
        # 让进度条一直循环
        while True:
            # 计算结束
            if self.work_threads[f"{finished_flag}"]:
                break
            if not self.work_threads[f"{work_flag}"]:
                my_signal.SetProgressBar.emit(progress_bar, "运行终止!!!", i)
                select_button.setEnabled(True)
                self.work_threads[f"{work_flag}"] = False
                return
            i += 1
            if i > 100:
                i = 0
            my_signal.SetProgressBar.emit(progress_bar, "正在运行...", i)
            time.sleep(0.1)
        my_signal.SetProgressBar.emit(progress_bar, "运行结束...", 100)
        select_button.setEnabled(True)
        self.work_threads[f"{work_flag}"] = False

    @Slot()  # 提前终止这个线程
    def on_predict_stop_button_clicked(self):
        if self.work_threads["predict_work"]:
            self.work_threads["predict_work"] = False
            self.predict_button.setEnabled(True)

    @Slot()
    def on_train_button_clicked(self):
        self.work_start(1, self.develop_filename, ["develop_finished", "develop_work"], self.train_progressBar,
                        self.train_button)

    @Slot()  # 提前终止这个线程
    def on_train_stop_button_clicked(self):
        if self.work_threads["develop_work"]:
            self.work_threads["develop_work"] = False
            self.train_button.setEnabled(True)

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

    # 去操作 modeling_file_path
    # # 生成数字的编辑栏
    # @Slot()
    # def on_NumsEdit_textChanged(self):
    #     self.GenNumsState.setText('还未生成')
    #
    # # 产生数字的按钮
    # @Slot()
    # def on_GenerateNumsButton_clicked(self):
    #     choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #     def workerThreadFunc():
    #         value='正在生成..'
    #         my_signal.SetGenState.emit(value)
    #         self.GenerateNumsButton.setEnabled(False)
    #         GenNums(self.savetxtpath, self.NumsEdit.text())
    #         self.GenerateNumsButton.setEnabled(True)
    #         # 使用信号量来释放
    #         value='生成成功!'
    #         my_signal.SetGenState.emit(value)
    #         # 用信号量取控制弹窗
    #         Msg='生成成功！'
    #         my_signal.popMeg.emit(Msg)
    #     if choice == QMessageBox.Yes:
    #         print('确定')
    #         print(self.NumsEdit.text())
    #         worker = Thread(target=workerThreadFunc)
    #         worker.start()
    #     elif choice == QMessageBox.No:
    #         print('取消')
    #
    # # 2022.5.24 新增模块 ¹新增选择原来的excel文件，取出带cas的全部excel文件 ²生成Txt文件
    # # 生成txt文件
    # @Slot()
    # def on_GenSpecialTxtBtn_clicked(self):
    #     if (self.ReadyToConvertExcel_path != ''):
    #         choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #         def GenTxtThreadFunc():
    #             # self.ConvertState.setText('正在转化，请等待....')
    #             # 禁用操作按钮
    #             self.GenSpecialTxtBtn.setEnabled(False)
    #             # 获取组件
    #             components=self.GenFinalTxtState
    #             value='正在生成...'
    #             my_signal.SetlabelValue.emit(value,components)
    #             # todo 生成Txt文件
    #             GenTxt(self.ReadyToConvertExcel_path)
    #             print("生成结束！")
    #             self.GenSpecialTxtBtn.setEnabled(True)
    #             value='生成成功！'
    #             my_signal.SetlabelValue.emit(value,components)
    #             # 用信号量取控制弹窗
    #             Msg='生成成功！'
    #             my_signal.popMeg.emit(Msg)
    #         if choice == QMessageBox.Yes:
    #             print('确定')
    #             worker = Thread(target=GenTxtThreadFunc)
    #             worker.start()
    #         elif choice == QMessageBox.No:
    #             print('取消')
    #     else:
    #         QMessageBox.warning(self, '警告', '请先选择带Final的Excel文件！')
    #
    # # 选择excel文件
    # @Slot()
    # def on_SelectSpecialExcelBtn_clicked(self):
    #     # 选择excel文件
    #     self.ReadyToConvertExcel_path, _ = QFileDialog.getOpenFileName(self, caption="选择你需要转化的的excel文件", dir=r'D:\DATA\ALL_Excel',
    #                                                      filter="选择python文件(*.xlsx)")
    #     print(f"ReadyToConvertExcel_path: {self.ReadyToConvertExcel_path}")
    # # 提取Excel（带全部cas的）文件
    # @Slot()
    # def on_GenFinalExcelButton_clicked(self):
    #     if (self.ReadyToConvertExcel_path != ''):
    #         choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #         def DealExcelThreadFunc():
    #             # self.ConvertState.setText('正在转化，请等待....')
    #             self.GenFinalExcelButton.setEnabled(False)
    #             # 获取组件
    #             components=self.GenFinalExcelState
    #             value='正在生成...'
    #             my_signal.SetlabelValue.emit(value,components)
    #             # 取出最后一列有内容的值存储到新的Excel里面
    #             ExtractFinalExcel(self.ReadyToConvertExcel_path)
    #             print("生成结束！")
    #             self.GenFinalExcelButton.setEnabled(True)
    #             value='生成成功！'
    #             my_signal.SetlabelValue.emit(value,components)
    #             # 用信号量取控制弹窗
    #             Msg='生成成功！'
    #             my_signal.popMeg.emit(Msg)
    #         if choice == QMessageBox.Yes:
    #             print('确定')
    #             worker = Thread(target=DealExcelThreadFunc)
    #             worker.start()
    #         elif choice == QMessageBox.No:
    #             print('取消')
    #     else:
    #         QMessageBox.warning(self, '警告', '请先选择需要提取Cas号的Excel文件！')
    #
    #
    # # 选择图片文件
    # @Slot()
    # def on_SelectPicButton_clicked(self):
    #     self.SelectPicPath = QFileDialog.getExistingDirectory(self, caption="选择你的图片文件", dir=r'D:\DATA\ALL_Pic')
    #     print(f"SelectPicPath: {self.SelectPicPath}")
    #
    # # 转化图片文件
    # @Slot()
    # def on_ConvertPicButton_clicked(self):
    #     if (self.SelectPicPath != '' and self.excel_path != ''):
    #         def ConvertPicThread():
    #             components=self.ConvertPicState
    #             value='正在转化..'
    #             my_signal.SetlabelValue.emit(value,components)
    #             # 禁用按钮
    #             self.ConvertPicButton.setEnabled(False)
    #             self.SavePicPath = self.SelectPicPath
    #             # print(f"SelectPicPath: {self.SavePicPath}")
    #             #  转化图片
    #             # excel地址
    #             read_path = self.excel_path
    #             # 图片地址
    #             filePath = self.SavePicPath
    #             rename_pic(read_path, filePath)
    #             # 恢复按钮
    #             self.ConvertPicButton.setEnabled(True)
    #             value='图片转化成功!'
    #             my_signal.SetlabelValue.emit(value,components)
    #             my_signal.popMeg.emit(value)
    #             print('图片转化结束！')
    #         choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #         if choice == QMessageBox.Yes:
    #             print('确定')
    #             worker=Thread(target=ConvertPicThread)
    #             worker.start()
    #         elif choice == QMessageBox.No:
    #             print('取消')
    #     else:
    #         QMessageBox.warning(self, '警告', '请先选择图片文件夹和Excel文件！')
    #
    # @Slot()
    # def on_Save2DMolButton_clicked(self):
    #     self.Save2DMolPath = QFileDialog.getExistingDirectory(self, caption="选择你的2Dmol存储文件", dir=r'D:\DATA\ALL_2DMol')
    #     print(f"Save2DMolPath: {self.Save2DMolPath}")
    #
    # @Slot()
    # def on_Save3DMolButton_clicked(self):
    #     self.Save3DMolPath = QFileDialog.getExistingDirectory(self, caption="选择你的3Dmol存储文件", dir=r'D:\DATA\ALL_3DMol')
    #     print(f"Save3DMolPath: {self.Save3DMolPath}")
    #
    # @Slot()
    # def on_SelectCsvButton_clicked(self):
    #     # 选择Csv文件
    #     # 返回值是元组， 不需要的参数用 _ 占位
    #     self.csv_path, _ = QFileDialog.getOpenFileName(self, caption="选择你的csv文件", dir=r'D:\DATA\ALL_Excel',
    #                                                    filter="选择csv文件(*.csv)")
    #     print(f"csv_path: {self.csv_path}")
    #
    # @Slot()
    # def on_SelectSdfButton_clicked(self):
    #     self.sdf_path, _ = QFileDialog.getOpenFileName(self, caption="选择你的sdf文件", dir=r'D:\DATA\ALL_SDF',
    #                                                    filter="sdf(*.sdf)")
    #     print(f"sdf_path: {self.sdf_path}")
    #
    # # 转化Excel按钮
    # @Slot()
    # def on_SaveExcelButton_clicked(self):
    #     if (self.csv_path != ''):
    #         choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #         def DealExcelThreadFunc():
    #             # self.ConvertState.setText('正在转化，请等待....')
    #             self.SaveExcelButton.setEnabled(False)
    #             components=self.ConvertState
    #             value='正在转化...'
    #             my_signal.SetlabelValue.emit(value,components)
    #             ConvertToExcel(self.csv_path)
    #             print("转化结束！")
    #             self.SaveExcelButton.setEnabled(True)
    #             self.ConvertToExcelFlag = True
    #             value='转化成功！'
    #             my_signal.SetlabelValue.emit(value,components)
    #             # 用信号量取控制弹窗
    #             Msg='生成成功！'
    #             my_signal.popMeg.emit(Msg)
    #         if choice == QMessageBox.Yes:
    #             print('确定')
    #             worker = Thread(target=DealExcelThreadFunc)
    #             worker.start()
    #         elif choice == QMessageBox.No:
    #             print('取消')
    #     else:
    #         QMessageBox.warning(self, '警告', '请先选择CSV文件！')
    #
    # # 将cas号添加到excel的按钮
    # @Slot()
    # def on_AddCasToExcelButton_clicked(self):
    #     if (self.ConvertToExcelFlag == True):
    #         def AddcasThread():
    #             components=self.AddCasState
    #             value='正在添加..'
    #             my_signal.SetlabelValue.emit(value,components)
    #             # 禁用按钮
    #             self.AddCasToExcelButton.setEnabled(False)
    #             ExcelPath = self.csv_path.replace('csv', 'xlsx')
    #             AddCasToExcel(ExcelPath)
    #             # 恢复按钮
    #             self.AddCasToExcelButton.setEnabled(True)
    #             print("添加成功！")
    #             self.ConvertToExcelFlag=False
    #             # 通过释放信号去弹窗
    #             value='添加成功！'
    #             my_signal.SetlabelValue.emit(value,components)
    #             Msg='添加成功!'
    #             my_signal.popMeg.emit(Msg)
    #         choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #         if choice == QMessageBox.Yes:
    #             print('确定')
    #             worker=Thread(target=AddcasThread)
    #             worker.start()
    #         elif choice == QMessageBox.No:
    #             print('取消')
    #     else:
    #         QMessageBox.warning(self, '警告', '请先转化Excel文件！')
    #
    # # 选择excel的按钮
    # @Slot()
    # def on_SelectExcelButton_clicked(self):
    #     # 选择excel文件
    #     self.excel_path, _ = QFileDialog.getOpenFileName(self, caption="选择你的excel文件", dir=r'D:\DATA\ALL_Excel',
    #                                                      filter="选择python文件(*.xlsx)")
    #     print(f"excel_path: {self.excel_path}")
    #
    # @Slot()
    # def on_SaveMol2DButton_clicked(self):
    #     if (self.sdf_path != '' and self.excel_path != '' and self.Save2DMolPath != ''):
    #         def ConvertThread():
    #             self.SaveMol2DButton.setEnabled(False)
    #             components=self.Convert2DState
    #             value='正在转化...'
    #             my_signal.SetlabelValue.emit(value,components)
    #             temp_data = GetCaslist(excel_path=r'{excel_path_f}'.format(excel_path_f=self.excel_path))
    #             cas_list = temp_data[0]
    #             cid_list = temp_data[1]
    #             #  将数据存储到cid_dic字典里面
    #             cid_dic = get_cid_dic(cid_list, cas_list)
    #             # 读取sdf的数据全部存储为列表了 写入sdf的路径
    #             sdf_file = Read_SDF(read_path=str(self.sdf_path))
    #             #  转化sdf到2Dmol文件
    #             Save_Mol_Path = self.Save2DMolPath
    #             ConvertSdfToMol(cas_list, cid_dic, sdf_file, Save_Mol_Path)
    #             # 恢复按钮
    #             self.SaveMol2DButton.setEnabled(True)
    #             value='2D转化成功!!'
    #             my_signal.SetlabelValue.emit(value,components)
    #             my_signal.popMeg.emit(value)
    #             print('转化结束！')
    #         choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #         if choice == QMessageBox.Yes:
    #             print('确定')
    #             worker=Thread(target=ConvertThread)
    #             worker.start()
    #         elif choice == QMessageBox.No:
    #             print('取消')
    #     else:
    #         QMessageBox.warning(self, '警告', '请先选择SDF或者Excel文件！也或许没有选择存储路径！')
    #
    # @Slot()
    # def on_SaveMol3DButton_clicked(self):
    #     if (self.sdf_path != '' and self.excel_path != '' and self.Save3DMolPath != ''):
    #         def Convert3DThread():
    #             self.SaveMol3DButton.setEnabled(False)
    #             components=self.Convert3DState
    #             value='正在转化...'
    #             my_signal.SetlabelValue.emit(value,components)
    #             temp_data = GetCaslist(excel_path=r'{excel_path_f}'.format(excel_path_f=self.excel_path))
    #             cas_list = temp_data[0]
    #             cid_list = temp_data[1]
    #             #  将数据存储到cid_dic字典里面
    #             cid_dic = get_cid_dic(cid_list, cas_list)
    #             # 读取sdf的数据全部存储为列表了 写入sdf的路径
    #             sdf_file = Read_SDF(read_path=str(self.sdf_path))
    #             Save_Mol_Path = self.Save3DMolPath
    #             ConvertSdfToMol(cas_list, cid_dic, sdf_file, Save_Mol_Path)
    #             # 恢复按钮
    #             self.SaveMol3DButton.setEnabled(True)
    #             value='3D转化成功!!'
    #             my_signal.SetlabelValue.emit(value,components)
    #             my_signal.popMeg.emit(value)
    #             print("转化结束！")
    #
    #         choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #         if choice == QMessageBox.Yes:
    #             print('确定')
    #             worker=Thread(target=Convert3DThread)
    #             worker.start()
    #
    #         elif choice == QMessageBox.No:
    #             print('取消')
    #     else:
    #         QMessageBox.warning(self, '警告', '请先选择SDF或者Excel文件！也或许没有选择存储路径！')
    #
    # def bind(self):
    #     # 和生成数字按钮绑定
    #     # self.GenerateNumsButton.clicked.connect(self.GenerateNumbers)
    #     my_signal.SetGenState.connect(self.setstatelabel)
    #     my_signal.popMeg.connect(self.PopMeg)
    #     my_signal.SetlabelValue.connect(self.SetAlllabelValue)
    #
    # # 设置label的内容
    # def SetAlllabelValue(self,value:str,components:object):
    #     self.components=components
    #     self.components.setText(value)
    #
    # # 定义弹窗的方法
    # def PopMeg(self,value:str):
    #     QMessageBox.information(self, '提示', value)
    #
    #
    # def  setstatelabel(self,value:str):
    #     self.GenNumsState.setText(value)
    #
    # def GenerateNumbers(self):
    #     def workerThreadFunc():
    #         value='正在生成..'
    #         # 信号释放
    #         my_signal.SetGenState.emit(value)
    #         GenNums(self.savetxtpath, self.NumsEdit.text())
    #         # 使用信号量来释放
    #         value='生成成功!'
    #         my_signal.SetGenState.emit(value)
    #     choice = QMessageBox.question(self, '确认', '您确认要执行该操作?')
    #     if choice == QMessageBox.Yes:
    #         print('确定')
    #         print(self.NumsEdit.text())
    #         worker = Thread(target=workerThreadFunc)
    #         worker.start()
    #         # self.GenNumsState.setText('生成成功！')
    #     elif choice == QMessageBox.No:
    #         print('取消')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 设置风格样式 Fusion,windows,windowsvista
    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    view = View()
    view.show()
    sys.exit(app.exec())

