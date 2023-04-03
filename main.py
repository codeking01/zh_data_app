"""
    -- coding: utf-8 --
    @Author: codeking
    @Data : 2022/5/1 17:49
    @File : main.py.py
"""

# 继承这个类  py可以多继承
import sys
from threading import Thread

from PySide6 import QtWidgets, QtCore
from PySide6.QtCore import Slot, Signal, QThread
from PySide6.QtWidgets import QFileDialog, QMessageBox, QLabel

from Signal import my_signal
from pubchemui import Ui_zh_data_app


class View(QtWidgets.QWidget, Ui_zh_data_app):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # 设置线程函数
        self.mysignal = Signal()
        #设置背景图
        # self.setStyleSheet(u"background-image: url(:/resource/a_bg);")
        # 设置标题
        # self.setWindowTitle("pubchem转化数据工具")
        self.csv_path = ''
        self.sdf_path = ''
        self.excel_path = ''
        # 需要转化的excel的地址
        self.ReadyToConvertExcel_path=''

        # 转化excel的状态
        self.ConvertToExcelFlag = False
        # 选择存储mol的路径
        self.Save2DMolPath = ''
        self.Save3DMolPath = ''
        # 选择图片文件地址
        self.SelectPicPath = ''
        self.SavePicPath = ''
        # txt生成的地址
        self.savetxtpath = r'D:/DATA/'
        # 槽函数自定义绑定
        self.bind()


    def bind(self):
        # 和生成数字按钮绑定
        # self.GenerateNumsButton.clicked.connect(self.GenerateNumbers)
        my_signal.SetGenState.connect(self.setstatelabel)
        my_signal.popMeg.connect(self.PopMeg)
        my_signal.SetlabelValue.connect(self.SetAlllabelValue)

    # 设置label的内容
    def SetAlllabelValue(self,value:str,components:object):
        self.components=components
        self.components.setText(value)

    # 定义弹窗的方法
    def PopMeg(self,value:str):
        QMessageBox.information(self, '提示', value)


    def  setstatelabel(self,value:str):
        self.GenNumsState.setText(value)


    @Slot()
    def on_load_Modeling_files_clicked(self):
        print('load_Modeling_files_clicked')

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
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    # 设置风格样式 Fusion,windows,windowsvista
    app.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    # window = QMainWindow() # 看自己的选择
    # window = QtWidgets.QWidget()  # 看自己的选择
    view = View()
    # view.setupUi(window)
    # view.setupUi(view)  这个可以放在构造函数中
    # window.show()
    view.show()
    sys.exit(app.exec())
