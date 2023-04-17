"""
    -- coding: utf-8 --
    @Author: codeking
    @Data : 2022/5/3 19:20
    @File : Signal.py
"""
from PySide6.QtCore import QObject, Signal


class MySignal(QObject):
    SetProgressBar = Signal(object, str, int)
    PopMeg = Signal(str)
    # SetConvertState=Signal(str)
    SetLabelValue = Signal(str, object)
    # 程序运行结束的弹窗
    PopWindow = Signal(str)


my_signal = MySignal()
