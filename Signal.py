"""
    -- coding: utf-8 --
    @Author: codeking
    @Data : 2022/5/3 19:20
    @File : Signal.py
"""
from PySide6.QtCore import QObject, Signal


class MySignal(QObject):
    SetGenState=Signal(str)
    popMeg=Signal(str)
    # SetConvertState=Signal(str)
    SetlabelValue=Signal(str,object)

my_signal=MySignal()