# author: code_king
# time: 2023/4/17 14:22
# file: ApiModel.py
from src.mian.dealModelMain import OperateModel


class ModelUse:
    def __init__(self):
        pass

    @staticmethod
    def develop_model(select_path,train_numbers):
        """
        :param train_numbers: 训练的次数
        :param select_path: 根据选择的excel去建模
        :return:
        """
        OperateModel.select_develop_model(excel_path=f"{select_path}",train_numbers=train_numbers)

    @staticmethod
    def predict_model(select_path):
        """
        :param select_path:   根据选择的excel去预测
        :return:
        """
        OperateModel.select_predict_model(select_path)


