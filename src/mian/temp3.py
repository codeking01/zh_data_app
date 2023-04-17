# author: code_king
# time: 2023/4/18 20:53
# file: temp3.py

# 读取excel
import pandas as pd
from openpyxl.reader.excel import load_workbook

# 使用openpyxl读取
A=load_workbook(r"C:\Users\king\Desktop\0221结果 - 副本.xlsx")
a=pd.read_excel(r"C:\Users\king\Desktop\0221结果 - 副本.xlsx")
b=pd.read_excel(r"C:\Users\king\Desktop\0221结果 - 副本.xlsx")
c=pd.read_excel(r"C:\Users\king\Desktop\0221结果 - 副本.xlsx")
d=pd.read_excel(r"C:\Users\king\Desktop\0221结果 - 副本.xlsx")
print(A, a, b, c, d)
