import copy

import pandas as pd
import numpy as np

# 生成包含中文字符的 NumPy 数组
arr = np.array([[3.0, '1', 'a', '这是一个包含中文字符的字符串'],
                [3.0, '2', 'a', '这也是一个包含中文字符的字符串'],
                [3.0, '3', 'a', np.nan],
                [6.0, '267', 'd', '这是另一个包含中文字符的字符串'],
                [6.0, '268', 'd', '这还是一个包含中文字符的字符串'],
                [6.0, '269', 'd', '这也是另一个包含中文字符的字符串']])

# 将 NumPy 数组转换为 Pandas DataFrame
df = pd.DataFrame(arr)
# print("arr",np.array(arr))

import pandas as pd


def check_contains_letters_or_chinese(arr):
    def contains_letters_or_chinese(s):
        for c in s:
            if (u'\u4e00' <= c <= u'\u9fff') or c.isalpha():
                return True
        return False

    df = pd.DataFrame(copy.deepcopy(arr))
    mask = df.applymap(lambda x: contains_letters_or_chinese(str(x)))
    mask = mask.any(axis=0)
    return mask.tolist()


# 输出结果
print(check_contains_letters_or_chinese(arr))

import pandas as pd
import numpy as np

# 构造示例数据
data = np.array([
    [3.0, '1', 'a', np.nan, 0.3896, -0.42138, 2.33126],
    [3.0, '2', 'a', np.nan, 0.3896, -0.42138, 2.33126],
    [3.0, '3', 'a', np.nan, 0.3896, -0.42138, 2.33126],
    [np.nan, '4', 'b', np.nan, 0.2036, np.nan, 2.33126],
    [np.nan, '5', 'b', np.nan, 0.2036, np.nan, 2.33126],
    [np.nan, '6', 'b', np.nan, 0.2036, np.nan, 2.33126],
    [np.nan, '7', 'b', np.nan, 0.2036, np.nan, 2.33126],
    [1.0, '8', 'c', 4.0, 0.2036, -0.5424, np.nan],
    [2.0, '9', 'c', 5.0, 0.2036, -0.5424, np.nan],
    [3.0, '10', 'c', 6.0, 0.2036, -0.5424, np.nan],
])

# 转换为 DataFrame
df = pd.DataFrame(data)

# 计算每列的缺失值比例
null_pct = df.isnull().sum() / len(df)
print("null_pct",null_pct)
# 筛选缺失值比例大于30%的列
high_null_cols = null_pct[null_pct > 0.3].index.tolist()

# 输出结果
print("超过30%缺失值的列：", high_null_cols)
print("数据：")
print(df)
