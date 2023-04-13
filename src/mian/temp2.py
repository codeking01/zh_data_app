import joblib
from sklearn.ensemble import RandomForestClassifier

def save_model_dic():
    # 创建一个字典，用于存储所有模型
    models = {}

    # 循环加载多个 pkl 文件
    for i in range(21, 23):
        # 创建一个 pkl 文件名
        model_name = i
        # 加载 pkl 文件中的模型
        # 将模型添加到字典中
        models[model_name] = RandomForestClassifier()
    # 将字典中的所有模型保存到一个 joblib 文件中
    joblib.dump(models, 'models.joblib')

# 从文件中加载所有模型
models = joblib.load('./all_models/models.joblib')
# "21.pkl" in models.keys()
print(models)

# new_d = ('Year', 'Month', 'Day')
# dict1 = dict.fromkeys(new_d)
# print("新的字典 : %s" % str(dict1))