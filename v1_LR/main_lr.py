# 线上分数：0.6

# 安装相关依赖库 如果是windows系统，cmd命令框中输入pip安装，参考上述环境配置
# !pip install sklearn
# !pip install pandas

# ---------------------------------------------------
# 导入库
import pandas as pd
import  numpy as np
import os
import random
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
set_seed(2022)
# ----------------数据探索----------------
# 只使用目标域用户行为数据
train_ads = pd.read_csv('./train/train_data_ads.csv',
                        usecols=['log_id', 'label', 'user_id', 'age', 'gender', 'residence', 'device_name',
                                 'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd'])

test_ads = pd.read_csv('./test/test_data_ads.csv',
                       usecols=['log_id', 'user_id', 'age', 'gender', 'residence', 'device_name',
                                'device_size', 'net_type', 'task_id', 'adv_id', 'creat_type_cd'])

# ----------------数据集采样----------------
train_ads = pd.concat([
    train_ads[train_ads['label'] == 0].sample(70000),
    train_ads[train_ads['label'] == 1].sample(10000),
])

# ----------------模型训练----------------
# 加载训练逻辑回归模型
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(
    train_ads.drop(['log_id', 'label', 'user_id'], axis=1),
    train_ads['label']
)

# ----------------结果输出----------------
# 模型预测与生成结果文件
test_ads['pctr'] = clf.predict_proba(
    test_ads.drop(['log_id', 'user_id'], axis=1),
)[:, 1]
test_ads[['log_id', 'pctr']].to_csv('submission.csv', index=None)