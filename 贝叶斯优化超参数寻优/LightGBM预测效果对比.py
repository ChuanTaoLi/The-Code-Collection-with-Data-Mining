'''原始数据'''
from matplotlib import rcParams
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, accuracy_score

rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False
train = pd.read_excel(r'D:\2024统计建模\美姑县_喜德县_训练集.xlsx')
x_train = train.iloc[:, 4:-1]
y_train = train.iloc[:, -1]

pa = {'n_estimators': 481, 'max_depth': 35, 'learning_rate': 0.1182827604856063, 'subsample': 0.581614058601332,
      'min_child_weight': 1, 'gamma': 0.923517035793613, 'colsample_bytree': 0.6129998201452649,
      'reg_alpha': 0.3623935688054989, 'reg_lambda': 0.4860325677688167}
'''调参前'''
test1 = pd.read_excel(r'D:\2024统计建模\美姑县_测试集.xlsx')
x_test1 = test1.iloc[:, 4:-1]
y_test1 = test1.iloc[:, -1]
lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
lgb_classifier.fit(x_train, y_train)
y_pred1 = lgb_classifier.predict(x_test1)
accuracy1 = accuracy_score(y_test1, y_pred1)
print(f'准确率: {accuracy1}')

# 计算 F1 分数
f1_score1 = f1_score(y_test1, y_pred1)

# 生成分类报告
classification_report1 = classification_report(y_test1, y_pred1)

print(f'F1分数: {f1_score1}')
print('分类报告:')
print(classification_report1)

'''调参后'''
lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1, **pa)
lgb_classifier.fit(x_train, y_train)
y_pred2 = lgb_classifier.predict(x_test1)
accuracy2 = accuracy_score(y_test2, y_pred2)
print(f'准确率: {accuracy2}')

# 计算 F1 分数
f1_score2 = f1_score(y_test2, y_pred2)

# 生成分类报告
classification_report2 = classification_report(y_test2, y_pred2)

print(f'F1分数: {f1_score2}')
print('分类报告:')
print(classification_report2)
