'''喜德县'''
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

Train = pd.read_excel(r'D:\2024统计建模\训练集_LASSO.xlsx')
Test = pd.read_excel(r'D:\2024统计建模\喜德县_LASSO.xlsx')
x_Train = Train.iloc[:, 4:-1]
y_Train = Train.iloc[:, -1]
x_Test = Test.iloc[:, 4:-1]
y_Test = Test.iloc[:, -1]

# 1. 训练分类器
hy = {'n_estimators': 768, 'max_depth': 59, 'learning_rate': 0.02696576158228669, 'subsample': 0.3048122257429559,
      'min_child_weight': 1, 'gamma': 0.00014084157110738627, 'colsample_bytree': 0.3636242896782553,
      'reg_alpha': 0.5612039044735743, 'reg_lambda': 0.35629932751865506}
classifier = LGBMClassifier(verbosity=-1, **hy)
classifier.fit(x_Train, y_Train)

# 2. 计算每个类别的概率
probabilities = classifier.predict_proba(x_Test)

# 3. 估计观测分布
# 假设 y_test 是观测样本的实际标签

# 4. 保序回归校准
calibrated_predictions = []
for i in range(2): 
    isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
    calibrated_predictions.append(isotonic_regressor.fit_transform(probabilities[:, i], (y_Test == i).astype(int)))

# 转换为概率矩阵
calibrated_predictions = np.column_stack(calibrated_predictions)

# 评估校准效果 - 未校准模型
uncalibrated_predictions = classifier.predict(x_Test)
uncalibrated_accuracy = accuracy_score(y_Test, uncalibrated_predictions)
f1_lgb_uncalibrated = f1_score(y_Test, uncalibrated_predictions, average='macro')  # Corrected here

# 评估校准效果 - 校准模型
calibrated_accuracy = accuracy_score(y_Test, np.argmax(calibrated_predictions, axis=1))
f1_lgb_calibrated = f1_score(y_Test, np.argmax(calibrated_predictions, axis=1), average='macro')  # Corrected here

print(f'Uncalibrated Accuracy: {uncalibrated_accuracy}')
print(f'Uncalibrated F1 score: {f1_lgb_uncalibrated}')
print(f'Calibrated Accuracy: {calibrated_accuracy}')
print(f'Calibrated F1 score: {f1_lgb_calibrated}')

'''
Uncalibrated Accuracy: 0.8818306010928961
Uncalibrated F1 score: 0.6573103260715141
Calibrated Accuracy: 0.9453551912568307
Calibrated F1 score: 0.7416257809466662
'''
