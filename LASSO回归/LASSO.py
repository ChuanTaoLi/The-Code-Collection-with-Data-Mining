import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib import rcParams
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LassoCV
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 设置matplotlib绘图的中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# 加载数据集
train = pd.read_excel(r'D:\2024统计建模\美姑县_喜德县_训练集.xlsx')
test_meigu = pd.read_excel(r'D:\2024统计建模\美姑县_测试集.xlsx')
test_xide = pd.read_excel(r'D:\2024统计建模\喜德县_测试集.xlsx')
input_data = pd.read_excel(r'D:\2024统计建模\编码_总数据集.xlsx')
input_train, input_test = train_test_split(input_data, test_size=0.2, random_state=42)
interation = 2000  # 迭代次数

# 定义LASSO回归函数
def lasso_regression(train, test, alpha):
    lassoreg = Lasso(alpha=alpha, max_iter=interation, fit_intercept=True)
    lassoreg.fit(train.drop(columns=['风险是否已消除']), train['风险是否已消除'])
    feature_count = np.sum(lassoreg.coef_ != 0)  # 计算该alpha下筛选出的特征数量
    y_pred = lassoreg.predict(test.drop(columns=['风险是否已消除']))
    mse = mean_squared_error(test['风险是否已消除'], y_pred)
    ret = [alpha, mse]
    ret.append(feature_count)  # 非零系数的数量
    ret.extend(lassoreg.coef_)
    return ret

# 记录不同alpha下的准确率以及各特征的回归系数
alpha_lasso = np.linspace(0.001, 0.5, interation)
col = ["alpha", "mse", "feature_count"] + list(input_train.iloc[:, 4:-1])
ind = ["alpha_%.4g" % alpha_lasso[i] for i in range(0, len(alpha_lasso))]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

# 调用函数，将函数返回的结果录入
for i in range(len(alpha_lasso)):
    coef_matrix_lasso.iloc[i] = lasso_regression(input_train.iloc[:, 4:], input_test.iloc[:, 4:], alpha_lasso[i])
coef_matrix_lasso.to_excel(r'LASSO系数矩阵.xlsx', index=True)

# 绘制岭迹图
plt.figure(figsize=(14, 6.8))
for i in np.arange(len(list(input_train.iloc[:, 4:-1]))):
    plt.plot(coef_matrix_lasso["alpha"],
             coef_matrix_lasso[list(input_train.iloc[:, 4:-1])[i]],
             color=plt.cm.Set1(i / len(list(input_train.iloc[:, 4:-1]))),
             label=list(input_train.iloc[:, 4:-1])[i])
    plt.legend(loc="upper right", ncol=5, prop={'size': 7})
    plt.xlabel("正则化系数", fontsize=14)
    plt.ylabel("回归系数", fontsize=14)
plt.savefig(r'LASSO回归岭迹图', dpi=600)
plt.show()

# 十折交叉验证选择最佳正则化系数
alpha_choose = np.linspace(0.001, 0.5, interation)
lasso_cv = LassoCV(alphas=alpha_choose, cv=10, max_iter=interation)
lasso_cv.fit(input_data.iloc[:, 4:-1], input_data.iloc[:, -1])
lasso_best_alpha = lasso_cv.alpha_
print(lasso_best_alpha)

# 计算VIF
def calculate_vif(df):
    vif = pd.DataFrame()
    vif['特征'] = df.columns
    vif['方差膨胀因子'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
    return vif

# 建立Lasso回归模型
lasso_model = Lasso(alpha=lasso_best_alpha, fit_intercept=True,
                    max_iter=interation, random_state=1412, selection='cyclic')
lasso_model.fit(input_data.iloc[:, 4:-1], input_data.iloc[:, -1])
selected_features = input_data.iloc[:, 4:-1].columns[lasso_model.coef_ != 0]

# 计算Lasso前后的VIF
vif_before = calculate_vif(input_data.iloc[:, 4:-1])
print("原始数据的方差膨胀因子:\n", vif_before)
input_data_selected = input_data.iloc[:, 4:-1][selected_features]
vif_after = calculate_vif(input_data_selected)
print("筛选特征后的方差膨胀因子:\n", vif_after)

# 保存结果
vif_before.to_excel(r'初始方差膨胀因子.xlsx', index=False)
vif_after.to_excel(r'LASSO后方差膨胀因子.xlsx', index=False)

# 保存选定特征的训练集和测试集
train_selected = train[['县', '姓名', '与户主关系', '户编号'] + selected_features.tolist() + ['风险是否已消除']]
train_selected.to_excel(r'训练集_LASSO.xlsx', index=False)

test_meigu_selected = test_meigu[
    ['县', '姓名', '与户主关系', '户编号'] + selected_features.tolist() + ['风险是否已消除']]
test_meigu_selected.to_excel(r'美姑县_LASSO.xlsx', index=False)

test_xide_selected = test_xide[
    ['县', '姓名', '与户主关系', '户编号'] + selected_features.tolist() + ['风险是否已消除']]
test_xide_selected.to_excel(r'喜德县_LASSO.xlsx', index=False)

