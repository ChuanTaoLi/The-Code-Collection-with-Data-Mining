from scipy import sparse
from sklearn.utils import check_X_y
from validation import check_target_type, check_ratio
import numpy as np


# 求给定矩阵中的样本点到中心点的马氏距离
def mashi_distance(x_array):
    # 给定矩阵的样本中心点
    x_mean = np.mean(x_array, axis=0)
    # 给定矩阵的协方差矩阵
    S = np.cov(x_array.T)
    ma_distances = []
    if np.linalg.det(S) != 0:
        for x_item in x_array:
            SI = np.linalg.inv(S)
            delta = x_item - x_mean
            # 给定矩阵中的相应样本点到中心点的马氏距离
            distance = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            # 这里是项目要求得到马氏距离的平方
            ma_distances.append(distance ** 2)
    else:
        print("矩阵行列式为0")
    return ma_distances


class MAHAKIL:
    def __init__(self, ratio='auto', sampling_type="over-sampling"):
        self.ratio = ratio
        self.sampling_type = sampling_type

    # 产出新样本前对所给数组x_old，y_old进行检测，看其长度，类型是否一致
    def fit(self, x_old, y_old):
        y_old = check_target_type(y_old)
        x_check, y_check = check_X_y(x_old, y_old, accept_sparse=['csr', 'csc'])
        # ratio_xy为少数类要产生新样本的数目
        self.ratio_xy = check_ratio(self.ratio, y_check, self.sampling_type)
        return self

    def sample(self, x_old, y_old):
        X_resampled = x_old.copy()
        y_resampled = y_old.copy()
        for class_sample, n_samples in self.ratio_xy.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y_old == class_sample)
            X_class = x_old[target_class_indices]
            X_new, y_new = self.make_samples(X_class, class_sample, n_samples)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))
        return X_resampled, y_resampled

    # MAHAKIL方法具体产生新样本的方式
    def make_samples(self, X_class, class_sample, n_samples):
        x_row = np.shape(X_class)[0]
        # 得到X_class数组的相关马氏距离
        mashi_distances = mashi_distance(X_class)
        # 将X_class数组里的每个样本和其马氏距离保存在mashi_zip数组里
        mashi_zip = zip(X_class, mashi_distances)
        # 将mashi_zip数组按其保存的马氏距离从大到小排序
        sample_arr = sorted(mashi_zip, key=lambda x: x[1], reverse=True)
        Nmid = int(x_row / 2)
        nb1 = []
        nb2 = []
        for i in range(Nmid):
            nb1.append(sample_arr[i][0])
        Nbin1 = list(zip(nb1, range(Nmid)))
        for j in range(Nmid, x_row):
            nb2.append(sample_arr[j - Nmid - 1][0])
        Nbin2 = list(zip(nb2, range(Nmid)))
        x_new_list = []
        xre_list = []
        nmid = 0
        for i in range(Nmid):
            x_reshape = (np.array(Nbin1[i][0]) + np.array(Nbin2[i][0])) * 0.5
            xre_list.append(x_reshape)
            nmid += 1
            if (len(x_new_list) + len(xre_list)) >= n_samples:
                break
        x_new_list.extend(list(zip(xre_list, range(nmid))))
        if len(x_new_list) >= n_samples:
            y_new = np.array([class_sample] * len(x_new_list))
            return xre_list, y_new
        x_new_copyl = x_new_list.copy()
        x_new_copyr = x_new_list.copy()
        nmid = 0
        # 将第一代祖先不断与后面的子孙样本点结合产生新样本，知道满足数量n_sampes
        while len(x_new_list) < n_samples:
            xleft_list = []
            xright_list = []
            for i in range(Nmid):
                x_reshape = (np.array(Nbin1[i][0]) + np.array(x_new_copyl[i][0])) * 0.5
                xleft_list.append(x_reshape)
                nmid += 1
                if (len(x_new_list) + len(xleft_list)) >= n_samples:
                    break
            x_new_copyl = list(zip(xleft_list, range(nmid)))
            x_new_list.extend(x_new_copyl)
            if (len(x_new_list) + len(xleft_list)) < n_samples:
                nmid = 0
                for j in range(Nmid):
                    x_reshape = (np.array(Nbin2[j][0]) + np.array(x_new_copyr[j][0])) * 0.5
                    xright_list.append(x_reshape)
                    nmid += 1
                    if (len(x_new_list) + len(xright_list)) >= n_samples:
                        break
                x_new_copyr = list(zip(xleft_list, range(nmid)))
                x_new_list.extend(x_new_copyr)
        y_new = np.array([class_sample] * len(x_new_list))
        x_new = []
        for item in range(len(x_new_list)):
            x_new.append(x_new_list[item][0])
        return np.array(x_new), y_new

    # 类似于主函数，入口
    def fit_sample(self, x_old, y_old):
        return self.fit(x_old, y_old).sample(x_old, y_old)


if __name__ == '__main__':
    import pandas as pd

    train = pd.read_excel(r'D:\2024统计建模\训练集_LASSO.xlsx')
    x_train = train.iloc[:, 4:-1]
    y_train = train.iloc[:, -1]
    mahakil = MAHAKIL()

    # 使用 fit_sample 方法对训练数据进行过采样
    x_resampled, y_resampled = mahakil.fit_sample(x_train.values, y_train.values)
    x_resampled = pd.DataFrame(x_resampled)
    y_resampled = pd.DataFrame(y_resampled)
    x_resampled.to_excel(r'x_resampled.xlsx', index=False)
    y_resampled.to_excel(r'y_resampled.xlsx', index=False)
    # '''LightGBM预测对比'''
    # import lightgbm as lgb
    # from sklearn.metrics import f1_score, accuracy_score
    #
    # test = pd.read_excel(r'D:\0统计建模\代码\美姑县_LASSO.xlsx')
    # x_test = test.iloc[:, 4:-1]
    # y_test = test.iloc[:, -1]
    #
    # '''MAHAKIL前'''
    # lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
    # lgb_classifier.fit(x_train, y_train)
    # y_pred = lgb_classifier.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'MAHAKIL前的准确率: {accuracy}')
    # f1_score_before = f1_score(y_test, y_pred)
    # print(f'MAHAKIL前的F1分数: {f1_score_before}')
    #
    # '''MAHAKIL后'''
    # lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
    # lgb_classifier.fit(x_resampled, y_resampled)
    # y_pred1 = lgb_classifier.predict(x_test)
    # accuracy1 = accuracy_score(y_test, y_pred1)
    # print(f'MAHAKIL后的准确率: {accuracy1}')
    # f1_score_after = f1_score(y_test, y_pred1)
    # print(f'MAHAKIL后的F1分数: {f1_score_after}')
    #
    # from imblearn.over_sampling import ADASYN
    #
    # adasyn = ADASYN(sampling_strategy='auto', random_state=42)
    # x_train_resampled1, y_train_resampled1 = adasyn.fit_resample(x_train, y_train)
    #
    # '''ADASYN后'''
    # lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
    # lgb_classifier.fit(x_train_resampled1, y_train_resampled1)
    # y_pred2 = lgb_classifier.predict(x_test)
    # accuracy2 = accuracy_score(y_test, y_pred2)
    # print(f'ADASYN后的准确率: {accuracy2}')
    # f1_score_adasyn = f1_score(y_test, y_pred2)
    # print(f'ADASYN后的F1分数: {f1_score_adasyn}')
    #
    # '''SMOTE后'''
    # from imblearn.over_sampling import SMOTE
    #
    # # 创建 SMOTE 类的实例
    # smote = SMOTE(random_state=None)
    #
    # # 使用 SMOTE 对训练数据进行过采样
    # x_train_resampled2, y_train_resampled2 = smote.fit_resample(x_train, y_train)
    #
    # # 使用 LightGBM 训练模型并进行预测
    # lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
    # lgb_classifier.fit(x_train_resampled2, y_train_resampled2)
    # y_pred3 = lgb_classifier.predict(x_test)
    # accuracy3 = accuracy_score(y_test, y_pred3)
    # print(f'SMOTE后的准确率: {accuracy3}')
    # f1_score_smote = f1_score(y_test, y_pred3)
    # print(f'SMOTE后的F1分数: {f1_score_smote}')
    #
    # test = pd.read_excel(r'D:\0统计建模\代码\喜德县_LASSO.xlsx')
    # x_test = test.iloc[:, 4:-1]
    # y_test = test.iloc[:, -1]
    #
    # '''MAHAKIL前'''
    # lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
    # lgb_classifier.fit(x_train, y_train)
    # y_pred = lgb_classifier.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f'MAHAKIL前的准确率: {accuracy}')
    # f1_score_before = f1_score(y_test, y_pred)
    # print(f'MAHAKIL前的F1分数: {f1_score_before}')
    #
    # '''MAHAKIL后'''
    # lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
    # lgb_classifier.fit(x_resampled, y_resampled)
    # y_pred1 = lgb_classifier.predict(x_test)
    # accuracy1 = accuracy_score(y_test, y_pred1)
    # print(f'MAHAKIL后的准确率: {accuracy1}')
    # f1_score_after = f1_score(y_test, y_pred1)
    # print(f'MAHAKIL后的F1分数: {f1_score_after}')
    #
    # from imblearn.over_sampling import ADASYN
    #
    # adasyn = ADASYN(sampling_strategy='auto', random_state=42)
    # x_train_resampled1, y_train_resampled1 = adasyn.fit_resample(x_train, y_train)
    #
    # '''ADASYN后'''
    # lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
    # lgb_classifier.fit(x_train_resampled1, y_train_resampled1)
    # y_pred2 = lgb_classifier.predict(x_test)
    # accuracy2 = accuracy_score(y_test, y_pred2)
    # print(f'ADASYN后的准确率: {accuracy2}')
    # f1_score_adasyn = f1_score(y_test, y_pred2)
    # print(f'ADASYN后的F1分数: {f1_score_adasyn}')
    #
    # '''SMOTE后'''
    # from imblearn.over_sampling import SMOTE
    #
    # # 创建 SMOTE 类的实例
    # smote = SMOTE(random_state=None)
    #
    # # 使用 SMOTE 对训练数据进行过采样
    # x_train_resampled2, y_train_resampled2 = smote.fit_resample(x_train, y_train)
    #
    # # 使用 LightGBM 训练模型并进行预测
    # lgb_classifier = lgb.LGBMClassifier(random_state=None, verbosity=-1)
    # lgb_classifier.fit(x_train_resampled2, y_train_resampled2)
    # y_pred3 = lgb_classifier.predict(x_test)
    # accuracy3 = accuracy_score(y_test, y_pred3)
    # print(f'SMOTE后的准确率: {accuracy3}')
    # f1_score_smote = f1_score(y_test, y_pred3)
    # print(f'SMOTE后的F1分数: {f1_score_smote}')
