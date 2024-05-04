import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

data = pd.read_excel(r'D:\2024统计建模\训练集_LASSO.xlsx')

X = data.iloc[:, 4:-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1412)

'''LightGBM'''


def LightGBM_objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 1, 1000, 1)
    max_depth = trial.suggest_int('max_depth', 1, 100, 1)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.999, log=True)
    subsample = trial.suggest_float('subsample', 0.05, 0.99)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 300)
    gamma = trial.suggest_float('gamma', 0, 1)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.1, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)

    clf = lgb.LGBMClassifier(n_estimators=n_estimators,
                             max_depth=max_depth,
                             learning_rate=learning_rate,
                             subsample=subsample,
                             min_child_weight=min_child_weight,
                             gamma=gamma,
                             colsample_bytree=colsample_bytree,
                             reg_alpha=reg_alpha,
                             reg_lambda=reg_lambda,
                             random_state=1412,
                             verbosity=-1)

    # 使用划分好的训练集进行拟合
    clf.fit(X_train, y_train)

    # 使用划分好的验证集进行交叉验证评估
    cv = KFold(n_splits=10, shuffle=True, random_state=1412)
    validation_accuracy = cross_validate(clf, X_test, y_test,
                                         scoring='accuracy',
                                         cv=cv,
                                         verbose=False,
                                         n_jobs=-1,
                                         error_score='raise')

    return np.mean(validation_accuracy['test_score'])


'''贝叶斯优化器'''


def optimizer_optuna(n_trials, algo, objective):
    if algo == 'TPE':
        sampler = optuna.samplers.TPESampler(n_startup_trials=50, n_ei_candidates=100)
    elif algo == 'GP':
        from optuna.integration import SkoptSampler
        import skopt
        sampler = SkoptSampler(skopt_kwargs={'base_estimator': 'GP',
                                             'n_initial_points': 10,
                                             'acq_func': 'EI'})

    study = optuna.create_study(sampler=sampler, direction='maximize')

    scores = []

    for _ in range(n_trials):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        best_score = study.best_value
        scores.append(best_score)

    best_params = study.best_params

    return best_params, scores


'''使用TPE过程进行贝叶斯优化'''
LightGBM_best_params, lightGBM_scores = optimizer_optuna(500, 'TPE', LightGBM_objective)

'''最优超参数组合'''
results = {
    'LightGBM': (LightGBM_best_params, lightGBM_scores),
}

for algorithm, (best_params, scores) in results.items():
    print(f'{algorithm} - Best Parameters: {best_params}, Best Scores: {scores}')

results = pd.DataFrame(results)
results.to_excel(r'LightGBM基于TPE过程贝叶斯优化最优超参数组合_LASSO.xlsx', index=False)
