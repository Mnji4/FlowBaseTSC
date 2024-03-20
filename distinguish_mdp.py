import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.kernel_density import KDEMultivariate,KDEMultivariateConditional
from scipy.stats import entropy
import torch
n = 100

# state = np.random.normal(0, 1, n)
# next_state = state + np.random.normal(0, 0.5, n)

# # 构建核密度估计对象
# kde = KDEMultivariate([state, next_state], bw='cv_ml')

# # 估计条件概率密度
# x = np.linspace(-3, 3, 100)
# y = np.linspace(-4, 4, 100)
# X, Y = np.meshgrid(x, y)
# z = [kde.pdf([X.ravel(), Y.ravel()])]

# # 绘制条件概率密度曲面
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, np.array(z).reshape(X.shape), cmap='viridis')
# ax.set_xlabel('State')
# ax.set_ylabel('Next State')
# ax.set_zlabel('Density')
# plt.savefig('1.png')

state = np.random.normal(0, 1, (20,2))
next_state = 0 * state + np.random.normal(0, 1, (20,2))/10
hist_state = np.concatenate((np.random.normal(0, 0.01, (20,2)),state),1)
# hist_state = np.concatenate((state+ np.random.normal(0, 1, (20,2))/10,next_state),1)
# 构建条件核密度估计对象
kde_single = KDEMultivariateConditional(endog=next_state, exog=state, dep_type='cc', indep_type='cc', bw='cv_ml')
kde_hist = KDEMultivariateConditional(endog=next_state, exog=hist_state, dep_type='cc', indep_type='cccc', bw='cv_ml')
# 估计条件概率密度
# exog_hist = np.random.normal(0, 1, (10,4))
# exog_single = exog_hist[:,2:]
# endog_predict = np.random.normal(0, 1, (10,2))

# 使用 pdf 方法评估概率密度函数
density_values1 = kde_single.pdf()
density_values2 = kde_hist.pdf()
print(density_values1)
print(density_values2)
print(torch.nn.functional.kl_div(torch.tensor(density_values1),torch.tensor(density_values2)))
print(entropy(density_values1,density_values2))
