from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import lasso_path, enet_path, ElasticNetCV

from scripts.msa import MSA_collection
from scripts.helpers import DATA_PATH

plot = False

data_train_path = DATA_PATH / "fasta" / "training"
data_test_path = DATA_PATH / "fasta" / "testing"

# create MSA
msa_train = MSA_collection(data_train_path)
msa_test = MSA_collection(data_test_path)

# extract time from filenames
t_train = [int(name.stem.split('_')[-1][1:]) for name in list(data_train_path.glob('*.fasta'))]
t_test = [int(name.stem.split('_')[-1][1:]) for name in list(data_test_path.glob('*.fasta'))]
t_train.sort()
t_test.sort()

# calculate diversities
div_train = np.array([msa.diversity for msa in msa_train.msa_collection])
div_test = np.array([msa.diversity for msa in msa_test.msa_collection])

if plot:
    #######################################
    # fit elasitc net- & lasso regression
    #######################################
    eps = 5e-3
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(
        div_train, t_train, eps=eps, fit_intercept=False
    )

    print("Computing regularization path using the elastic net...")
    alphas_enet, coefs_enet, _ = enet_path(
        div_train, t_train, eps=eps, l1_ratio=0.5, fit_intercept=False
    )

    # Display results
    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_lasso = -np.log10(alphas_lasso)
    neg_log_alphas_enet = -np.log10(alphas_enet)

    plt.figure(2)
    for coef_l, coef_e, c in zip(coefs_lasso, coefs_enet, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
        l2 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso and Elastic-Net Paths')
    plt.legend((l1[-1], l2[-1]), ('Lasso', 'Elastic-Net'), loc='lower left')
    plt.axis('tight')

    plt.figure(3)
    for coef_l, c in zip(coefs_lasso, colors):
        l1 = plt.plot(neg_log_alphas_lasso, coef_l, c=c)
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Lasso')
    plt.axis('tight')

    plt.figure(4)
    for (coef_e, c) in zip(coefs_enet, colors):
        l1 = plt.plot(neg_log_alphas_enet, coef_e, c=c)
    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    plt.title('Elastic-Net')
    plt.axis('tight')
    plt.show()

from sklearn.metrics import mean_squared_error
model = ElasticNetCV(l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                     #n_alphas=500,
                     alphas=np.logspace(-5, -0.3, 500),
                     cv=5,
                     fit_intercept=False,
                     max_iter=6000,
                     n_jobs=7)
model.fit(X=div_train, y=t_train)

# intercept = model.intercept_
# coeff = model.coef_
# t_est_train = []
# print(intercept)
# print(coeff)
# for i in range(len(div_train)):
#     t_est_train.append(intercept + np.matmul(div_train[i], coeff))
#     print(f"t = {t_train[i]};  t_est = {t_est_train[-1]}")
#
# t_est_test = []
# for i in range(len(div_test)):
#     t_est_test.append(intercept + np.matmul(div_test[i], coeff))
#     print(f"t = {t_test[i]};  t_est = {t_est_test[-1]}")

print(f"intercept: {model.intercept_}")
print(f"alpha: {model.alpha_}")
print(f"l1 ratio: {model.l1_ratio_}")
train_pred = model.predict(div_train)
test_pred = model.predict(div_test)
print(f"mean squared error: {mean_squared_error(y_true=t_test,y_pred=test_pred)}")

fig = plt.figure()
plt.plot(np.arange(100), np.arange(100), color='black')
plt.scatter(t_train, train_pred, color='tab:blue')
plt.scatter(t_test, test_pred, color="tab:green")
plt.show()