"""
Perform cross-validatio for lasso- (lasso) and elastic net (elastic_net) regression
Compute regularization paths for lasso and elastic net (regularization_path)
"""

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path, ElasticNetCV, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split

from scripts.msa import MSA_collection
from scripts.helpers import DATA_PATH

regularization_path = True
lasso = True
elastic_net = True
scale_data = False
scaler = StandardScaler

data_path = DATA_PATH / "fasta"

# create MSA
msa = MSA_collection(data_path)

# extract time from filenames
t = [int(name.stem.split('_')[-1][1:]) for name in list(data_path.glob('*.fasta'))]
t.sort()

# calculate diversities
div = np.array([msa.diversity for msa in msa.msa_collection])

# split data into training and testing set (set random_state for reproducibility)
div_train, div_test, t_train, t_test = train_test_split(
    div, t, test_size=0.3, random_state=10)

# scale data
if scale_data:
    div_train = scale(div_train, with_mean=True)
    div_test = scale(div_test, with_mean=True)


#########################################################
# compute regularization paths for lasso & elastic net
#########################################################
if regularization_path:
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

#################################
# Lasso Cross Validation
#################################
if lasso:
    print('*'*80)
    print("performing lasso cross-validation")
    print('*'*80)

    # perform cross-validation and model fit
    model_lasso = LassoCV(
        eps=1e-3,
        alphas=np.logspace(-3, 0, 100),
        # n_alphas=100,
        cv=5,
        fit_intercept=False,
        max_iter=4000,
        n_jobs=7
    )
    model_lasso.fit(X=div_train, y=t_train)

    # plot alpha vs. mse
    fig = plt.figure()
    plt.errorbar(model_lasso.alphas_, model_lasso.mse_path_.mean(axis=1),
                 yerr=model_lasso.mse_path_.std(axis=1))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("alpha")
    plt.ylabel("mse")
    plt.show()

    # make predictions based on model
    train_pred = model_lasso.predict(div_train)
    test_pred = model_lasso.predict(div_test)
    mse = mean_squared_error(t_test, test_pred)
    print(f"alpha: {model_lasso.alpha_}")
    print(f"mean squared error: {mse}")

    # plot predictions
    fig1 = plt.figure()
    plt.scatter(t_train, train_pred, color='tab:blue', label="training set")
    plt.scatter(t_test, test_pred, color="tab:green", label="testing set")
    plt.ylabel('t estimate')
    plt.xlabel('t')
    plt.title("Lasso")
    if not scale_data:
        plt.plot(np.arange(100), np.arange(100), color='black')
        plt.annotate(f"alpha = {round(model_lasso.alpha_, 3)}", xy=(0, 80))
        plt.annotate(f"mse = {round(mse, 1)}", xy=(0, 70))
    plt.legend()
    fig1.show()


#################################
# Elastic Net Cross Validation
#################################
if elastic_net:
    print('\n')
    print('*'*80)
    print("performing elastic net cross-validation")
    print('*'*80)

    # perform cross validation and fit model
    model_en = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
        # n_alphas=500,
        alphas=np.logspace(-3, 0, 500),
        cv=5,
        fit_intercept=False,
        max_iter=4000,
        n_jobs=7
    )
    model_en.fit(X=div_train, y=t_train)


    # make prediction based on model
    train_pred = model_en.predict(div_train)
    test_pred = model_en.predict(div_test)
    mse = mean_squared_error(t_test, test_pred)
    print(f"alpha: {model_en.alpha_}")
    print(f"l1 ratio: {model_en.l1_ratio_}")
    print(f"mean squared error: {mse}")

    # plot predictions
    fig1 = plt.figure()
    plt.scatter(t_train, train_pred, color='tab:blue', label="training set")
    plt.scatter(t_test, test_pred, color="tab:green", label="testing set")
    plt.ylabel('t estimate')
    plt.xlabel('t')
    plt.title("Elastic Net")
    if not scale_data:
        plt.plot(np.arange(100), np.arange(100), color='black')
        plt.annotate(f"alpha = {round(model_en.alpha_, 3)}", xy=(0, 100))
        plt.annotate(f"l1-ratio = {model_en.l1_ratio_}", xy=(0, 90))
        plt.annotate(f"mse = {round(mse, 1)}", xy=(0, 80))
    plt.legend()
    fig1.show()