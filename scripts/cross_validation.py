"""
Perform cross-validatio for lasso- (lasso) and elastic net (elastic_net) regression
Compute regularization paths for lasso and elastic net (regularization_path)
"""

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path, ElasticNetCV, LassoCV, LassoLarsIC, Lasso, LassoLars, lars_path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split

from scripts.msa import MSA_collection
from scripts.helpers import DATA_PATH, calc_aic

test = 1
regularization_path = False
lasso = False
lasso_aic = 0
elastic_net = False
scale_data = False
scaler = StandardScaler

data_path = DATA_PATH / "fasta"


# create MSA
msa = MSA_collection(data_path)

# split data into training and testing set (set random_state for reproducibility)
training_set, testing_set = train_test_split(
    msa.info, test_size=0.3, random_state=10)
x_train = training_set["diversity_per_site"].values.tolist()
x_test = testing_set["diversity_per_site"].values.tolist()
y_train = training_set["time"].values
y_test = testing_set["time"].values

# scale data
if scale_data:
    x_train = scale(x_train, with_mean=True)
    x_test = scale(x_test, with_mean=True)


##################################
# Lasso model fit using AIC
##################################
if lasso_aic:
    def plot_ic_criterion(model, name, color):
        EPSILON = 1e-5
        criterion_ = model.criterion_
        plt.semilogx(model.alphas_ + EPSILON, criterion_, '--', color=color,
                     linewidth=3, label='%s criterion' % name)
        plt.axvline(model.alpha_ + EPSILON, color=color, linewidth=3,
                    label='alpha: %s estimate' % name)
        plt.xlabel(r'$\alpha$')
        plt.ylabel('criterion')

    print('\n')
    print('*'*80)
    print("Lasso fit using AIC")
    print('*'*80)

    model_lasso_aic = LassoLarsIC(
        criterion='aic',
        fit_intercept=False,
        normalize=False,
    )
    model_lasso_aic.fit(X=x_train, y=y_train)

    fig1 = plt.figure()
    plot_ic_criterion(model_lasso_aic, 'AIC', 'b')
    #plt.plot(model_lasso_aic.alphas_, model_lasso_aic.criterion_)
    plt.xscale('log')
    plt.show()
    alphas = model_lasso_aic.alphas_

if test:
    # FIXME: Currently not working. Try with lasso_path

    fit_dict = {}
    # alphas, _, coeff = lars_path(
    #     X=x_train, y=np.array(y_train),
    #     alpha_min=0,
    #     method='lasso'
    # )

    # alphas_, _, coeff_, _ = lars_path(
    #     x_train, np.array(y_train), Gram='auto', copy_Gram=True, alpha_min=0.0,
    #     method='lasso', verbose=False, max_iter=500,
    #     return_n_iter=True, positive=False)
    print(x_train)
    alphas, coeff, _ = lasso_path(
        X=x_train, y=y_train, precompute=False,
        alpha_min=0, max_iter=4000, n_alphas=500, #alphas=alphas_,
        fit_intercept=False
    )

    # calculate aic
    aic = calc_aic(x_train, y_train, coeff)

    fig = plt.figure()
    plt.plot(alphas, aic)
    plt.xscale('log')
    plt.show()

    print(alphas[np.argmin(aic)])
    coeff_optimal = coeff[:, np.argmin(aic)]
    train_pred = np.dot(x_train, coeff_optimal)
    test_pred = np.dot(x_test, coeff_optimal)
    fig1 = plt.figure()
    plt.scatter(y_train, train_pred, color='tab:blue', label="training set")
    plt.scatter(y_test, test_pred, color="tab:green", label="testing set")
    plt.ylabel('t estimate')
    plt.xlabel('t')
    plt.title("Lasso")
    fig1.show()


#########################################################
# compute regularization paths for lasso & elastic net
#########################################################
if regularization_path:
    eps = 5e-3
    print("Computing regularization path using the lasso...")
    alphas_lasso, coefs_lasso, _ = lasso_path(
        x_train, y_train, eps=eps, fit_intercept=False
    )

    print("Computing regularization path using the elastic net...")
    alphas_enet, coefs_enet, _ = enet_path(
        x_train, y_train, eps=eps, l1_ratio=0.5, fit_intercept=False
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
    model_lasso.fit(X=x_train, y=y_train)

    # make predictions based on model
    train_pred = model_lasso.predict(x_train)
    test_pred = model_lasso.predict(x_test)
    mse = mean_squared_error(y_test, test_pred)
    print(f"alpha: {model_lasso.alpha_}")
    print(f"mean squared error: {mse}")

    # plot predictions
    fig1 = plt.figure()
    plt.scatter(y_train, train_pred, color='tab:blue', label="training set")
    plt.scatter(y_test, test_pred, color="tab:green", label="testing set")
    plt.ylabel('t estimate')
    plt.xlabel('t')
    plt.title("Lasso")
    if not scale_data:
        plt.plot(np.arange(100), np.arange(100), color='black')
        plt.annotate(f"alpha = {round(model_lasso.alpha_, 3)}", xy=(0, 80))
        plt.annotate(f"mse = {round(mse, 1)}", xy=(0, 70))
    plt.legend()
    fig1.show()

    # plot alpha vs. mse
    fig = plt.figure()
    plt.errorbar(model_lasso.alphas_, model_lasso.mse_path_.mean(axis=1),
                 yerr=model_lasso.mse_path_.std(axis=1))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("alpha")
    plt.ylabel("mse")
    plt.show()




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
    model_en.fit(X=x_train, y=y_train)


    # make prediction based on model
    train_pred = model_en.predict(x_train)
    test_pred = model_en.predict(x_test)
    mse = mean_squared_error(y_test, test_pred)
    print(f"alpha: {model_en.alpha_}")
    print(f"l1 ratio: {model_en.l1_ratio_}")
    print(f"mean squared error: {mse}")

    # plot predictions
    fig1 = plt.figure()
    plt.scatter(y_train, train_pred, color='tab:blue', label="training set")
    plt.scatter(y_test, test_pred, color="tab:green", label="testing set")
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