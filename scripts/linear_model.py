from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import lasso_path, enet_path, ElasticNetCV, LassoCV, LassoLarsIC, Lasso, LassoLars, lars_path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, scale
from sklearn.model_selection import train_test_split

from scripts.msa import MSA_collection
from scripts.helpers import DATA_PATH, calc_aic


class linear_model:

    def __init__(self, msa_path):
        self.msa = MSA_collection(msa_path)
        self._split_data()

    def _split_data(self):
        """ Split data into training and testing set (70 : 30) """
        training_set, testing_set = train_test_split(
            self.msa.info, test_size=0.3, random_state=10)
        self.x_train = training_set["diversity_per_site"].values.tolist()
        self.x_test = testing_set["diversity_per_site"].values.tolist()
        self.y_train = training_set["time"].values
        self.y_test = testing_set["time"].values

    def create_model(self, type, **kwargs):
        if type == "lassoCV":
            self.lassoCV(**kwargs)
        elif type == "lassoAIC":
            self.lassoAIC(**kwargs)
        elif type == "lassoLarsAIC":
            self.lassoLarsAIC(**kwargs)
        elif type == "elasticNetCV":
            self.elasticNetCV(**kwargs)
        else:
            raise ValueError(
                f"'type' {type} is not a valid option. 'type' must be 'lassoCV', 'lassoAIC', 'lassoLarsAIC' or 'elasticNetCV"
            )

    def lassoCV(self, **kwargs):
        print('*' * 80)
        print("performing lasso cross-validation")
        print('*' * 80)

        # perform cross-validation and model fit
        self.model = LassoCV(**kwargs)
        self.model.fit(X=self.x_train, y=self.y_train)

        # make predictions based on model
        y_train_pred = self.model.predict(self.x_train)
        y_test_pred = self.model.predict(self.x_test)
        mse = mean_squared_error(self.y_test, y_test_pred)
        alpha = self.model.alpha_
        print(f"alpha: {self.model.alpha_}")
        print(f"mean squared error: {mse}")

        # plot predictions
        self.plot_predictions(y_train_pred, y_test_pred, alpha, mse)

        # plot alpha vs. mse
        fig = plt.figure()
        plt.errorbar(self.model.alphas_, self.model.mse_path_.mean(axis=1),
                     yerr=self.model.mse_path_.std(axis=1))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("alpha")
        plt.ylabel("mse")
        plt.show()

    def lassoAIC(self, **kwargs):
        print('*' * 80)
        print("performing lasso AIC model selection")
        print('*' * 80)
        alphas, coeff, _ = lasso_path(X=self.x_train, y=self.y_train, **kwargs)

        # calculate aic
        aic = calc_aic(self.x_train, self.y_train, coeff)

        # create model
        min_idx = np.argmin(aic)
        self.model = Lasso(alpha=alphas[min_idx], fit_intercept=False, precompute=False)

        # make model predictions
        coeff_optimal = coeff[:, np.argmin(aic)]
        train_pred = np.dot(self.x_train, coeff_optimal)
        test_pred = np.dot(self.x_test, coeff_optimal)
        alpha = alphas[np.argmin(aic)]
        mse = np.power(self.y_test - test_pred, 2).mean()

        # plot AIC vs. alphas
        self.plot_alpha(alphas, aic, 'AIC')

        # plot time vs. time predictions
        self.plot_predictions(train_pred, test_pred, alpha, mse)

    def lassoLarsAIC(self):
        self.model = LassoLarsIC(
            criterion='aic',
            fit_intercept=False,
            normalize=False,
        )
        self.model.fit(X=self.x_train, y=self.y_train)
        alphas = self.model.alphas_
        aic = self.model.criterion_

        # make model predictions
        coeff_optimal = self.model.coef_
        train_pred = np.dot(self.x_train, coeff_optimal)
        test_pred = np.dot(self.x_test, coeff_optimal)
        alpha = alphas[np.argmin(aic)]
        mse = np.power(self.y_test - test_pred, 2).mean()

        # plot AIC vs. alpha
        self.plot_alpha(alphas, aic, 'AIC')

        # plot time vs. time predictions
        self.plot_predictions(train_pred, test_pred, alpha, mse)

    def elasticNetCV(self, **kwargs):
        print('*' * 80)
        print("performing elastic net cross-validation")
        print('*' * 80)

        # perform cross validation and fit model
        self.model = ElasticNetCV(**kwargs)
        self.model.fit(X=self.x_train, y=self.y_train)

        # make prediction based on model
        train_pred = self.model.predict(self.x_train)
        test_pred = self.model.predict(self.x_test)
        alpha = self.model.alpha_
        mse = mean_squared_error(self.y_test, test_pred)
        print(f"alpha: {alpha}")
        print(f"l1 ratio: {self.model.l1_ratio_}")
        print(f"mean squared error: {mse}")

        # plot predictions
        self.plot_predictions(train_pred, test_pred, alpha, mse)

    def plot_predictions(self, y_train_pred, y_test_pred, alpha, mse):
        fig = plt.figure()
        plt.scatter(self.y_train, y_train_pred, color='tab:blue', label="training set")
        plt.scatter(self.y_test, y_test_pred, color='tab:green', label='testing set')
        plt.xlabel('time')
        plt.ylabel('time prediction')

        plt.plot(np.arange(max(self.y_train)), np.arange(max(self.y_train)), color='black')
        plt.annotate(f"alpha = {round(alpha, 3)}", xy=(0, 80))
        plt.annotate(f"mse = {round(mse, 1)}", xy=(0, 70))

        plt.legend()
        fig.show()

    def plot_alpha(self, x, y, y_label):
        min_idx = np.argmin(y)

        fig = plt.figure()
        plt.semilogx(x, y, '--', color='black',
                     linewidth=3, label=y_label)
        plt.axvline(x[min_idx], color='black', linewidth=3,
                    label='alpha: best estimate')
        plt.xlabel('alpha')
        plt.ylabel(y_label)
        fig.show()


data_path = DATA_PATH / "fasta"
model = linear_model(data_path)
lasso_kwargs = {
    'eps': 1e-3,
    'alphas': np.logspace(-3, 0, 100),
    'cv': 5,
    'fit_intercept': False,
    'max_iter': 4000,
    'n_jobs': 7
}
lasso_aic_kwargs = {
    'precompute': False,
    'alpha_min': 0,
    'max_iter': 4000,
    'n_alphas': 500,
    'fit_intercept': False
}
en_kwargs = {
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
    'alphas': np.logspace(-3, 0, 500),
    'cv': 5,
    'fit_intercept': False,
    'max_iter': 4000,
    'n_jobs': 7
}

# model.create_model('lassoCV', **lasso_kwargs)
# model.create_model('lassoAIC', **lasso_aic_kwargs)
# model.create_model('lassoLarsAIC')
model.create_model('elasticNetCV', **en_kwargs)