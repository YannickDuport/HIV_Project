import copy

from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ray

from helpers import split

from decimal import Decimal
from matplotlib.colors import LogNorm

from sklearn.linear_model import lasso_path, ElasticNetCV, LassoCV, \
    LassoLarsIC, Lasso, ElasticNet, LassoLars
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from scripts.msa import MSA_collection
from scripts.helpers import DATA_PATH, calc_aic, calc_aic_1d, calc_aic_depr


class linear_model:

    def __init__(self, msa_path):
        self.msa = MSA_collection(msa_path, True)
        self._split_data()

    def _split_data(self):
        """ Split data into training and testing set (70 : 30) """
        training_set, testing_set = train_test_split(
            self.msa.info, test_size=0.3) #, random_state=10)
        self.x_train = np.array(training_set["diversity_per_site"].values.tolist())
        self.x_test = np.array(testing_set["diversity_per_site"].values.tolist())
        self.y_train = training_set["time"].values
        self.y_test = testing_set["time"].values
        self.weights_train = training_set["sample_weight"].values
        self.weights_test = testing_set["sample_weight"].values

    def create_model(self, type, alphas=None, l1_ratios=None, **kwargs):
        if alphas is None:
            alphas = np.logspace(-3, 0, 100)
        if l1_ratios is None:
            l1_ratios = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]

        if type == "lassoCV":
            self.crossValidation(alphas=alphas, l1_ratios=[1], **kwargs)
        elif type == "lassoAIC":
            self.lassoAIC(alphas=alphas, **kwargs)
        elif type == "elasticNetCV":
            self.crossValidation_parallel(alphas=alphas, l1_ratios=l1_ratios, **kwargs)

        # Deprecated (only used as validation of other methods)
        elif type == "lassoCV_depr":
            self.lassoCV_depr(alphas=alphas, **kwargs)
        elif type == "lassoAIC":
            self.lassoAIC(alphas=alphas, **kwargs)
        elif type == "lassoLarsAIC_depr":
            self.lassoLarsAIC_depr(alphas=alphas, **kwargs)
        elif type == "elasticNetCV_depr":
            self.elasticNetCV_depr(**kwargs)

        else:
            raise ValueError(
                f"'type' {type} is not a valid option. 'type' must be 'lassoCV', 'lassoAIC' or 'elasticNetCV"
            )

    @ray.remote
    def _elasticNet(self, idx_subset, alphas, l1_ratios, **kwargs):
        print(idx_subset)
        # create training and testing set for current fold
        x_train = np.delete(self.x_train, idx_subset, axis=0)
        x_test = self.x_train[idx_subset]
        y_train = np.delete(self.y_train, idx_subset, axis=0)
        y_test = self.y_train[idx_subset]
        weights_train = np.delete(self.weights_train, idx_subset)
        weights_test = self.weights_train[idx_subset]

        mses = []
        for ratio in l1_ratios:
            mses.append([])
            for alpha in alphas:
                model = ElasticNet(alpha=alpha, l1_ratio=ratio, **kwargs)
                model.fit(x_train, y_train, sample_weight=weights_train)

                # make predictions based on model
                y_test_pred = model.predict(x_test)
                mse = mean_squared_error(y_test, y_test_pred, sample_weight=weights_test)
                mses[-1].append(mse)

        return mses

    def crossValidation_parallel(self, alphas, l1_ratios, folds=10, random_state=None, n_jobs=None, **kwargs):
        print("start Cross Validation")

        ray.init(num_cpus=n_jobs)

        # split dataset into n subsets
        cv_idx_subset = split(self.y_train, folds, random_state)

        # run cross validation to find optimal alpha (and l1 ratio)
        mses_ids = []
        for i, idx_subset in enumerate(cv_idx_subset):
            # create and fit model for current fold
            mses_ids.append(self._elasticNet.remote(self, idx_subset, alphas, l1_ratios, **kwargs))
        mses = ray.get(mses_ids)

        # find optimal alpha, l1_ratio and the corresponding mse (by minimizing mse)
        mses = np.array(mses)
        mse_per_alpha = mses.mean(axis=0)
        mse_per_alpha_sd = mses.std(axis=0)
        idx_min_row, idx_min_col = np.unravel_index(mse_per_alpha.argmin(), mse_per_alpha.shape)
        mse_min = mse_per_alpha[idx_min_row, idx_min_col] / folds
        l1_ratio_min = l1_ratios[idx_min_row]
        alpha_min = alphas[idx_min_col]

        print(f"mse = {mse_min}")
        print(f"alpha = {alpha_min}")
        print(f"l1 ratio = {l1_ratio_min}")

        # build and fit model
        self.model = ElasticNet(alpha=alpha_min, l1_ratio=l1_ratio_min, **kwargs)
        self.model.fit(X=self.x_train, y=self.y_train, sample_weight=self.weights_train)

        # make predictions based on model
        y_train_pred = self.model.predict(self.x_train)
        y_test_pred = self.model.predict(self.x_test)
        mse_train = mean_squared_error(self.y_train, y_train_pred, sample_weight=self.weights_train)
        mse_test = mean_squared_error(self.y_test, y_test_pred, sample_weight=self.weights_test)

        method = "Lasso" if l1_ratios == [1] else "ElasticNet"
        plot_kwargs = {
            "alpha": alpha_min,
            "l1_ratio": l1_ratio_min,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "title": f"Weighted {method} CV",
        }
        self.plot_predictions(y_train_pred, y_test_pred, **plot_kwargs)

        # plot alpha vs. mse
        fig = plt.figure()
        for k, ratio in enumerate(l1_ratios):
            plt.errorbar(alphas, mse_per_alpha[k], mse_per_alpha_sd[k], label=ratio)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("alpha")
        plt.ylabel("mse")
        plt.legend()
        plt.show()

        return alpha_min, mse_min


    def crossValidation(self, alphas, l1_ratios, folds=5, random_state=None, **kwargs):
        print("start Cross Validation")

        # split dataset into n subsets
        cv_idx_subset = split(self.y_train, folds, random_state)

        # run cross validation to find optimal alpha (and l1 ratio)
        mses = []
        for i, idx_subset in enumerate(cv_idx_subset):
            print(f"Fold {i+1}/{folds}")

            # create and fit model for current fold
            mse_per_alpha = self._elasticNet(idx_subset, alphas, l1_ratios, **kwargs)
            mses.append(mse_per_alpha)

        # find optimal alpha, l1_ratio and the corresponding mse (by minimizing mse)
        mses = np.array(mses)
        mse_per_alpha = mses.mean(axis=0)
        mse_per_alpha_sd = mses.std(axis=0)
        idx_min_row, idx_min_col = np.unravel_index(mse_per_alpha.argmin(), mse_per_alpha.shape)
        mse_min = mse_per_alpha[idx_min_row, idx_min_col] / folds
        l1_ratio_min = l1_ratios[idx_min_row]
        alpha_min = alphas[idx_min_col]

        print(f"mse = {mse_min}")
        print(f"alpha = {alpha_min}")
        print(f"l1 ratio = {l1_ratio_min}")

        # build and fit model
        self.model = ElasticNet(alpha=alpha_min, l1_ratio=l1_ratio_min, **kwargs)
        self.model.fit(X=self.x_train, y=self.y_train, sample_weight=self.weights_train)

        # make predictions based on model
        y_train_pred = self.model.predict(self.x_train)
        y_test_pred = self.model.predict(self.x_test)
        mse_train = mean_squared_error(self.y_train, y_train_pred, sample_weight=self.weights_train)
        mse_test = mean_squared_error(self.y_test, y_test_pred, sample_weight=self.weights_test)

        method = "Lasso" if l1_ratios==[1] else "ElasticNet"
        plot_kwargs = {
            "alpha": alpha_min,
            "l1_ratio": l1_ratio_min,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "title": f"Weighted {method} CV",
        }
        self.plot_predictions(y_train_pred, y_test_pred, **plot_kwargs)

        # plot alpha vs. mse
        fig = plt.figure()
        for k, ratio in enumerate(l1_ratios):
            plt.errorbar(alphas, mse_per_alpha[k], mse_per_alpha_sd[k], label=ratio)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("alpha")
        plt.ylabel("mse")
        plt.legend()
        plt.show()

        return alpha_min, mse_min


    def lassoAIC(self, alphas=None, **kwargs):
        print('*' * 80)
        print("performing lasso AIC model selection")
        print('*' * 80)

        # tune for optimal alpha by minimizing AIC
        aic = []
        coefficients = []
        for alpha in alphas:
            # create and fit model
            model = ElasticNet(alpha, l1_ratio=1, **kwargs)
            model.fit(X=self.x_train, y=self.y_train, sample_weight=self.weights_train)

            # calculate aic
            coeff = model.coef_
            coefficients.append(coeff)
            aic.append(calc_aic_1d(self.x_train, self.y_train, coeff, self.weights_train))

        # create model
        min_idx = np.argmin(aic)
        alpha_min = alphas[min_idx]
        self.model = ElasticNet(alpha=alpha_min, l1_ratio=1, fit_intercept=False, precompute=False)
        self.model.fit(X=self.x_train, y=self.y_train, sample_weight=self.weights_train)

        # make model predictions
        y_train_pred = self.model.predict(self.x_train)
        y_test_pred = self.model.predict(self.x_test)
        mse_train = mean_squared_error(self.y_train, y_train_pred, sample_weight=self.weights_train)
        mse_test = mean_squared_error(self.y_test, y_test_pred, sample_weight=self.weights_test)

        # plot AIC vs. alphas
        self.plot_alpha(alphas, aic, 'AIC')

        # plot time vs. time predictions
        plot_kwargs = {
            "alpha": alpha_min,
            "l1_ratio": 1,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "title": "Weighted Lasso AIC",
        }
        self.plot_predictions(y_train_pred, y_test_pred, **plot_kwargs)


    def lassoCV_depr(self, alphas, **kwargs):
        print('*' * 80)
        print("performing lasso cross-validation")
        print('*' * 80)

        # perform cross-validation and model fit
        self.model = LassoCV(alphas=alphas, **kwargs)
        self.model.fit(X=self.x_train, y=self.y_train)

        # make predictions based on model
        y_train_pred = self.model.predict(self.x_train)
        y_test_pred = self.model.predict(self.x_test)
        mse_train = mean_squared_error(self.y_train, y_train_pred)
        mse_test = mean_squared_error(self.y_test, y_test_pred)
        alpha = self.model.alpha_
        print(f"alpha: {self.model.alpha_}")
        print(f"mean squared error: {mse_test}")

        # plot predictions
        plot_kwargs = {
            "alpha": alpha,
            "l1_ratio": 1,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "title": "Lasso CV",
        }
        self.plot_predictions(y_train_pred, y_test_pred, **plot_kwargs)

        # plot alpha vs. mse
        fig = plt.figure()
        plt.errorbar(self.model.alphas_, self.model.mse_path_.mean(axis=1),
                     yerr=self.model.mse_path_.std(axis=1))
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("alpha")
        plt.ylabel("mse")
        plt.show()


    def lassoLarsAIC_depr(self, alphas, **kwargs):
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


    def elasticNetCV_depr(self, weights, **kwargs):
        print('*' * 80)
        print("performing elastic net cross-validation")
        print('*' * 80)

        # perform cross validation and fit model
        # alphas = np.logspace(-3, 0, 100)
        self.model = ElasticNetCV(alphas=alphas, **kwargs)
        self.model.fit(X=self.x_train, y=self.y_train)

        # make prediction based on model
        train_pred = self.model.predict(self.x_train)
        test_pred = self.model.predict(self.x_test)
        alpha = self.model.alpha_
        mse_train = mean_squared_error(self.y_train, train_pred)
        mse_test = mean_squared_error(self.y_test, test_pred)
        print(f"alpha: {alpha}")
        print(f"l1 ratio: {self.model.l1_ratio_}")
        print(f"mean squared error: {mse_test}")

        # plot predictions
        plot_kwargs = {
            "alpha": alpha,
            "l1_ratio": self.model.l1_ratio_,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "title": "ElasticNet CV",
        }
        self.plot_predictions(train_pred, test_pred, **plot_kwargs)

        # plot alpha vs. mse

        fig = plt.figure()
        for i, path in enumerate(self.model.mse_path_):
            plt.errorbar(self.model.alphas_, path.mean(axis=1), yerr=path.std(axis=1),
                         label=kwargs["l1_ratio"][i])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("alpha")
        plt.ylabel("mse")
        plt.legend()
        plt.show()


    def plot_predictions(self, y_train_pred, y_test_pred, alpha, l1_ratio, mse_train, mse_test, title):
        fig = plt.figure()
        plt.scatter(self.y_train, y_train_pred, color='tab:blue', label="training set", s=np.sqrt(self.weights_train*20)*10)
        plt.scatter(self.y_test, y_test_pred, color='tab:green', label='testing set')
        plt.plot(np.arange(max(self.y_train)), np.arange(max(self.y_train)), color='black')
        plt.xlabel('time')
        plt.ylabel('time predicted')

        plt.title(title)
        plt.annotate(f"alpha: {Decimal(alpha):.2E}\n"
                     f"l1 ratio: {l1_ratio}\n"
                     f"mse (training set): {round(mse_train, 1)}\n"
                     f"mse (testing set): {round(mse_test, 1)}\n"
                     f"mse ratio (test/train): {round(mse_test/mse_train, 3)}",
                     xy=(0, 60))
        # plt.annotate(f"alpha = {round(alpha, 4)}", xy=(0, 80))
        # plt.annotate(f"mse = {round(mse, 1)}", xy=(0, 70))

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


    def plot_heatmap(self, x, y, color):
    # FIXME: fix ticklabels and order (ascending)
        fig, ax = plt.subplots()
        ax = sns.heatmap(color, xticklabels=x, yticklabels=y,
                         norm=LogNorm(vmin=np.min(color), vmax=np.max(color)))
        fig.show()

""" Deprecated

    def crossValidation(self, weights, folds=5, random_state=None, **kwargs):
        print("start Cross Validation")
        cv_idx_subset = split(self.y_train, folds, random_state)

        l1_ratio = [1]
        alphas = np.logspace(-3, 0, 100)
        mses = []
        n=0

        for idx_subset in cv_idx_subset:
            print(f"Fold {n+1}/{folds}")

            # create training and testing sets for current fold
            x_train = np.delete(self.x_train, idx_subset, axis=0)
            x_test = self.x_train[idx_subset]
            y_train = np.delete(self.y_train, idx_subset, axis=0)
            y_test = self.y_train[idx_subset]

            # create and fit model for current fold
            mse_per_alpha = self._lasso(alphas, x_train, y_train, x_test, y_test, **kwargs)
            mses.append(mse_per_alpha)
            n+=1

        mses = np.array(mses)
        mse_per_alpha = mses.mean(axis=0)
        mse_per_alpha_sd = mses.std(axis=0)
        idx_min = np.argmin(mse_per_alpha)
        mse_min = mse_per_alpha[idx_min]
        alpha_min = alphas[idx_min]

        self.model = Lasso(alpha=alpha_min, **kwargs)
        self.model.fit(X=self.x_train, y=self.y_train)

        # make predictions based on model
        y_train_pred = self.model.predict(self.x_train)
        y_test_pred = self.model.predict(self.x_test)
        mse_train = mean_squared_error(self.y_train, y_train_pred)
        mse_test = mean_squared_error(self.y_test, y_test_pred)
        print(f"alpha: {alpha_min}")
        print(f"mean squared error: {mse_test}")

        plot_kwargs = {
            "alpha": alpha_min,
            "l1_ratio": 1,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "title": "Weighted Lasso CV",
        }
        self.plot_predictions(y_train_pred, y_test_pred, **plot_kwargs)

        # plot alpha vs. mse
        fig = plt.figure()
        plt.errorbar(alphas, mse_per_alpha, yerr=mse_per_alpha_sd)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("alpha")
        plt.ylabel("mse")
        plt.show()

        return alpha_min, mse_min
    
    def _lasso(self, alphas,  x=None, y=None, x_test=None, y_test=None, weights=None, **kwargs):
        if x is None:
            x = self.x_train
        if y is None:
            y = self.y_train
        if weights is None:
            weights = np.full(len(y), 1)

        mses = []
        for alpha in alphas:
            model = Lasso(alpha, **kwargs)
            model.fit(x, y)

            # make predictions based on model
            # y_train_pred = model.predict(self.x_train)
            y_test_pred = model.predict(x_test)
            mse = mean_squared_error(y_test, y_test_pred)
            mses.append(mse)

        return mses
"""

data_path = DATA_PATH / "fasta"
model = linear_model(data_path)
lasso_kwargs = {
    #'eps': 1e-3,
    #'alphas': np.logspace(-3, 0, 100),
    #'cv': 5,
    'fit_intercept': False,
    'max_iter': 4000,
    #'n_jobs': 7
}
lasso_aic_kwargs = {
    'precompute': False,
    #'alpha_min': 0,
    'max_iter': 4000,
    # 'n_alphas': 1000,
    'fit_intercept': False
}
en_kwargs = {
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
    # 'alphas': np.logspace(-3, 0, 500),
    'cv': 5,
    'fit_intercept': False,
    'max_iter': 4000,
    'n_jobs': 7
}

weights = np.repeat(1, 105)
# weights = np.random.sample(105)
alphas = np.logspace(-3, 0, 100)

# odel.create_model("lassoCV", alphas, **lasso_kwargs)
model.create_model("elasticNetCV", alphas, **lasso_kwargs)
# model.create_model('lassoAIC_weight', alphas, **lasso_aic_kwargs)


# model.create_model('lassoCV', alphas, **lasso_kwargs)
# model.create_model('lassoAIC', alphas, **lasso_aic_kwargs)
# model.create_model('lassoLarsAIC', alphas)
# model.create_model('lassoLarsAIC_weight', alphas, **lasso_aic_kwargs)
# model.create_model('elasticNetCV', alphas, **en_kwargs)
