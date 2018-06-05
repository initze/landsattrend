import numpy as np
from sklearn import metrics, linear_model
import pandas as pd
import matplotlib.pyplot as plt

__author__ = 'Ingmar Nitze'

class Breakpoint(object):
    """
    """
    def __init__(self, x, y, n_breaks=1, predictor='r2'):
        """
        :param x:
        :param y:
        :param n_breaks:
        :param predictor:
        """
        self.x = x
        self.y = y
        self.n_breaks = n_breaks
        self.predictor = predictor
        self.data_len_ = len(x)
        self._validate_input()

    def _validate_input(self):
        if self.n_breaks not in [1,2]:
            raise ValueError('Currently only 1 or 2 Breakpoints supported!')
        if self.predictor not in ["r2", "mae"]:
            raise ValueError('Currently only "mae" or "r2" supported!')

    def _define_indices(self):
        """define indices of possible combinations"""
        if self.n_breaks == 1:
            self.rr_ = np.arange(2, self.data_len_ - 1)
            self.cc_ = np.array([self.data_len_] * (len(self.rr_)))

        if self.n_breaks == 2:
            gd_r, gd_c = np.mgrid[2:self.data_len_ - 1, 3:self.data_len_]
            r, c = np.triu_indices(self.data_len_ - 4)
            self.rr_, self.cc_ = gd_r[r, c], gd_c[r, c]

    def _calc_segments(self):
        """
        :return:
        """
        mae = []
        r2 = []
        yy_list = []
        for i, j in zip(self.rr_, self.cc_):
            yy = []
            # make trend of segments
            yy.extend(self.regression(self.x[:i], self.y[:i]))
            yy.extend(self.regression(self.x[i:j], self.y[i:j]))
            if self.n_breaks == 2:
                yy.extend(self.regression(self.x[j:], self.y[j:]))

            yy_list.append(yy)
            mae.append(metrics.mean_absolute_error(self.y, yy))
            r2.append(np.corrcoef(self.y, yy)[0, 1] ** 2)

        # results to pandas
        columns = ['break_loc1', 'break_loc2', 'break_year1', 'break_year2',
                   'mae', 'r2', 'mae_linear', 'r2_linear', 'n_breaks']
        results = pd.DataFrame(columns=columns)
        results.loc[:, 'break_loc1'] = self.rr_
        results.loc[:, 'break_year1'] = self.x[self.rr_]
        if self.n_breaks == 3:
            results.loc[:, 'break_loc2'] = self.cc_
            results.loc[:, 'break_year2'] = self.x[self.cc_]
        results.loc[:, 'mae'] = mae
        results.loc[:, 'r2'] = r2
        results.loc[:, 'n_breaks'] = self.n_breaks

        # calc full series
        y_full = self.regression(self.x, self.y)
        # metrics for single linear regression
        results.loc[:, 'mae_linear'] = metrics.mean_absolute_error(self.y, y_full)
        results.loc[:, 'r2_linear'] = np.corrcoef(self.y, y_full)[0,1] ** 2
        self.results_ = results

    def _get_best_fit(self):
        if self.predictor == 'mae':
            self.results_best_ = self.results_.loc[self.results_[self.predictor].idxmin()]
        elif self.predictor == 'r2':
            self.results_best_ = self.results_.loc[self.results_[self.predictor].idxmax()]

    def fit(self):
        """
        Wrapping function to call specific functions
        :return:
        """
        self._define_indices()
        self._calc_segments()
        self._get_best_fit()

    def plot(self):
        """
        function to plot results
        :return:
        """
        idx = self.results_best_['break_loc1']
        plt.plot(self.x, self.y)
        plt.plot(self.x[:], self.regression(self.x[:], self.y[:]), ls='--')
        plt.plot(self.x[:idx], Breakpoint.regression(self.x[:idx], self.y[:idx]))
        plt.plot(self.x[idx:], Breakpoint.regression(self.x[idx:], self.y[idx:]))
        plt.show()
        pass

    @staticmethod
    def regression(x, y):
        """
        :param x: np.ndarray
        :param y: np.ndarray
        :return: np.ndarray
        """
        mod1 = linear_model.LinearRegression().fit(np.expand_dims(x, 1), y)
        return mod1.predict(np.expand_dims(x, 1))
