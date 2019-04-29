__author__ = "Ricardo"


import numpy as np
import Chapter_1 as C1
import Chapter_2 as C2
import pandas as pd
from scipy import stats as st


def get_deviations_residuals(error):
    e_m = np.median(error)
    d = abs(error - e_m)
    return d


def Brown_Forsythe_Test(X, Y, coef, bias, alpha):
    # print(X.T, Y.T)
    data = []
    data.append(list(X.T))
    data.append(list(Y.T))
    df = pd.DataFrame(data, index=list('XY')).T
    print(df)
    g1 = df.sample(frac=1)
    group1 = g1[:13]
    group2 = g1[13:]
    error1 = group1['Y'] - C1.predict(group1["X"], coef, bias)
    error2 = group2['Y'] - C1.predict(group2["X"], coef, bias)
    dev1 = get_deviations_residuals(error1)
    dev2 = get_deviations_residuals(error2)
    s_square = (np.dot((dev1 - np.average(dev1)).T, (dev1 - np.average(dev1)))
                + np.dot((dev1 - np.average(dev2)).T, (dev1 - np.average(dev2))) / (len(X) -2))
    s = np.power(s_square, 0.5)
    T_BF = (np.average(dev1) - np.average(dev2)) / (s * np.power(1 / 13 + 1 / 12, 0.5))
    t_test = st.t.ppf(1 - alpha / 2, len(X) - 2)


def main():
    X, Y = C1.load("Toluca.txt")
    n = len(X)
    coef, bias = C1.LSE_linear_regression(X, Y)
    SS_ = C2.get_SS(X,Y,coef, bias)
    Y_hat = C1.predict(X, coef, bias)
    MSE = SS_[2] / (n - 2)
    error = Y_hat - Y
    Brown_Forsythe_Test(X, Y)


if __name__ == '__main__':
    main()


