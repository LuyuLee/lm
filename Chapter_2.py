__author__ = 'Ricardo'


import Chapter_1 as C1
import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt


def get_S_coef(X, MSE):
    devia_x = X - np.average(X)
    lxx = np.dot(devia_x, devia_x.T)
    s_square = MSE / lxx
    return s_square ** 0.5


def Two_Sided_Test(coef, S_coef, alpha, n):
    t = coef / S_coef
    test = st.t.ppf(1 - alpha / 2, n - 2)
    #print("F: ", test)
    #print("t: ", abs(t))
    if abs(t) <= test:
        return "Acc"
    else:
        return "Ref"

def One_Sided_Test(coef,S_coef, alpha, n):
    t = coef / S_coef
    test = st.t.ppf(1 - alpha, n - 2)
    #print("F: ", test)
    #print("t: ", abs(t))
    if abs(t) <= test:
        return "Acc"
    else:
        return "Ref"


def get_S_Bias(X, MSE):
    n = len(X)
    X_ave = np.average(X)
    #print("X_ave", X_ave)
    devia_x = X - X_ave
    lxx = np.dot(devia_x, devia_x.T)
    #print(lxx)
    #print(MSE)
    S_square_bias = MSE * (1 / n + np.power(X_ave, 2) / lxx)
    #print(S_square_bias)
    return np.power(S_square_bias, 0.5)


def CI_bias(X, MSE, bias_expect, alpha):
    n = len(X)
    t = st.t.ppf(1 - alpha/2, n - 2)
    #print(t)
    s_bias = get_S_Bias(X, MSE)
    dif = s_bias * t
    #print(s_bias)
    low_bound = bias_expect - dif
    high_bound = bias_expect + dif
    ans = []
    ans.append(low_bound)
    ans.append(high_bound)
    return ans


def get_S_Y_hat(X, MSE, X_pre):
    n = len(X)
    X_ave = np.average(X)
    #print("X_ave", X_ave)
    devia_x = X - X_ave
    lxx = np.dot(devia_x, devia_x.T)
    S_square_Y = MSE * (1 / n + np.power((X_pre - X_ave), 2) / lxx)
    return np.power(S_square_Y, 0.5)


def get_S_Y_mean_hat(X, m, MSE, X_pre):
    n = len(X)
    X_ave = np.average(X)
    # print("X_ave", X_ave)
    devia_x = X - X_ave
    lxx = np.dot(devia_x, devia_x.T)
    S_square_Y = MSE * (1 / n + np.power((X_pre - X_ave), 2) / lxx)
    S_square_Y_mean = MSE / m + S_square_Y
    return np.power(S_square_Y_mean, 0.5)


def get_SS(X, Y, coef, bias):
    n = len(X)
    Y_hat = C1.predict(X, coef, bias)
    SSE = C1.get_SSE(Y, Y_hat)
    Y_square = np.power(Y, 2)
    #print(Y_square)
    SST = sum(Y_square) - n * np.power(np.average(Y), 2)
    dev = X - np.average(X)
    SSR = np.power(coef, 2) * np.dot(dev, dev.T)
    print("SST: %i" % SST, "   SSE + SSR = %i" % (SSE + SSR))
    ans = []
    ans.append(SST)
    ans.append(SSR)
    ans.append(SSE)
    return ans


def CI_EY(X, MSE, X_hat, coef, bias, alpha):
    n = len(X)
    Y_hat_E = C1.predict(X_hat, coef, bias)
    # print(Y_hat_E)
    t = st.t.ppf(1 - alpha/2, n - 2)
    S_Y_hat = get_S_Y_hat(X, MSE, X_hat)
    # print(S_Y_hat)
    dif = S_Y_hat * t
    low_bound = Y_hat_E - dif
    high_bound = Y_hat_E + dif
    ans = []
    ans.append(low_bound)
    ans.append(high_bound)
    return ans


def CI_Predict_mean_Y(X, m, MSE, X_hat, coef, bias, alpha):
    n = len(X)
    Y_hat_E = C1.predict(X_hat, coef, bias)
    # print(Y_hat_E)
    t = st.t.ppf(1 - alpha/2, n - 2)
    S_Y_hat_mean = get_S_Y_mean_hat(X, m, MSE, X_hat)
    # print(S_Y_hat)
    dif = S_Y_hat_mean * t
    low_bound = Y_hat_E - dif
    high_bound = Y_hat_E + dif
    ans = []
    ans.append(low_bound)
    ans.append(high_bound)
    return ans


def get_W(alpha, n):
    W_square = 2 * st.f.ppf(1 - alpha, 2, n - 2)
    return np.power(W_square, 0.5)


def get_CI_WH_Y(X, alpha, coef, bias, MSE, predict=-1):
    n = len(X)
    Y_low = []
    Y_high = []
    WH = get_W(alpha, n)
    # print(WH)
    if predict != -1:
        x = predict
        Y_hat = C1.predict(x, coef=coef, bias=bias)
        s_Y = get_S_Y_hat(X, MSE, x)
        dif = WH * s_Y
        l = Y_hat - dif
        h = Y_hat + dif
        ans = []
        ans.append(l)
        ans.append(h)
        return ans
    for x in X:
        Y_hat = C1.predict(x, coef=coef, bias=bias)
        s_Y = get_S_Y_hat(X, MSE, x)
        dif = WH * s_Y
        l = Y_hat - dif
        h = Y_hat + dif
        Y_low.append(l)
        Y_high.append(h)
    ans = []
    ans.append(Y_low)
    ans.append(Y_high)
    return ans


def draw_CB_WH(X, alpha, coef, bias, MSE):
    X.sort()
    plt.plot(X, coef * X + bias)
    plt.plot(X, get_CI_WH_Y(X, alpha, coef, bias, MSE)[0])
    plt.plot(X, get_CI_WH_Y(X, alpha, coef, bias, MSE)[1])
    plt.show()


def F_test(X, coef, MSE, alpha):
    n = len(X)
    s_coef = get_S_coef(X, MSE)
    F_t = np.power(coef / s_coef, 2)
    F = st.f.ppf(1 - alpha, 1, n - 2)
    print("The F estimator is: %.1f" % F_t, "    The F_test value is : %.2f" % F)
    if F <= F_t:
        return "Ref"
    else:
        return "Acc"


def get_r(X, Y):
    devia_x = X - np.average(X)
    devia_y = Y - np.average(Y)
    lxy = np.dot(devia_x, devia_y.T)
    lxx = np.dot(devia_x, devia_x.T)
    lyy = np.dot(devia_y, devia_y.T)
    return lxy / np.power(lxx * lyy, 0.5)


def r_t_test(r, alpha, n):
    t = st.t.ppf(1 - alpha / 2, n - 2)
    t_test = abs(r * np.power(n - 2, 0.5) / np.power(1 - np.power(r, 2), 0.5))
    print("t* = %.2f" % t_test, "  t_test =  ", t)
    if t_test > t:
        return "Ref"
    else:
        return "Acc"

def main():
    X, Y = C1.load('toluca.txt')
    coef, bias = C1.LSE_linear_regression(X, Y)
    Y_hat = C1.predict(X, coef, bias)
    SSE = C1.get_SSE(Y, Y_hat)
    MSE = SSE / (len(X) - 2)
    S_coef = get_S_coef(X, MSE)
    #print(coef, S_coef)
    ans_two = Two_Sided_Test(coef, S_coef, 0.05, len(X))
    print("test_two_side", ans_two)
    ans_one = One_Sided_Test(coef, S_coef, 0.05, len(X))
    print("test_one_side", ans_one)
    CI_Bias = CI_bias(X, MSE, bias, 0.1)
    print("CI_bias: ", CI_Bias)
    X_pre = 100
    CI_Y_hat = CI_EY(X, MSE, X_pre, coef, bias, 0.1)
    print("X= 65 ", "CI_Y= ", CI_EY(X, MSE, 65, coef, bias, 0.1))
    print("X= %i " % X_pre, "CI_Y= ", CI_Y_hat)
    X_pre = 100
    m = 3
    CI_Y_hat_mean = CI_Predict_mean_Y(X, m, MSE, X_pre, coef, bias, 0.1)
    print("X= %i " % X_pre, "m = ", m, "CI_Y= ", CI_Y_hat_mean)
    x = 100
    ans = get_CI_WH_Y(X, 0.1, coef, bias, MSE, predict=x)
    print("X= %i " % x, "CI_Y_WH= ", ans)
    #draw_CB_WH(X, 0.1, coef, bias, MSE)
    ans = F_test(X, coef, MSE, 0.05)
    print("F test: ", ans)
    SS_ = get_SS(X, Y, coef, bias)
    print("R_square = %.3f" % (SS_[1] / SS_[0]))
    print("R = %.3f" % np.power((SS_[1] / SS_[0]), 0.5))
    X, Y = C1.load('CH02TA04.txt')
    r = get_r(X, Y)
    print("related r = ", r)
    ans = r_t_test(r, 0.01, len(X))
    print("T_r test: ", ans)


if __name__ == "__main__":
    main()
