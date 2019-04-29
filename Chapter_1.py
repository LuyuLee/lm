__author__ = 'Ricardo'


import numpy as np


def load(FileName):
    file = open(FileName, 'r')
    X = []
    Y = []
    for fr in file:
        lines = fr.strip('\n').split()
        # print(lines)
        X.append(float(lines[0]))
        Y.append(float(lines[1]))
    file.close()
    return np.array(X), np.array(Y)


def LSE_linear_regression(X, Y):
    devia_x = X - np.average(X)
    devia_y = Y - np.average(Y)
    lxy = np.dot(devia_x, devia_y.T)
    lxx = np.dot(devia_x, devia_x.T)
    coef = lxy / lxx
    bias = (sum(Y) - coef * (sum(X))) / len(X)
    return coef, bias


def get_SSE(Y_p, Y_hat):
    error = Y_p - Y_hat
    #print(error)
    SSE = np.dot(error, error.T)
    return SSE


def predict(X, coef, bias):
    Y_hat = coef * X + bias
    return Y_hat


def main():
    X, Y = load('toluca.txt')
    # print('X: ', X)
    # print('Y: ', Y)
    coef, bias = LSE_linear_regression(X, Y)
    print("coef: %.2f" % coef, "Bias: %.2f" % bias)
    Y_hat = predict(X, coef, bias)
    SSE = get_SSE(Y, Y_hat)
    MSE = SSE / (len(X) - 2)
    print("SSE: %.0f" % SSE, "MSE: %.0f" % MSE)

if __name__ == '__main__':
    main()




