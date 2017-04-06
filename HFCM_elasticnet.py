import numpy as np
import scipy as si
import matplotlib.pyplot as plt
import pandas as pd
# import statsmodels.api as sm
import FuzzyCluster as fc
from FCMs import *
# from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn import linear_model
from sklearn.linear_model import ridge_regression
import lightning.regression as rg




def splitData(dataset, ratio=0.85):
    len_train_data = np.int(len(dataset) * ratio)
    return dataset[:len_train_data], dataset[len_train_data:]


# form feature matrix from sequence
def create_dataset(seq, belta, Order, current_node):
    Nc, K = seq.shape
    samples = np.zeros(shape=(K, Order * Nc + 2))
    for m in range(Order, K):
        for n_idx in range(Nc):
            for order in range(Order):
                samples[m - Order, n_idx * Order + order + 1] = seq[n_idx, m - order]
        samples[m - Order, 0] = 1
        samples[m - Order, -1] = reverseFunc(seq[current_node, m], belta)
    return samples


def predict(samples, weight, steepness, belta):
    # samples: each row is a sample, each column is one feature
    K, _ = samples.shape
    predicted_data = np.zeros(shape=(1, K))
    for t in range(K):
        features = samples[t, :-1]
        predicted_data[0, t] = steepness * np.dot(weight, features)
        predicted_data[0, t] = transferFunc(predicted_data[0, t], belta)
    return predicted_data


def main():
    # load time series data

    # data set 2: TAIEX
    TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')
    dataset = TAIEX.values.flatten()
    # data set 3: sunspot
    # dataset = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()[:, 1]
    # series = pd.read_csv('sunspot.csv', delimiter=';').values[:, 1]
    # df = pd.DataFrame(series)
    # result = seasonal_decompose(series, model='additive', freq=12)
    # dataset = result.resid[6:-7]
    # normalize data into [0, 1]
    # dateset 5: seasonal
    # dataset = pd.read_csv('Seasonal10-9.csv').as_matrix()[:, 0]

    # dataset 6: sales of shampoo over a three year
    # dataset = pd.read_csv('sales-of-shampoo-over-a-three-ye.csv', skiprows=1).as_matrix()[:, 1]
    # detrending by difference time series
    # if np.all(dataset > 0):
    #     dataset = np.diff(np.log(dataset), 1)
    # else:
    #     dataset = np.diff(dataset, 1)
    # dataset = np.linspace(0, 5, 100)
    # dataset = [i**2 for i in dataset]

    dataset = np.diff(dataset, 1)
    minV = np.min(dataset)
    maxV = np.max(dataset)
    dataset = (dataset - minV) / (maxV - minV)
    # dataset = 1 - dataset
    # # plot data
    # plt.plot(data)
    # plt.title('sunspot time series')
    # plt.xlabel('year')
    # plt.show()

    # number of nodes
    Nc = 10
    # order of HFCMs
    Order = 4
    # steepness of sigmoid function
    belta = 1
    # range of weights
    lowBund = -1
    uppBnd = 1

    # partition dataset into train set and test set
    ratio = 0.7
    if len(dataset) > 2 * Order / (1 - ratio):
        train_data, test_data = splitData(dataset, ratio)
    else:
        train_data = dataset[:]
        test_data = dataset[:]
    len_train_data = len(train_data)
    len_test_data = len(test_data)

    U_train, center = fc.fcm(train_data, Nc)
    # calculate membership of test data to the
    # prototype obtained in training state
    # center = np.linspace(0.8, 0.8, Nc)
    # expo = 2
    # dist = np.zeros(shape=(Nc, len_train_data))
    # for i in range(Nc):
    #     for j in range(len_train_data):
    #         dist[i, j] = np.sqrt(np.power(center[i] - train_data[j], 2))
    # tmp = np.power(dist, -2 / (expo - 1))
    # U_train = tmp / np.sum(tmp, axis=0)
    # nvar = len_train_data - Order
    # 30 independent runs
    nTotalRun = 1
    # reg = rg.SDCARegressor(loss='absolute', alpha=0.0001, l1_ratio=1)
    # reg = linear_model.LinearRegression(fit_intercept=False)
    # reg = rg.FistaRegressor(penalty='l1')
    reg = linear_model.LinearRegression()  # regressor()
    for nrun in range(nTotalRun):
        # enet = ElasticNet(alpha=.1, l1_ratio=0.7)
        # from sklearn.linear_model import Lasso

        # alpha = 0.001
        # lasso = Lasso(alpha=alpha)

        # # ElasticNet
        # from sklearn.linear_model import ElasticNet
        #
        # enet = ElasticNet(alpha=alpha, l1_ratio=0.3)

        # solving Ax = b to obtain x(x is the weight vector corresponding to certain node)
        # A = np.zeros(shape=(nvar, Nc * Order))
        # b = np.zeros(shape=(nvar, 1))
        # learned weight matrix
        W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
        samples_train = {}
        for node_solved in range(Nc):  # solve each node in turn
            samples = create_dataset(U_train, belta, Order, node_solved)
            # delete last "Order" rows (all zeros)
            samples_train[node_solved] = samples[:-Order, :]
            # enet.fit(samples[:, :-1], samples[:, -1])
            # W_learned[node_solved, :] = enet.coef_
            reg.fit(samples[:, :-1], samples[:, -1])
            W_learned[node_solved, :] = reg.coef_
            # W_learned[node_solved, :] = lasso.coef_
        steepness = np.max(np.abs(W_learned), axis=1)
        for i in range(Nc):
            if steepness[i] > 1:
                W_learned[i, :] /= steepness[i]
        # print(W_learned)

        # predict on training data set
        trainPredict = np.zeros(shape=(Nc, len_train_data - Order))
        for i in range(Nc):
            trainPredict[i, :] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)
        # fig1 = plt.figure()
        # ax1 = fig1.add_subplot(211)
        # # fig1.hold()
        # for i in range(Nc):
        #     ax1.plot(U_train[i, Order:])
        #
        # ax1.set_xlabel('n')
        # ax1.set_title('Membership of train data')
        # ax2 = fig1.add_subplot(212)
        # for i in range(Nc):
        #     ax2.plot(trainPredict[i, :])
        # ax2.set_xlabel('n')
        # ax2.set_title('Membership of predicted train data')
        # fig1.tight_layout()
        # plt.show()


        # reconstruct part
        # use train data to obtain optimal weights of all nodes to A_{new}
        _, len_trainPredict = trainPredict.shape
        A_train = np.zeros(shape=(len_trainPredict - Order, Order * Nc + 1))
        b_train = np.zeros(shape=(len_trainPredict - Order, 1))

        for m in range(Order, len_trainPredict):  # iterate each sample
            for n_idx in range(Nc):
                for order in range(Order):
                    A_train[m-Order, n_idx * Order + order + 1] = U_train[n_idx, m + Order - order]  # trainPredict[n_idx, m-order]
            b_train[m-Order, 0] = reverseFunc(train_data[m+Order], belta)
        A_train[:, 0] = 1
        # weights of node A_{new} of train data
        # reg.fit(A_train, b_train)
        # x_new = reg.coef_
        reg_new = rg.FistaRegressor()
        x_new = reg_new.fit(A_train, b_train).coef_
        # enet.fit(A_train, b_train)
        # x_new = enet.coef_
        # calculate errors in train data
        new_trainPredict = np.matrix(A_train) * np.matrix(x_new).T
        # need to be transformed by sigmoid function
        for i in range(len(new_trainPredict)):
            new_trainPredict[i] = transferFunc(new_trainPredict[i], belta)

        # plot train data series and predicted train data series
        fig2 = plt.figure()
        ax_2 = fig2.add_subplot(111)
        ax_2.plot(train_data[2 * Order:], 'ro--', label='the original data')
        ax_2.plot(new_trainPredict, 'g+-', label='the predicted data')
        ax_2.set_xlabel('Year')
        ax_2.set_title('time series(train dataset)')
        ax_2.legend()


        # test data
        # calculate membership of test data to the
        # prototype obtained in training state
        expo = 2
        dist = np.zeros(shape=(Nc, len_test_data))
        for i in range(Nc):
            for j in range(len_test_data):
                dist[i, j] = np.sqrt(np.power(center[i] - test_data[j], 2))
        tmp = np.power(dist, -2/(expo-1))
        U_test = tmp / np.sum(tmp, axis=0)
        # U_test, _ = fc.fcm(test_data, Nc)

        testPredict = np.zeros(shape=(Nc, len_test_data - Order))
        samples_test = {}
        for i in range(Nc):  # solve each node in turn
            samples = create_dataset(U_test, belta, Order, i)
            samples_test[i] = samples[:-Order, :]  # delete the last "Order' rows(all zeros)
            testPredict[i, :] = predict(samples_test[i], W_learned[i, :], steepness[i], belta)


        # fig3 = plt.figure()
        # ax31 = fig3.add_subplot(211)
        # for i in range(Nc):
        #     ax31.plot(U_test[i, Order:])
        # ax31.set_xlabel('n')
        # ax31.set_title('Membership of test data')
        #
        # ax32 = fig3.add_subplot(212)
        # for i in range(Nc):
        #     ax32.plot(testPredict[i, :])
        # ax32.set_xlabel('n')
        # ax32.set_title('Membership of predicted test data')
        # fig3.tight_layout()

        # transform predicted train data set into real space
        A_test = np.zeros(shape=(len_test_data - 2*Order, Order*Nc + 1))
        for m in range(Order,  len_test_data-Order):  # iterate each test sample
            for n_idx in range(Nc):
                for order in range(Order):
                    A_test[m-Order, n_idx*Order+order+1] = testPredict[n_idx, m-order]
        A_test[:, 0] = 1
        # prediction on test data
        new_testPredict = np.matrix(A_test) * np.matrix(x_new).T
        for i in range(len(new_testPredict)):
            new_testPredict[i] = transferFunc(new_testPredict[i], belta)

        fig4 = plt.figure()
        ax41 = fig4.add_subplot(111)
        ax41.plot(np.array(test_data[2*Order:]), 'ro--', label='the origindal data')
        ax41.plot(np.array(new_testPredict), 'g+-', label='the predicted data')
        ax41.set_ylim([0, 1])
        ax41.set_xlabel('Year')
        ax41.set_title('time series(test dataset)')
        ax41.legend()
        print("steepness are")
        print(steepness)
        plt.show()
        print('Waiting for debug')





if __name__ == '__main__':
    # earthTest()
    # TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')
    # dataset = TAIEX.parse('clean_v1_2000')
    main()
    # dta = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()[:, 1]

    # data = sm.datasets.co2.load_pandas().data
    # # deal with missing values. see issue
    # # dta.co2.interpolate(inplace=True)
    #
    # res = sm.tsa.seasonal_decompose(dta)
    # resplot = res.plot()
    # plt.show()
