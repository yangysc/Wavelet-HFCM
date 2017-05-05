import numpy as np
import matplotlib.pyplot as plt
from FCMs import transferFunc, reverseFunc
import pandas as pd
import time

def splitData(dataset, ratio=0.85):
    len_train_data = int(len(dataset) * ratio)
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
    from oct2py import octave
    import scipy.io as sio
    octave.addpath('/home/shanchao/octave/ltfat-2.2.0/wavelets')
    octave.addpath('/home/shanchao/octave/ltfat-2.2.0/comp')
    octave.addpath('/home/shanchao/octave/ltfat-2.2.0')
    octave.addpath('/home/shanchao/octave/ltfat-2.2.0/x86_64-pc-linux-gnu-api-v50+')
    # load time series data

    # data set 2: TAIEX
    TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')
    dataset = TAIEX.values.flatten()

    # data set 3: sunspot
    # dataset = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()[:, 1]

    # data set 4 : MG chaos
    # dataset = sio.loadmat('MG_chaos.mat')['dataset']
    # # only use data from t=124 : t=1123  (all data previous are not in the same pattern!)
    # dataset = dataset[123:1122]
    dataset = list(range(500))
    # dataset = pd.read_csv('AirPassengers.csv', delimiter=',').as_matrix()[:, 2]


    minV = np.min(dataset)
    maxV = np.max(dataset)
    dataset = (dataset - minV) / (maxV - minV)

    # number of nodes
    Nc = int(np.floor(np.log2(len(dataset)))) - 1
    # order of HFCMs
    Order = 4
    # steepness of sigmoid function
    belta = 1

    # partition dataset into train set and test set
    ratio = 0.7
    if len(dataset) > 2 * Order / (1 - ratio):
        train_data, test_data = splitData(dataset, ratio)
    else:
        train_data = dataset[:]
        test_data = dataset[:]
    len_train_data = len(train_data)
    len_test_data = len(test_data)
    max_level = Nc - 1
    wavelet_type = 'db1'
    wavelet_coffis = octave.ufwt(dataset, wavelet_type, max_level).transpose()
    U_train = wavelet_coffis[:, :len_train_data]
    U_test = wavelet_coffis[:, len_train_data:]
    # U_train = octave.ufwt(train_data, wavelet_type, max_level).transpose()

    # normalize wavelet into [0, 1]
    minV_train_wavelet = np.zeros(shape=(Nc, 1))
    maxV_train_wavelet = np.zeros(shape=(Nc, 1))
    for i in range(Nc):
        minV_train_wavelet[i, 0] = np.min(U_train[i, :])
        maxV_train_wavelet[i, 0] = np.max(U_train[i, :])
        if np.abs(maxV_train_wavelet[i, 0] - minV_train_wavelet[i, 0]) > 0.00001:
            U_train[i, :] = (U_train[i, :] - minV_train_wavelet[i, 0]) / (maxV_train_wavelet[i, 0] - minV_train_wavelet[i, 0])

    # 30 independent runs
    nTotalRun = 1

    for nrun in range(nTotalRun):
        # from sklearn.linear_model import ElasticNet
        # clf = ElasticNet(alpha=.01, l1_ratio=0.01)
        # reg = linear_model.LinearRegression(fit_intercept=False)

        import lightning.regression as rg
        alpha = 1e-2
        eta_svrg = 1e-2
        tol = 1e-24
        start = time.time()
        from sklearn.linear_model import Ridge
        # clf = Ridge(alpha=0.5)
        clf = rg.SVRGRegressor(alpha=alpha, eta=eta_svrg,
                      n_inner=1.0, max_iter=100, random_state=0, tol=tol)

        # clf = rg.SDCARegressor(alpha=alpha,
        #               max_iter=100, n_calls=len_train_data/2, random_state=0, tol=tol)
        # solving Ax = b to obtain x(x is the weight vector corresponding to certain node)

        # learned weight matrix
        W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
        samples_train = {}
        for node_solved in range(Nc):  # solve each node in turn
            samples = create_dataset(U_train, belta, Order, node_solved)
            # delete last "Order" rows (all zeros)
            samples_train[node_solved] = samples[:-Order, :]
            # reg.fit(samples[:, :-1], samples[:, -1])
            # W_learned[node_solved, :] = reg.coef_
            # use ridge regression
            clf.fit(samples[:, :-1], samples[:, -1])
            W_learned[node_solved, :] = clf.coef_
        end_time = time.time()
        print("solving L2 using %f(s) time" % (end_time - start))
        steepness = np.max(np.abs(W_learned), axis=1)
        for i in range(Nc):
            if steepness[i] > 1:
                W_learned[i, :] /= steepness[i]
        # print(W_learned)

        # predict on training data set
        trainPredict = np.zeros(shape=(Nc, len_train_data))
        for i in range(Nc):
            trainPredict[i, :Order] = U_train[i, :Order]
            trainPredict[i, Order:] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(211)
        # fig1.hold()
        for i in range(Nc):
            ax1.plot(U_train[i, Order:])

        ax1.set_xlabel('n')
        ax1.set_title('Wavelets of train data')
        ax2 = fig1.add_subplot(212)
        for i in range(Nc):
            ax2.plot(trainPredict[i, :])
        ax2.set_xlabel('n')
        ax2.set_title('Wavelets of predicted train data')
        fig1.tight_layout()
        # plt.show()

        # re-normalize wavelet from [0,1] into real dimension
        for i in range(Nc):
            if np.abs(maxV_train_wavelet[i, 0] - minV_train_wavelet[i, 0]) > 0.00001:
                trainPredict[i, :] = trainPredict[i, :] * (
                    maxV_train_wavelet[i, 0] - minV_train_wavelet[i, 0]) + minV_train_wavelet[i, 0]
                U_train[i, :] = U_train[i, :] * (
                    maxV_train_wavelet[i, 0] - minV_train_wavelet[i, 0]) + minV_train_wavelet[i, 0]
        # new_trainPredict = octave.iufwt(U_train[:, Order:].transpose(), wavelet_type, max_level)
        new_trainPredict = octave.iufwt(trainPredict[:, :].transpose(), wavelet_type, max_level)

        # plot train data series and predicted train data series
        fig2 = plt.figure()
        ax_2 = fig2.add_subplot(111)
        ax_2.plot(train_data[Order:], 'ro--', label='the original data')
        ax_2.plot(new_trainPredict[Order:], 'g+-', label='the predicted data')
        ax_2.set_xlabel('Year')
        ax_2.set_title('time series(train dataset) by wavelet')
        ax_2.legend()
        # plt.show()

        # test data
        # U_test = octave.ufwt(test_data, wavelet_type, max_level).transpose()
        minV_test_wavelet = np.zeros(shape=(Nc, 1))
        maxV_test_wavelet = np.zeros(shape=(Nc, 1))
        for i in range(Nc):
            minV_test_wavelet[i, 0] = np.min(U_test[i, :])
            maxV_test_wavelet[i, 0] = np.max(U_test[i, :])
            if np.abs(maxV_test_wavelet[i, 0] - minV_test_wavelet[i, 0]) > 0.00001:
                U_test[i, :] = (U_test[i, :] - minV_test_wavelet[i, 0]) / (maxV_test_wavelet[i, 0] - minV_test_wavelet[i, 0])
        testPredict = np.zeros(shape=(Nc, len_test_data))
        samples_test = {}
        for i in range(Nc):  # solve each node in turn
            samples = create_dataset(U_test, belta, Order, i)
            samples_test[i] = samples[:-Order, :]  # delete the last "Order' rows(all zeros)
            testPredict[i, :Order] = U_test[i, :Order]
            testPredict[i, Order:] = predict(samples_test[i], W_learned[i, :], steepness[i], belta)

        fig3 = plt.figure()
        ax31 = fig3.add_subplot(211)
        for i in range(Nc):
            ax31.plot(U_test[i, Order:])
        ax31.set_xlabel('n')
        ax31.set_title('Wavelets of test data')

        ax32 = fig3.add_subplot(212)
        for i in range(Nc):
            ax32.plot(testPredict[i, Order:])
        ax32.set_xlabel('n')
        ax32.set_title('Wavelets of predicted test data')
        fig3.tight_layout()

        # re-normalize wavelet from [0,1] into real dimension
        for i in range(Nc):
            if np.abs(maxV_test_wavelet[i, 0] - minV_test_wavelet[i, 0]) > 0.00001:
                testPredict[i, :] = testPredict[i, :] * (
                    maxV_test_wavelet[i, 0] - minV_test_wavelet[i, 0]) + minV_test_wavelet[i, 0]
        new_testPredict = octave.iufwt(testPredict.transpose(), wavelet_type, max_level)

        fig4 = plt.figure()
        ax41 = fig4.add_subplot(111)
        ax41.plot(np.array(test_data[Order:]), 'ro--', label='the origindal data')
        ax41.plot(np.array(new_testPredict[Order:]), 'g+-', label='the predicted data')
        ax41.set_ylim([0, 1])
        ax41.set_xlabel('Year')
        ax41.set_title('time series(test dataset) by wavelet')
        ax41.legend()
        print(steepness)
        plt.show()
        print('Waiting for debug')  #
#
#
# def earthTest():
#     import numpy
#     from pyearth import Earth
#     from matplotlib import pyplot
#
#     # Create some fake data
#     numpy.random.seed(0)
#     m = 1000
#     n = 10
#     X = 80 * numpy.random.uniform(size=(m, n)) - 40
#     y = numpy.abs(X[:, 6] - 4.0) + 1 * numpy.random.normal(size=m)
#
#     # Fit an Earth model
#     model = Earth()
#     x = np.array([i for i in range(len(y))])
#     model.fit(x, y)
#
#     # Print the model
#     print(model.trace())
#     print(model.summary())
#
#     # Plot the model
#     y_hat = model.predict(X)
#     pyplot.figure()
#     pyplot.plot(X[:, 6], y, 'r.')
#     pyplot.plot(X[:, 6], y_hat, 'b.')
#     pyplot.xlabel('x_6')
#     pyplot.ylabel('y')
#     pyplot.title('Simple Earth Example')
#     pyplot.show()


if __name__ == '__main__':
    # earthTest()
    # TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')
    # dataset = TAIEX.parse('clean_v1_2000')
    main()
    # dta = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()[:, 1]
    # import pywt
    #
    # # (cA, cD/) /= pywt.dwt([1, 2, 3, 4, 5, 6], 'db1')
    # # print(cA, cD/)
    # dataset = list(range(8))
    # max_level = pywt.swt_max_level(len(dataset))
    # print("max level is %d" % max_level)
    # cofficients = pywt.swt(dataset, wavelet='db1', level=1)
    # for coff in cofficients:
    #     print(coff)


    # data = sm.datasets.co2.load_pandas().data
    # # deal with missing values. see issue
    # # dta.co2.interpolate(inplace=True)
    #
    # res = sm.tsa.seasonal_decompose(dta)
    # resplot = res.plot()
    # plt.show()
