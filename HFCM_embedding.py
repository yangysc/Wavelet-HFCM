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
                samples[m - Order, n_idx * Order + order + 1] = seq[n_idx, m - 1 - order]
        samples[m - Order, 0] = 1
        samples[m - Order, -1] = reverseFunc(seq[current_node, m], belta)
    return samples


def predict(samples, weight, steepness, belta):
    # samples: each row is a sample, each column is one feature
    K, _ = samples.shape
    predicted_data = np.zeros(shape=(1, K))
    for t in range(K):
        features = samples[t, :-1]
        predicted_data[0, t] = transferFunc(steepness * np.dot(weight, features), belta)
    return predicted_data


# normalize data set into [0, 1] or [-1, 1]
def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:   # 2-D
        Nc = len(data)
        minV = np.zeros(shape=Nc)
        maxV= np.zeros(shape=Nc)
        for i in range(Nc):
            minV[i] = np.min(data[i, :])
            maxV[i] = np.max(data[i, :])
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':   # normalize to [0, 1]
                    data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
    else:   # 1D
        minV = np.min(data)
        maxV = np.max(data)
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = (data - minV) / (maxV - minV)
            else:
                data = 2 * (data - minV) / (maxV - minV) - 1
    return data, maxV, minV


# re-normalize data set from [0, 1] or [-1, 1] into its true dimension
def re_normalize(ori_data, maxV, minV, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:  # 2-D
        Nc = len(data)
        for i in range(Nc):
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':   # normalize to [0, 1]
                    data[i, :] = data[i, :] * (maxV[i] - minV[i]) + minV[i]
                else:
                    data[i, :] = (data[i, :] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
    else:  # 1-D
        if np.abs(maxV - minV) > 0.00001:
            if flag == '01':  # normalize to [0, 1]
                data = data * (maxV - minV) + minV
            else:
                data = (data + 1) * (maxV - minV) / 2 + minV
    return data


def HFCM_ridge(dataset, ratio=0.7, plot_flag=False):
    from oct2py import octave
    octave.addpath('/home/shanchao/octave/ltfat-2.2.0/wavelets')
    octave.addpath('/home/shanchao/octave/ltfat-2.2.0/comp')
    octave.addpath('/home/shanchao/octave/ltfat-2.2.0')
    octave.addpath('/home/shanchao/octave/ltfat-2.2.0/x86_64-pc-linux-gnu-api-v50+')

    # dataset = pd.read_csv('AirPassengers.csv', delimiter=',').as_matrix()[:, 2]
    normalize_style = '-1'
    dataset, maxV, minV = normalize(dataset, normalize_style)

    # number of nodes

    # order of HFCMs
    Order = 8
    # steepness of sigmoid function
    belta = 1

    # partition dataset into train set and test set\
    if len(dataset) > 30:
        # ratio = 0.83
        train_data, test_data = splitData(dataset, ratio)
    else:
        train_data, test_data = splitData(dataset, 1)
        test_data = train_data

    len_train_data = len(train_data)
    len_test_data = len(test_data)
    # max_level = Nc - 1
    # wavelet_type = 'db2'
    import pyeemd
    U_train = pyeemd.ceemdan(train_data)  # , S_number=5, num_siftings=10)
    # U_train = pyeemd.ceemdan(train_data, S_number=4, num_siftings=50)
    Nc = len(U_train)
    # normalize wavelet into [0, 1]
    U_train, maxV_train_wavelet, minV_train_wavelet = normalize(U_train, normalize_style)
    # 30 independent runs
    nTotalRun = 1

    for nrun in range(nTotalRun):
        from sklearn.linear_model import ElasticNet
        # clf = ElasticNet(alpha=.1, l1_ratio=0.001)
        # from sklearn import linear_model
        # clf = linear_model.LinearRegression(fit_intercept=False)

        import lightning.regression as rg
        alpha = 1e-3
        eta_svrg = 1e-2
        tol = 1e-24
        start = time.time()
        # from sklearn.linear_model import Ridge
        # clf = Ridge(alpha=1)
        clf = rg.SVRGRegressor(alpha=alpha, eta=eta_svrg,
                               n_inner=1, max_iter=100, tol=tol)
        # clf = rg.SAGRegressor(eta='auto', alpha=1.0, beta=0.0, loss='smooth_hinge', penalty=None, gamma=1.0, max_iter=100,
        #              n_inner=1.0, tol=tol, verbose=0, callback=None, random_state=None)
        # clf = rg.SDCARegressor(alpha=alpha,
        #               max_iter=500, n_calls=len_train_data, tol=tol)
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
        if plot_flag:
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
        trainPredict = re_normalize(trainPredict, maxV_train_wavelet, minV_train_wavelet, normalize_style)
        # for i in range(Nc):
        #     if np.abs(maxV_train_wavelet[i, 0] - minV_train_wavelet[i, 0]) > 0.00001:
        #         trainPredict[i, :] =  (1 + trainPredict[i, :] * (
        #             maxV_train_wavelet[i, 0] - minV_train_wavelet[i, 0]) + minV_train_wavelet[i, 0]) / 2
        #         U_train[i, :] = U_train[i, :] * (
        #             maxV_train_wavelet[i, 0] - minV_train_wavelet[i, 0]) + minV_train_wavelet[i, 0]
        # new_trainPredict = octave.iufwt(U_train[:, :].transpose(), wavelet_type, max_level)
        new_trainPredict = np.sum(trainPredict, 0)
        if plot_flag:
            # plot train data series and predicted train data series
            fig2 = plt.figure()
            ax_2 = fig2.add_subplot(111)
            ax_2.plot(train_data[Order:], 'ro--', label='the original data')
            ax_2.plot(new_trainPredict[Order:], 'g+-', label='the predicted data')
            ax_2.set_xlabel('Year')
            ax_2.set_title('time series(train dataset) by wavelet')
            ax_2.legend()
        plt.show()

        # test data
        # U_test = pyeemd.ceemdan(test_data, S_number=4, num_siftings=50)
        #
        # U_test, maxV_test_wavelet, minV_test_wavelet = normalize(U_test, normalize_style)
        # # maxV_test_wavelet = np.zeros(shape=(Nc, 1))
        # # for i in range(Nc):
        # #     minV_test_wavelet[i, 0] = np.min(U_test[i, :])
        # #     maxV_test_wavelet[i, 0] = np.max(U_test[i, :])
        # #     if np.abs(maxV_test_wavelet[i, 0] - minV_test_wavelet[i, 0]) > 0.00001:
        # #         U_test[i, :] = 2 * (U_test[i, :] - minV_test_wavelet[i, 0]) * scale_factor / (
        # #         maxV_test_wavelet[i, 0] - minV_test_wavelet[i, 0]) - 1
        # testPredict = np.zeros(shape=(Nc, len_test_data))
        # samples_test = {}
        # for i in range(Nc):  # solve each node in turn
        #     samples = create_dataset(U_test, belta, Order, i)
        #     samples_test[i] = samples[:-Order, :]  # delete the last "Order' rows(all zeros)
        #     testPredict[i, :Order] = U_test[i, :Order]
        #     testPredict[i, Order:] = predict(samples_test[i], W_learned[i, :], steepness[i], belta)
        # if plot_flag:
        #     fig3 = plt.figure()
        #     ax31 = fig3.add_subplot(211)
        #     for i in range(Nc):
        #         ax31.plot(U_test[i, Order:])
        #     ax31.set_xlabel('n')
        #     ax31.set_title('Wavelets of test data')
        #
        #     ax32 = fig3.add_subplot(212)
        #     for i in range(Nc):
        #         ax32.plot(testPredict[i, Order:])
        #     ax32.set_xlabel('n')
        #     ax32.set_title('Wavelets of predicted test data')
        #     fig3.tight_layout()
        #
        # # re-normalize wavelet from [0,1] into real dimension
        # testPredict = re_normalize(testPredict, maxV_test_wavelet, minV_test_wavelet, normalize_style)
        # # for i in range(Nc):
        # #     if np.abs(maxV_test_wavelet[i, 0] - minV_test_wavelet[i, 0]) > 0.00001:
        # #         testPredict[i, :] = (1 + testPredict[i, :] * (
        # #             maxV_test_wavelet[i, 0] - minV_test_wavelet[i, 0]) + minV_test_wavelet[i, 0]) / 2
        # new_testPredict = np.sum(testPredict, 0)
        #
        # if plot_flag:
        #     fig4 = plt.figure()
        #     ax41 = fig4.add_subplot(111)
        #     ax41.plot(np.array(test_data[Order:]), 'ro--', label='the origindal data')
        #     ax41.plot(np.array(new_testPredict[Order:]), 'g+-', label='the predicted data')
        #     ax41.set_ylim([0, 1])
        #     ax41.set_xlabel('Year')
        #     ax41.set_title('time series(test dataset) by wavelet')
        #     ax41.legend()
        #     print(steepness)
        #     plt.show()
        # if len(dataset) > 30:
        #     data_predicted = np.hstack((new_trainPredict[:, 0], new_testPredict[:, 0]))
        # else:
        #     data_predicted = new_testPredict[:, 0]
        # data_predicted = re_normalize(data_predicted, maxV, minV, normalize_style)
        # # re-normalize predicted data
        # return data_predicted


def main():
    # load time series data
    # data set 1: enrollment
    # dataset = np.array([13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807,
    #                     16919, 16388, 15433, 15497, 15145,15163, 15984, 16859, 18150, 18970,
    #                     19328, 19337, 18876])
    # time = range(len(dataset))
    # #
    # data set 2: TAIEX(use 70% data as train data)
    # TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')
    # dataset = TAIEX.values.flatten()
    # time = range(len(dataset))
    # data set 3: sunspot
    # sunspot = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()
    # dataset = sunspot[:-1, 1]
    # time = sunspot[:-1, 0]
    # # data set 4 : MG chaos( even use 10% data as train data)
    # import scipy.io as sio
    # dataset = sio.loadmat('MG_chaos.mat')['dataset'].flatten()
    # # only use data from t=124 : t=1123  (all data previous are not in the same pattern!)
    # dataset = dataset[123:1123]

    # dataset = dataset[118:1118]
    # time = range(len(dataset))
    # plt.plot(dataset[500:], 'ob')
    # plt.ylim([-0.1, 1.38])
    # plt.xlim([0, 500])
    # plt.show()

    '''
    trend data
    '''
    # linear dataset
    # dataset = list(range(500))
    # square dataset
    # dataset = [i for i in np.linspace(0, 10, 100)]
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')
    dataset = pd.read_csv('monthly-electricity-production-i.csv', delimiter=';', parse_dates=[0], date_parser=dateparse).as_matrix()
    time = dataset[:, 0]
    dataset = np.array(dataset[:, 1], dtype=np.float)
    HFCM_ridge(dataset, ratio=0.7, plot_flag=True)
    # data_predicted = HFCM_ridge(dataset, ratio=0.7)
    # mse, rmse = statistics(dataset, data_predicted)
    # print('MSE is %f. RMSE is %f' % (mse, rmse))
    #
    # fig4 = plt.figure()
    # ax41 = fig4.add_subplot(111)
    #
    # ax41.plot(time, dataset, 'ro--', label='the original data')
    # ax41.plot(time, data_predicted, 'g+-', label='the predicted data')
    # # ax41.set_ylim([0, 1])
    # ax41.set_xlabel('Year')
    # ax41.set_title('time series prediction ')
    # ax41.legend()
    # plt.show()


def statistics(origin, predicted):
    # # compute RMSE
    # length = len(origin)
    # err = 0
    # for i in range(length):
    #     err = np.power(origin[i] - predicted[i], 2) / (i + 1) + err * i / (i+1)
    # print('mannel err is %f' % (err))
    # err = np.linalg.norm(origin - predicted, 2) / length
    # return err
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(origin, predicted)
    rmse = np.sqrt(mse)
    return mse, rmse


if __name__ == '__main__':
    main()
    # earthTest()
    # TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')
    # dataset = TAIEX.values.flatten()
    # from oct2py import octave
    # octave.addpath('/home/shanchao/octave/ltfat-2.2.0/wavelets')
    # octave.addpath('/home/shanchao/octave/ltfat-2.2.0/comp')
    # octave.addpath('/home/shanchao/octave/ltfat-2.2.0')
    # octave.addpath('/home/shanchao/octave/ltfat-2.2.0/x86_64-pc-linux-gnu-api-v50+')
    #
    # # dataset = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()
    # # dataset = dataset[:-1, 1]
    # dataset = (dataset - min(dataset)) / (max(dataset) - min(dataset))
    # maxLevel = 2
    # Order = 100
    # wavelets = octave.ufwt(dataset, 'db4', maxLevel)
    # reverse_dataset = octave.iufwt(wavelets[:Order, :], 'db4', maxLevel).flatten()
    # plt.plot(dataset[:Order], '-*')
    # plt.plot(reverse_dataset, '--o')
    # print(np.linalg.norm(dataset[:Order] - reverse_dataset))
    # plt.show()

