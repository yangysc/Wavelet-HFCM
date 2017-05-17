import numpy as np
import matplotlib.pyplot as plt
from FCMs import transferFunc, reverseFunc
import pandas as pd
import time
from modwt import modwt


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

# def normalize(ori_data, high=0.9, low=-1, flag='01'):
#     data = ori_data.copy()
#     if len(data.shape) > 1:   # 2-D
#         N , K = data.shape
#         minV = np.zeros(shape=K)
#         maxV = np.zeros(shape=K)
#         for i in range(N):
#             minV[i] = np.min(data[i, :])
#             maxV[i] = np.max(data[i, :])
#             if np.abs(maxV[i] - minV[i]) > 0.00001:
#                 if flag == '01':   # normalize to [0, 1]
#                     data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
#                 else:
#                     data[i, :] = (high - low) * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) + low
#         return data, maxV, minV
#     else:   # 1D
#         minV = np.min(data)
#         maxV = np.max(data)
#         if np.abs(maxV - minV) > 0.00001:
#             if flag == '01':  # normalize to [0, 1]
#                 data = (data - minV) / (maxV - minV)
#             else:
#                 data = (high - low) * (data - minV) / (maxV - minV) + low
#         return data, maxV, minV
#
#
# # re-normalize data set from [0, 1] or [-1, 1] into its true dimension
# def re_normalize(ori_data, maxV, minV, high=1, low=-1, flag='01'):
#     data = ori_data.copy()
#     if len(data.shape) > 1:  # 2-D
#         Nc, K = data.shape
#         for i in range(Nc):
#             if np.abs(maxV[i] - minV[i]) > 0.00001:
#                 if flag == '01':   # normalize to [0, 1]
#                     data[i, :] = data[i, :] * (maxV[i] - minV[i]) + minV[i]
#                 else:
#                     data[i, :] = (data[i, :] - low) * (maxV[i] - minV[i]) / (high - low) + minV[i]
#     else:  # 1-D
#         if np.abs(maxV - minV) > 0.00001:
#             if flag == '01':  # normalize to [0, 1]
#                 data = data * (maxV - minV) + minV
#             else:
#                 data = (data - low) * (maxV - minV) / (high - low) + minV
#     return data


# normalize data set into [0, 1] or [-1, 1]
def normalize(ori_data, flag='01'):
    data = ori_data.copy()
    if len(data.shape) > 1:   # 2-D
        N , K = data.shape
        minV = np.zeros(shape=K)
        maxV = np.zeros(shape=K)
        for i in range(N):
            minV[i] = np.min(data[i, :])
            maxV[i] = np.max(data[i, :])
            if np.abs(maxV[i] - minV[i]) > 0.00001:
                if flag == '01':   # normalize to [0, 1]
                    data[i, :] = (data[i, :] - minV[i]) / (maxV[i] - minV[i])
                else:
                    data[i, :] = 2 * (data[i, :] - minV[i]) / (maxV[i] - minV[i]) - 1
        return data, maxV, minV
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
        Nc, K = data.shape
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


# # normalize data set into [0, 1] or [-1, 1]
# def normalize(ori_data, flag='01'):
#     data = ori_data.copy()
#     if len(data.shape) > 1:   # 2-D
#         _, K = data.shape
#         minV = np.zeros(shape=K)
#         maxV = np.zeros(shape=K)
#         for i in range(K):
#             minV[i] = np.min(data[:, i])
#             maxV[i] = np.max(data[:, i])
#             if np.abs(maxV[i] - minV[i]) > 0.00001:
#                 if flag == '01':   # normalize to [0, 1]
#                     data[:, i] = (data[:, i] - minV[i]) / (maxV[i] - minV[i])
#                 else:
#                     data[:, i] = 2 * (data[:, i] - minV[i]) / (maxV[i] - minV[i]) - 1
#         return data, maxV, minV
#     else:   # 1D
#         minV = np.min(data)
#         maxV = np.max(data)
#         if np.abs(maxV - minV) > 0.00001:
#             if flag == '01':  # normalize to [0, 1]
#                 data = (data - minV) / (maxV - minV)
#             else:
#                 data = 2 * (data - minV) / (maxV - minV) - 1
#         return data, maxV, minV
#
#
# # re-normalize data set from [0, 1] or [-1, 1] into its true dimension
# def re_normalize(ori_data, maxV, minV, flag='01'):
#     data = ori_data.copy()
#     if len(data.shape) > 1:  # 2-D
#         Nc, K = data.shape
#         for i in range(K):
#             if np.abs(maxV[i] - minV[i]) > 0.00001:
#                 if flag == '01':   # normalize to [0, 1]
#                     data[:, i] = data[:, i] * (maxV[i] - minV[i]) + minV[i]
#                 else:
#                     data[:, i] = (data[:, i] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
#     else:  # 1-D
#         if np.abs(maxV - minV) > 0.00001:
#             if flag == '01':  # normalize to [0, 1]
#                 data = data * (maxV - minV) + minV
#             else:
#                 data = (data + 1) * (maxV - minV) / 2 + minV
#     return data


# normalize data set into [0, 1] or [-1, 1]
# def normalize(ori_data, flag='01'):
#     data = ori_data.copy()
#     if len(data.shape) > 1:   # 2-D
#         _, K = data.shape
#         minV = np.zeros(shape=K)
#         maxV= np.zeros(shape=K)
#         for i in range(K):
#             minV[i] = np.min(data[:, i])
#             maxV[i] = np.max(data[:, i])
#             if np.abs(maxV[i] - minV[i]) > 0.00001:
#                 if flag == '01':   # normalize to [0, 1]
#                     data[:, i] = (data[:, i] - minV[i]) / (maxV[i] - minV[i])
#                 else:
#                     data[:, i] = 2 * (data[:, i] - minV[i]) / (maxV[i] - minV[i]) - 1
#     else:   # 1D
#         minV = np.min(data)
#         maxV = np.max(data)
#         if np.abs(maxV - minV) > 0.00001:
#             if flag == '01':  # normalize to [0, 1]
#                 data = (data - minV) / (maxV - minV)
#             else:
#                 data = 2 * (data - minV) / (maxV - minV) - 1
#     return data, maxV, minV
#
#
# # re-normalize data set from [0, 1] or [-1, 1] into its true dimension
# def re_normalize(ori_data, maxV, minV, flag='01'):
#     data = ori_data.copy()
#     if len(data.shape) > 1:  # 2-D
#         Nc, K = data.shape
#         for i in range(K):
#             if np.abs(maxV[i] - minV[i]) > 0.00001:
#                 if flag == '01':   # normalize to [0, 1]
#                     data[:, i] = data[:, i] * (maxV[i] - minV[i]) + minV[i]
#                 else:
#                     data[:, i] = (data[:, i] + 1) * (maxV[i] - minV[i]) / 2 + minV[i]
#     else:  # 1-D
#         if np.abs(maxV - minV) > 0.00001:
#             if flag == '01':  # normalize to [0, 1]
#                 data = data * (maxV - minV) + minV
#             else:
#                 data = (data + 1) * (maxV - minV) / 2 + minV
#     return data

def wavelet_transform(x, J):
    N = len(x)
    C = np.zeros(shape=(J + 1, N))
    # W: wavelet coefficients
    W = np.zeros(shape=(J + 1, N))
    C[0, :] = x.copy()
    for j in range(1, J + 1):
        for k in range(1, N):
            C[j, k] = 1 / 2 * (C[j - 1, k] + C[j - 1, k - np.power(2, j - 1)])
            W[j, k] = C[j - 1][k] - C[j, k]

    W[0, :] = C[J, :]
    return W[:, np.power(2, J):]
    # # calculate MODWT increment
    # # k is start point index
    # # length of time series
    # K = len(x)
    # wavelet_type = filter_type
    # coffis = np.zeros(shape=(J+1, K-k))
    #
    # coffis_dict[k] = modwt(x[: k], wavelet_type, J)  # for the reconstruction of k'th point
    # for i in range(k, K):
    #     temp_coffis = modwt(x[: i+1], wavelet_type, J)
    #     coffis_dict[i+1] = temp_coffis
    #     coffis[:, i-k] = temp_coffis[:, -1]
    # return coffis


def wavelet_reconstruct(predicted_coffis):
    # calculate MODWT increment
    # k is start point index
    return np.sum(predicted_coffis, axis=0)
    # from modwt import imodwt
    # return imodwt(predicted_coffis, filter_type)[start_point:]
    # # length of time series
    # Nc, len_series = predicted_coffis.shape
    # x = np.zeros(shape=(len_series,))
    # wavelet_type = filter_type
    #
    # for i in range(len_series):
    #     temp_x = imodwt(np.hstack((coffis_dict[i + start_point], np.reshape(predicted_coffis[:, i], (Nc, 1)))), wavelet_type)
    #     x[i] = temp_x[-1]
    # return x



def HFCM_ridge(dataset1, ratio=0.7, plot_flag=False):

    # dataset = np.diff(dataset)
    # from modwt import modwt, imodwt
    # dataset = pd.read_csv('AirPassengers.csv', delimiter=',').as_matrix()[:, 2]
    normalize_style = '-01'
    dataset_copy = dataset1.copy()
    dataset, maxV, minV = normalize(dataset1, normalize_style)

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
    # grid search
    # best parameters
    best_Order = -1
    best_Nc = -1
    best_W_learned = None
    best_steepness = None
    best_predict = np.zeros(shape=len_train_data)
    best_alpha = 0
    min_nmse = np.inf
    min_rmse = np.inf
    # from sklearn import linear_model
    for Order in range(2, 10):
        for Nc in range(2, 8):
            for alpha in np.linspace(0.1, 50, 500):
                max_level = Nc - 1
                coffis = wavelet_transform(dataset, max_level)
                coffis, maxV_wavelet, minV_wavelet = normalize(coffis, normalize_style)
                k = 2 ** max_level
                U_train = coffis[:, :len_train_data - k]
                #
                # fig1 = plt.figure()
                # ax1 = fig1.add_subplot(111)
                # # fig1.hold()
                # for i in range(Nc):
                #     ax1.plot(U_train[i, :])
                #
                # ax1.set_xlabel('n')
                # ax1.set_title('Wavelets of train data')
                # plt.show()

                # normalize wavelet into [0, 1]
                # U_train, maxV_train_wavelet, minV_train_wavelet = normalize(U_train, normalize_style)
                # 30 independent runs
                nTotalRun = 1

                for nrun in range(nTotalRun):

                    import lightning.regression as rg
                    # alpha = 1e-2
                    # eta_svrg = 1e-2 * 7  # more small, more focus on test data
                    tol = 1e-24
                    start = time.time()
                    # from sklearn.linear_model import Ridge
                    # clf = Ridge(alpha=0.01, fit_intercept=False)
                    from sklearn import linear_model
                    # clf = linear_model.BayesianRidge(fit_intercept=False)
                    # clf = linear_model.ElasticNet(fit_intercept=False)
                    # clf = rg.SVRGRegressor(alpha=alpha, eta=eta_svrg,
                    #                        n_inner=1, max_iter=100, tol=tol)
                    # clf = rg.SAGRegressor(eta='auto', alpha=1.0, beta=0.0, loss='smooth_hinge', penalty=None, gamma=1.0, max_iter=100,
                    #              n_inner=1.0, tol=tol, verbose=0, callback=None, random_state=None)
                    # clf = rg.SDCARegressor(alpha=alpha,
                    #               max_iter=100, n_calls=len_train_data, tol=tol)
                    # solving Ax = b to obtain x(x is the weight vector corresponding to certain node)

                    # learned weight matrix
                    # clf = linear_model.LinearRegression(fit_intercept=False)
                    clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol)
                    # clf  = linear_model.RANSACRegressor(linear_model.Ridge(alpha=.1, fit_intercept=False))
                    # from sklearn.kernel_ridge import KernelRidge0
                    # clf = KernelRidge(alpha=1.0)
                    W_learned = np.zeros(shape=(Nc, Nc * Order + 1))
                    samples_train = {}
                    for node_solved in range(Nc):  # solve each node in turn
                        samples = create_dataset(U_train, belta, Order, node_solved)
                        # delete last "Order" rows (all zeros)
                        samples_train[node_solved] = samples[:-Order, :]
                        # use ridge regression
                        clf.fit(samples[:, :-1], samples[:, -1])
                        W_learned[node_solved, :] = clf.coef_

                        # W_learned[node_solved, :] = clf.estimator_.coef_
                    # end_time = time.time()
                    # print("solving L2 using %f(s) time" % (end_time - start))
                    steepness = np.max(np.abs(W_learned), axis=1)
                    for i in range(Nc):
                        if steepness[i] > 1:
                            W_learned[i, :] /= steepness[i]
                    # print(W_learned)

                    # predict on training data set
                    trainPredict = np.zeros(shape=(Nc, len_train_data-k))
                    for i in range(Nc):
                        trainPredict[i, :Order] = U_train[i, :Order]
                        trainPredict[i, Order:] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)
                    if plot_flag:
                        fig1 = plt.figure()
                        ax1 = fig1.add_subplot(211)
                        # fig1.hold()
                        for i in range(Nc):
                            ax1.plot(U_train[i, :], label=str(i))

                        ax1.set_xlabel('n')
                        ax1.set_title('Wavelets of train data')
                        ax1.legend()
                        ax2 = fig1.add_subplot(212)
                        for i in range(Nc):
                            ax2.plot(trainPredict[i, :])
                        ax2.set_xlabel('n')
                        ax2.set_title('Wavelets of predicted train data')
                        fig1.tight_layout()
                    # plt.show()

                    # # re-normalize wavelet from [0,1] into real dimension
                    trainPredict = re_normalize(trainPredict, maxV_wavelet, minV_wavelet, normalize_style)

                    # # reconstruct part
                    new_trainPredict = wavelet_reconstruct(trainPredict)
                    new_trainPredict = np.hstack((train_data[:k], new_trainPredict))

                    # print('Error is %f' % np.linalg.norm(np.array(train_data)[k:] - new_trainPredict, 2))
                    if plot_flag:
                        # plot train data series and predicted train data series
                        fig2 = plt.figure()
                        ax_2 = fig2.add_subplot(111)
                        ax_2.plot(train_data, 'ro--', label='the original data')
                        ax_2.plot(new_trainPredict, 'g+-', label='the predicted data')
                        ax_2.set_xlabel('Year')
                        ax_2.set_title('time series(train dataset) by wavelet')
                        ax_2.legend()
                    # fig2 = plt.figure()
                    # ax_2 = fig2.add_subplot(111)
                    # ax_2.plot(train_data[k:], 'ro--', label='the original data')
                    # ax_2.plot(new_trainPredict, 'g+-', label='the predicted data')
                    # ax_2.set_xlabel('Year')
                    # ax_2.set_title('time series(train dataset) by wavelet')
                    # ax_2.legend()
                    # plt.show()
                    # exit(6)

                    mse, rmse, nmse = statistics(dataset[:len_train_data], new_trainPredict)
                    print("Nc -> %d, Order -> %d, alpha -> %f: rmse -> %f, min_rmse is %f" % (Nc, Order, alpha, rmse, min_rmse))
                    # use rmse as performance index
                    if rmse < min_rmse:
                        min_rmse = rmse
                        best_Nc = Nc
                        best_Order = Order
                        best_predict[:] = new_trainPredict
                        best_W_learned = W_learned
                        best_steepness = steepness
                        best_alpha = alpha
    # # test data
    max_level = best_Nc - 1
    coffis = wavelet_transform(dataset, max_level)
    coffis, maxV_wavelet, minV_wavelet = normalize(coffis, normalize_style)
    k = 2 ** max_level
    U_test = coffis[:, len_train_data-k-best_Order:]   # use last Order data point of train dataset
    #  to forecast first Order data point in test dat
    # _, k1 = U_test.shape
    # for i in range(k1):
    #     if np.abs(maxV_wavelet[i] - minV_wavelet[i]) > 0.00001:
    #         U_test[:, i] = 2 * (U_test[:, i] - minV_wavelet[i]) \
    #                       / (maxV_wavelet[i] - minV_wavelet[i]) - 1
    testPredict = np.zeros(shape=(best_Nc, len_test_data))
    samples_test = {}
    for i in range(best_Nc):  # solve each node in turn
        samples = create_dataset(U_test, belta, best_Order, i)
        samples_test[i] = samples[:-best_Order, :]  # delete the last "Order' rows(all zeros)
        # testPredict[i, :Order] = U_test[i, :Order]
        testPredict[i, :] = predict(samples_test[i], best_W_learned[i, :], best_steepness[i], belta)
    if plot_flag:
        fig3 = plt.figure()
        ax31 = fig3.add_subplot(211)
        for i in range(best_Nc):
            ax31.plot(U_test[i, :])
        ax31.set_xlabel('n')
        ax31.set_title('Wavelets of test data')

        ax32 = fig3.add_subplot(212)
        for i in range(best_Nc):
            ax32.plot(testPredict[i, :])
        ax32.set_xlabel('n')
        ax32.set_title('Wavelets of predicted test data')
        fig3.tight_layout()

    # re-normalize wavelet from [0,1] into real dimension
    testPredict = re_normalize(testPredict, maxV_wavelet, minV_wavelet, normalize_style)
    new_testPredict = wavelet_reconstruct(testPredict)

    if plot_flag:
        fig4 = plt.figure()
        ax41 = fig4.add_subplot(111)
        ax41.plot(np.array(test_data), 'ro--', label='the origindal data')
        ax41.plot(np.array(new_testPredict), 'g+-', label='the predicted data')
        # ax41.set_ylim([0, 1])
        ax41.set_xlabel('Year')
        ax41.set_title('time series(test dataset) by wavelet')
        ax41.legend()
        print(steepness)
        plt.show()

    data_predicted = np.hstack((best_predict, new_testPredict))

    data_predicted = re_normalize(data_predicted, maxV, minV, normalize_style)
                # mse, rmse, nmse = statistics(dataset_copy, data_predicted)
                # print("Nc -> %d, Order -> %d, rmse -> %f, min_rmse is %f" % (Nc, Order, rmse, min_rmse))
                # # use rmse as performance index
                # if rmse < min_rmse:
                #     min_rmse = rmse
                #     best_Nc = Nc
                #     best_Order = Order
                #     best_predict = data_predicted
                # # use nmse as index
                # print("Nc -> %d, Order -> %d, nmse -> %f, min_nmse is %f" % (Nc, Order, nmse, min_nmse))
                # if nmse < min_nmse:
                #     min_nmse = nmse
                #     best_Nc = Nc
                #     best_Order = Order
                #     best_predict = data_predicted
    # return  [1,1]
    return data_predicted, rmse, min_nmse, best_Order, best_Nc, best_alpha





def main():
    # load time series data
    # data set 1: enrollment
    # dataset = np.array([13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807,
    #                     16919, 16388, 15433, 15497, 15145,15163, 15984, 16859, 18150, 18970,
    #                     19328, 19337, 18876])
    # time = range(len(dataset))
    # #
    # data set 2: TAIEX(use 70% data as train data)
    TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')  # ratio = 0.75
    dataset = TAIEX.values.flatten()
    time = range(len(dataset))
    # data set 3: sunspot
    # sunspot = pd.read_csv('sunspot.csv', delimiter=';').as_matrix()
    # dataset = sunspot[:-1, 1]
    # time = sunspot[:-1, 0]
    # # data set 4 : MG chaos( even use 10% data as train data)
    # import scipy.io as sio
    # dataset = sio.loadmat('MG_chaos.mat')['dataset'].flatten()
    # only use data from t=124 : t=1123  (all data previous are not in the same pattern!)
    # dataset = dataset[123:1123]
    # #
    # dataset = dataset[118:1118]
    # time = range(len(dataset))
    # # plt.plot(dataset[500:], 'ob')
    # plt.ylim([-0.1, 1.38])
    # plt.xlim([0, 500])
    # plt.show()

    '''
    trend data
    '''
    # linear dataset
    # dataset = np.array(list(range(00)))
    # time = range(len(dataset))
    # # square dataset
    # dataset = np.array([i for i in np.linspace(0, 100, 1000)])
    # dataset = np.array([np.sin(i) for i in np.linspace(0, 100, 1000)])
    # dataset = np.array([np.power(i, 2) for i in np.linspace(0, 100, 1000)])
    # time = range(len(dataset))
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')
    # dataset = pd.read_csv('monthly-electricity-production-i.csv', delimiter=';', parse_dates=[0], date_parser=dateparse).as_matrix()
    # time = dataset[:, 0]
    # dataset = np.array(dataset[:, 1], dtype=np.float)

    # Outlier detection data
    # src = '/home/shanchao/Documents/For_research/By_Python/HFCM/NAB-master/data/realTraffic/occupancy_6005.csv'
    # dataset = pd.read_csv(src, delimiter=',').as_matrix()[:, 1]
    # dataset =np.array(dataset, dtype=np.float)
    # time = range(len(dataset))
    # HFCM_ridge(dataset, 0.75, True)
    ratio = 0.806
    data_predicted, rmse, nmse, best_Order, best_Nc, best_alpha = HFCM_ridge(dataset, ratio)
    print('RMSE is %f, NMSE is %f' % (rmse, nmse))
    len_train_data = int(len(dataset) * ratio)
    mse, rmse, nmse = statistics(dataset, data_predicted)
    print('*' * 80)
    print('best Order is %d, best Nc is %d, best alpha is %f' % (best_Order, best_Nc, best_alpha))
    print('Forecasting all: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(rmse, 2), rmse, nmse))
    mse, rmse, nmse = statistics(dataset[len_train_data:], data_predicted[len_train_data:])
    print('Forecasting test: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(rmse, 2), rmse, nmse))

    fig4 = plt.figure()
    ax41 = fig4.add_subplot(111)

    ax41.plot(time, dataset, 'ro--', label='the original data')
    ax41.plot(time, data_predicted, 'g+-', label='the predicted data')
    # ax41.set_ylim([0, 1])
    ax41.set_xlabel('Year')
    ax41.set_title('time series prediction ')
    ax41.legend()
    plt.show()



def HaarWaveletTransform(x, J):
    N = len(x)
    C = np.zeros(shape=(J+1, N))
    # W: wavelet coefficients
    W = np.zeros(shape=(J+1, N))
    C[0, :] = x.copy()
    for j in range(1, J+1):
        for k in range(1, N):
            C[j, k] = 1/2 * (C[j-1, k] + C[j-1, k - np.power(2, j-1)])
            W[j, k] = C[j-1][k] - C[j, k]

    W[0, :] = C[J, :]
    return W[:, np.power(2, J):]



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
    meanV = np.mean(origin)
    dominator = np.linalg.norm(predicted - meanV, 2)
    return mse, rmse, mse / np.power(dominator, 2)


if __name__ == '__main__':




    main()

    # # dataset = pd.read_csv('AirPassengers.csv', delimiter=',').as_matrix()[:, 2]
    # TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')  # ratio = 0.75
    # dataset = TAIEX.values.flatten()
    # time = range(len(dataset))
    # normalize_style = '01'
    # dataset_copy = dataset.copy()
    # dataset, maxV, minV = normalize(dataset, normalize_style)
    # Nc = 4
    # U_train = HaarWaveletTransform(dataset, Nc-1)
    # re_construct = np.sum(U_train, 0)
    # k = 2**(Nc-1)
    # # print(np.all(np.abs(re_construct - dataset[k:]) < 0.0000001))
    # #
    # # fig1 = plt.figure()
    # # ax1 = fig1.add_subplot(111)
    # # # fig1.hold()
    # # for i in range(Nc):
    # #     ax1.plot(U_train[i, :])
    # # # plt.show()
    # #
    # # fig2 = plt.figure()
    # # ax_2 = fig2.add_subplot(111)
    # # ax_2.plot(dataset[k:], 'ro--', label='the original data')
    # # ax_2.plot(re_construct, 'g+-', label='the predicted data')
    # # ax_2.set_xlabel('Year')
    # # ax_2.set_title('time series reconstruction using modwt(imodwt)')
    # # ax_2.legend()
    # # plt.savefig('5_9_imodwt_outcome//' + wavelet_type + '.png')
    # # plt.show()
    # # steepness of sigmoid function
    # belta = 1
    # ratio = 0.7
    # # partition dataset into train set and test set\
    # if len(dataset) > 30:
    #     # ratio = 0.83
    #     train_data, test_data = splitData(dataset, ratio)
    # else:
    #     train_data, test_data = splitData(dataset, 1)
    #     test_data = train_data
    #
    # len_train_data = len(train_data)
    # len_test_data = len(test_data)
    # # grid search
    # # best parameters
    # best_Order = -1
    # best_Nc = -1
    # min_nmse = np.inf
    #
    # Nc = 4
    # max_level = Nc - 1
    # import pywt as pw
    #
    # wavelet_type = 'db1'
    # # former_coffis =
    #
    # k = 2 ** max_level
    # coffis = wavelet_transform(dataset, max_level)
    # # # test causal
    # # U_train = coffis[:, :10]
    # # U_train_2 = wavelet_transform(dataset[:k+10], max_level)
    # # print(np.all(U_train_2 == U_train))
    # # exit(10)
    #
    # coffis = wavelet_transform(dataset, max_level)
    # coffis, maxV_wavelet, minV_wavelet = normalize(coffis, normalize_style)
    # U_train = coffis[:, :len_train_data - k]
    # U_train = re_normalize(U_train, maxV_wavelet, minV_wavelet, normalize_style)
    # new_trainPredict = wavelet_reconstruct(U_train)
    # fig2 = plt.figure()
    # ax_2 = fig2.add_subplot(111)
    # ax_2.plot(train_data[k:], 'ro--', label='the original data')
    # ax_2.plot(new_trainPredict, 'g+-', label='the predicted data')
    # ax_2.set_xlabel('Year')
    # ax_2.set_title('time series reconstruction(on train dataset) using modwt(imodwt)')
    # ax_2.legend()
    # # plt.savefig('5_9_imodwt_outcome//' + wavelet_type + '.png')
    # # plt.show()
    # U_test = coffis[:, len_train_data-k:]
    # # new_testPredict = wavelet_reconstruct(np.hstack((coffis_dict[len_train_data][:, :len_train_data], U_test)), coffis_dict,
    # #                                       wavelet_type, len_train_data)
    # U_test = re_normalize(U_test, maxV_wavelet[len_train_data - k:], minV_wavelet[len_train_data - k:], normalize_style)
    #
    # new_testPredict = wavelet_reconstruct(U_test)
    # fig2 = plt.figure()
    # ax_2 = fig2.add_subplot(111)
    # ax_2.plot(test_data, 'ro--', label='the original data')
    # ax_2.plot(new_testPredict, 'g+-', label='the predicted data')
    # ax_2.set_xlabel('Year')
    # ax_2.set_title('time series reconstruction(on test data) using modwt(imodwt)')
    # ax_2.legend()
    # # plt.savefig('5_9_imodwt_outcome//' + wavelet_type + '.png')
    # plt.show()
    # #
