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


def wavelet_transform(x, J, k, filter_type, coffis_dict):
    # calculate MODWT increment
    # k is start point index
    # length of time series
    K = len(x)
    wavelet_type = filter_type
    coffis = np.zeros(shape=(J+1, K-k))

    coffis_dict[k] = modwt(x[: k], wavelet_type, J)  # for the reconstruction of k'th point
    for i in range(k, K):
        temp_coffis = modwt(x[: i+1], wavelet_type, J)
        coffis_dict[i+1] = temp_coffis
        coffis[:, i-k] = temp_coffis[:, -1]
    return coffis


def wavelet_reconstruct(predicted_coffis, coffis_dict, filter_type, start_point):
    # calculate MODWT increment
    # k is start point index
    # return np.sum(predicted_coffis, axis=0)
    from modwt import imodwt
    return imodwt(predicted_coffis, filter_type)[start_point:]
    # # length of time series
    # Nc, len_series = predicted_coffis.shape
    # x = np.zeros(shape=(len_series,))
    # wavelet_type = filter_type
    #
    # for i in range(len_series):
    #     temp_x = imodwt(np.hstack((coffis_dict[i + start_point], np.reshape(predicted_coffis[:, i], (Nc, 1)))), wavelet_type)
    #     x[i] = temp_x[-1]
    # return x



def HFCM_ridge(dataset, ratio=0.7, plot_flag=False):


    # from modwt import modwt, imodwt
    # dataset = pd.read_csv('AirPassengers.csv', delimiter=',').as_matrix()[:, 2]
    normalize_style = '-01'
    dataset_copy = dataset.copy()
    dataset, maxV, minV = normalize(dataset, normalize_style)

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
    min_nmse = np.inf
    for Order in range(5, 6):
        for Nc in range(6, 7):
            max_level = Nc - 1

            wavelet_type = 'haar'
            k = 2 ** max_level

            coffis_dict = {}
            coffis = wavelet_transform(dataset, max_level, k, wavelet_type, coffis_dict)
            # # test imodwt
            # U_train = coffis[:, :len_train_data - k]
            # U_train, maxV_train_wavelet, minV_train_wavelet = normalize(U_train, normalize_style)

            # new_trainPredict = wavelet_reconstruct(U_train, coffis_dict, wavelet_type, k)
            #
            # fig1 = plt.figure()
            # ax1 = fig1.add_subplot(111)
            # # fig1.hold()
            # for i in range(Nc):
            #     ax1.plot(U_train[i, :])
            #
            # ax1.set_xlabel('n')
            # ax1.set_title('Wavelets of train data')
            # fig1.tight_layout()
            #
            # new_trainPredict = wavelet_reconstruct(np.hstack((coffis_dict[len_train_data][:, :k], U_train)), coffis_dict, wavelet_type, k)
            # fig2 = plt.figure()
            # ax_2 = fig2.add_subplot(111)
            # ax_2.plot(train_data[k:], 'ro--', label='the original data')
            # ax_2.plot(new_trainPredict, 'g+-', label='the predicted data')
            # ax_2.set_xlabel('Year')
            # ax_2.set_title('time series reconstruction using modwt(imodwt)')
            # ax_2.legend()
            # # plt.savefig('imodwt_outcome//' + wavelet_type + '.png')
            # plt.show()
            # exit(4)

            U_train = coffis[:, :len_train_data - k]

            # normalize wavelet into [0, 1]
            U_train, maxV_train_wavelet, minV_train_wavelet = normalize(U_train, normalize_style)
            # 30 independent runs
            nTotalRun = 1

            for nrun in range(nTotalRun):

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
                    # use ridge regression
                    clf.fit(samples[:, :-1], samples[:, -1])
                    W_learned[node_solved, :] = clf.coef_
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
                        ax1.plot(U_train[i, :])

                    ax1.set_xlabel('n')
                    ax1.set_title('Wavelets of train data')
                    ax2 = fig1.add_subplot(212)
                    for i in range(Nc):
                        ax2.plot(trainPredict[i, :])
                    ax2.set_xlabel('n')
                    ax2.set_title('Wavelets of predicted train data')
                    fig1.tight_layout()
                # plt.show()

                # # re-normalize wavelet from [0,1] into real dimension
                trainPredict = re_normalize(trainPredict, maxV_train_wavelet, minV_train_wavelet, normalize_style)

                # # reconstruct part
                new_trainPredict = wavelet_reconstruct(np.hstack((coffis_dict[len_train_data][:, :k], trainPredict)), coffis_dict, wavelet_type, k)

                # print('Error is %f' % np.linalg.norm(np.array(train_data)[k:] - new_trainPredict, 2))
                if plot_flag:
                    # plot train data series and predicted train data series
                    fig2 = plt.figure()
                    ax_2 = fig2.add_subplot(111)
                    ax_2.plot(train_data[k:], 'ro--', label='the original data')
                    ax_2.plot(new_trainPredict, 'g+-', label='the predicted data')
                    ax_2.set_xlabel('Year')
                    ax_2.set_title('time series(train dataset) by wavelet')
                    ax_2.legend()
                plt.show()
                exit(6)

                # # test data
                U_test = coffis[:, len_train_data-k:]
                # normalize test data
                U_test, maxV_test_wavelet, minV_test_wavelet = normalize(U_test, normalize_style)

                testPredict = np.zeros(shape=(Nc, len_test_data))
                samples_test = {}
                for i in range(Nc):  # solve each node in turn
                    samples = create_dataset(U_test, belta, Order, i)
                    samples_test[i] = samples[:-Order, :]  # delete the last "Order' rows(all zeros)
                    testPredict[i, :Order] = U_test[i, :Order]
                    testPredict[i, Order:] = predict(samples_test[i], W_learned[i, :], steepness[i], belta)
                if plot_flag:
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
                testPredict = re_normalize(testPredict, maxV_test_wavelet, minV_test_wavelet, normalize_style)
                new_testPredict = wavelet_reconstruct(np.hstack((coffis_dict[len_train_data], testPredict)), coffis_dict, wavelet_type, len_train_data)

                # if plot_flag:
                #     fig4 = plt.figure()
                #     ax41 = fig4.add_subplot(111)
                #     ax41.plot(np.array(test_data), 'ro--', label='the origindal data')
                #     ax41.plot(np.array(new_testPredict), 'g+-', label='the predicted data')
                #     # ax41.set_ylim([0, 1])
                #     ax41.set_xlabel('Year')
                #     ax41.set_title('time series(test dataset) by wavelet')
                #     ax41.legend()
                #     print(steepness)
                #     plt.show()
                #
                # data_predicted = np.hstack((dataset[:k], new_trainPredict[:], new_testPredict[:]))
                #
                # data_predicted = re_normalize(data_predicted, maxV, minV, normalize_style)
                # mse, rmse, nmse = statistics(dataset_copy, data_predicted)
                # print("Nc -> %d, Order -> %d, nmse -> %f, min_nmse is %f" % (Nc, Order, nmse, min_nmse))
                # if nmse < min_nmse:
                #     min_nmse = nmse
                #     best_Nc = Nc
                #     best_Order = Order

    # return data_predicted, rmse, min_nmse, best_Order, best_Nc





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
    # # # only use data from t=124 : t=1123  (all data previous are not in the same pattern!)
    # # dataset = dataset[123:1123]
    # # #
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
    # dataset = np.array(list(range(500)))
    # time = range(len(dataset))
    # square dataset
    # dataset = np.array([i**2 for i in np.linspace(0, 10, 100)])
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
    HFCM_ridge(dataset, 0.5, True)
    # # data_predicted, rmse, nmse, best_Order, best_Nc = HFCM_ridge(dataset, 0.5, True)
    # # mse, rmse = statistics(dataset, data_predicted)
    # print('*' * 80)
    # print('best Order is %d, best Nc is %d' % (best_Order, best_Nc))
    # print('MSE is %f. RMSE is %f, NMSE is %f' % (np.power(rmse, 2), rmse, nmse))
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




    # main()
    # # from modwt import modwt, imodwt
    # dataset = pd.read_csv('AirPassengers.csv', delimiter=',').as_matrix()[:, 2]
    TAIEX = pd.read_excel('2000_TAIEX.xls', sheetname='clean_v1_2000')  # ratio = 0.75
    dataset = TAIEX.values.flatten()
    time = range(len(dataset))
    normalize_style = '01'
    dataset_copy = dataset.copy()
    dataset, maxV, minV = normalize(dataset, normalize_style)
    Nc = 4
    U_train = HaarWaveletTransform(dataset, Nc-1)
    re_construct = np.sum(U_train, 0)
    k = 2**(Nc-1)
    print(np.all(re_construct == dataset[k:]))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # fig1.hold()
    for i in range(Nc):
        ax1.plot(U_train[i, :])
    # plt.show()

    fig2 = plt.figure()
    ax_2 = fig2.add_subplot(111)
    ax_2.plot(dataset[k:], 'ro--', label='the original data')
    ax_2.plot(re_construct, 'g+-', label='the predicted data')
    ax_2.set_xlabel('Year')
    ax_2.set_title('time series reconstruction using modwt(imodwt)')
    ax_2.legend()
    # plt.savefig('5_9_imodwt_outcome//' + wavelet_type + '.png')
    plt.show()
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
    # for wavelet_type in pw.wavelist(family='haar', kind='discrete'):
    #     k = 2 ** max_level
    #     coffis = modwt(dataset, wavelet_type, max_level)
    #     U_train = coffis[:, :k+10]
    #     U_train_2 = modwt(dataset[:k+10], wavelet_type, max_level)
    #     print(np.all(U_train_2 == U_train))
    #     # for wavelet_type in pw.wavelist(kind='discrete'):
    #     #     print(wavelet_type)
    #     # coffis_dict is used to store all  coffis of previous points(value) of current  point index(key)
    #     # coffis_dict = {}
    #     # coffis = wavelet_transform(dataset, max_level, k, wavelet_type, coffis_dict)
    #     #
    #     # U_train = coffis[:, :len_train_data - k]
    #     #
    #     # coffis_dict_2 = {}
    #     # U_train_2 = wavelet_transform(dataset[:len_train_data], max_level, k, wavelet_type, coffis_dict_2)
    #
    #     # U_train, maxV_train_wavelet, minV_train_wavelet = normalize(U_train, normalize_style)
    #     # print('difference is %f' % np.linalg.norm(U_train - U_train_2))
    #     # fig1 = plt.figure()
    #     # ax1 = fig1.add_subplot(211)
    #     # # fig1.hold()
    #     # for i in range(Nc):
    #     #     ax1.plot(U_train[i, :])
    #     #
    #     # ax1.set_xlabel('n')
    #     # ax1.set_title('Wavelets of train data')
    #     # ax2 = fig1.add_subplot(212)
    #     # for i in range(Nc):
    #     #     ax2.plot(U_train_2[i, :])
    #     # ax2.set_xlabel('n')
    #     # ax2.set_title('Wavelets of predicted train data')
    #     # fig1.tight_layout()
    #     # c1 = np.hstack((coffis_dict_2[len_train_data][:, :k], U_train_2))
    #     # c2 = modwt(train_data, wavelet_type, max_level)
    #
    #     # new_trainPredict = wavelet_reconstruct(np.hstack((coffis_dict_2[len_train_data][:, :k], U_train_2)), coffis_dict_2, wavelet_type, k)
    #     # new_trainPredict = wavelet_reconstruct(U_train_2, coffis_dict_2, wavelet_type, k)
    #     # fig2 = plt.figure()
    #     # ax_2 = fig2.add_subplot(111)
    #     # ax_2.plot(train_data[k:], 'ro--', label='the original data')
    #     # ax_2.plot(new_trainPredict, 'g+-', label='the predicted data')
    #     # ax_2.set_xlabel('Year')
    #     # ax_2.set_title('time series reconstruction using modwt(imodwt)')
    #     # ax_2.legend()
    #     # # plt.savefig('5_9_imodwt_outcome//' + wavelet_type + '.png')
    #     # plt.show()
    #     # U_test = coffis[:, len_train_data-k:]
    #     # # new_testPredict = wavelet_reconstruct(np.hstack((coffis_dict[len_train_data][:, :len_train_data], U_test)), coffis_dict,
    #     # #                                       wavelet_type, len_train_data)
    #     # from modwt import imodwt
    #     # new_testPredict = imodwt(np.hstack((coffis_dict[len_train_data+len_test_data][:, :len_train_data], U_test)), wavelet_type)[len_train_data:]
    #     # fig2 = plt.figure()
    #     # ax_2 = fig2.add_subplot(111)
    #     # ax_2.plot(test_data, 'ro--', label='the original data')
    #     # ax_2.plot(new_testPredict, 'g+-', label='the predicted data')
    #     # ax_2.set_xlabel('Year')
    #     # ax_2.set_title('time series reconstruction(on test data) using modwt(imodwt)')
    #     # ax_2.legend()
    #     # # plt.savefig('5_9_imodwt_outcome//' + wavelet_type + '.png')
    #     # plt.show()
    #     #
