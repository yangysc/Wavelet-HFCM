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


def wavelet_reconstruct(predicted_coffis):
    return np.sum(predicted_coffis, axis=0)


def HFCM_ridge(dataset1, ratio=0.7, plot_flag=False):

    # dataset = np.diff(dataset)
    # from modwt import modwt, imodwt
    # dataset = pd.read_csv('AirPassengers.csv', delimiter=',').as_matrix()[:, 2]
    normalize_style = '-01'
    dataset_copy = dataset1.copy()
    dataset, maxV, minV = normalize(dataset1, normalize_style)
    # dataset = dataset1
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
    validation_ratio = 0.2
    len_validation_data = int(len_train_data * validation_ratio)

    small_alpha = [1e-12, 1e-14, 1e-20]
    # small_alpha = [1e-20]
    # small_alpha = np.linspace(1e-15, 0.1, 20)
    # small_alpha = [1e-20]
    Order_list = list(range(2, 9))
    Nc_list = list(range(2, 8))
    # alpha_list = np.hstack((small_alpha, np.linspace(0.1, 15, 30)))
    alpha_list = small_alpha
    # rmse_total = np.zeros(shape=(len(Nc_list), len(Order_list)))
    best_Order = -1
    best_Nc = -1

    best_alpha_inall = np.zeros(shape=(len(Nc_list), len(Order_list)))
    best_alpha_scala = -1  # 记录最优(Nc, Order)下最优的alpha
    min_nmse = np.inf
    min_rmse_inall = np.inf
    best_W_learned_inall = None
    best_steepness_inall = None
    best_predict_inall = np.zeros(shape=len_train_data)

    for Oidx, Order in enumerate(Order_list):
        for Nidx, Nc in enumerate(Nc_list):
            # min_rmse 用于记录每个(Order, Nc)下的最小的rmse（优化alpha ）
            min_rmse = np.inf
            best_alpha = -1

            best_W_learned = None
            best_steepness = None
            best_predict = np.zeros(shape=len_train_data)
            # Grid Search for optimizing alpha
            for alpha in alpha_list:
                max_level = Nc - 1
                coffis = wavelet_transform(dataset, max_level)
                np.savetxt('coffis.txt', coffis, delimiter=',')
                # coffis, maxV_wavelet, minV_wavelet = normalize(coffis, normalize_style)
                k = 2 ** max_level
                U_train = coffis[:, :len_train_data - k - len_validation_data]

                # the ridge regression
                tol = 1e-24
                from sklearn import linear_model
                clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol)
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
                trainPredict = np.zeros(shape=(Nc, len_train_data-k-len_validation_data))
                for i in range(Nc):
                    trainPredict[i, :Order] = U_train[i, :Order]
                    trainPredict[i, Order:] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)
                if plot_flag:
                    fig1 = plt.figure()
                    ax1 = fig1.add_subplot(211)
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
                # trainPredict = re_normalize(trainPredict, maxV_wavelet, minV_wavelet, normalize_style)

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

                # validation stage for choosing right parameters
                U_validation = coffis[:, len_train_data - k - len_validation_data - Order:len_train_data - k]
                validationPredict = np.zeros(shape=(Nc, len_validation_data))
                samples_validation = {}
                for i in range(Nc):  # solve each node in turn
                    samples = create_dataset(U_validation, belta, Order, i)
                    samples_validation[i] = samples[:-Order, :]  # delete the last "Order' rows(all zeros)
                    # testPredict[i, :Order] = U_test[i, :Order]
                    validationPredict[i, :] = predict(samples_validation[i], W_learned[i, :], steepness[i], belta)
                # validationPredict = re_normalize(validationPredict, maxV_wavelet, minV_wavelet, normalize_style)
                new_validationPredict = wavelet_reconstruct(validationPredict)
                mse, rmse, nmse = statistics(dataset[len_train_data - len_validation_data:len_train_data], new_validationPredict)
                # rmse_total[Nidx, Oidx] = rmse

                print("Nc -> %d, Order -> %d, alpha -> %g: rmse -> %f  | min_rmse is %f, min_rmse_inall is %f (%d, %d)"
                      % (Nc, Order, alpha, rmse, min_rmse, min_rmse_inall, best_Nc, best_Order))
                # use rmse as performance index
                if rmse < min_rmse:
                    min_rmse = rmse

                    best_predict[:] = np.hstack((new_trainPredict, new_validationPredict))
                    best_W_learned = W_learned
                    best_steepness = steepness
                    best_alpha = alpha
            # 记录当前(Nc, Order)下的最优 alpha
            best_alpha_inall[Nidx, Oidx] = best_alpha
            # 判断当前的(Nc, Order)下，全局rmse是否减小
            if min_rmse < min_rmse_inall:
                min_rmse_inall = min_rmse
                best_Nc = Nc
                best_Order = Order
                best_predict_inall = best_predict
                best_W_learned_inall = best_W_learned
                best_steepness_inall = best_steepness
                best_alpha_scala = best_alpha



    # print(rmse_total)
    if len(dataset) <= 30:
        data_predicted = best_predict
        data_predicted = re_normalize(data_predicted, maxV, minV, normalize_style)
        return data_predicted, rmse, min_nmse, best_Order, best_Nc, best_alpha
    else:
        # # test data
        max_level = best_Nc - 1
        coffis = wavelet_transform(dataset, max_level)
        # coffis, maxV_wavelet, minV_wavelet = normalize(coffis, normalize_style)
        k = 2 ** max_level
        U_test = coffis[:, len_train_data-k-best_Order:]   # use last Order data point of train dataset

        testPredict = np.zeros(shape=(best_Nc, len_test_data))
        samples_test = {}
        for i in range(best_Nc):  # solve each node in turn
            samples = create_dataset(U_test, belta, best_Order, i)
            samples_test[i] = samples[:-best_Order, :]  # delete the last "Order' rows(all zeros)
            # testPredict[i, :Order] = U_test[i, :Order]
            testPredict[i, :] = predict(samples_test[i], best_W_learned_inall[i, :], best_steepness_inall[i], belta)
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
        # testPredict = re_normalize(testPredict, maxV_wavelet, minV_wavelet, normalize_style)
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

        data_predicted = np.hstack((best_predict_inall, new_testPredict))
        data_predicted = re_normalize(data_predicted, maxV, minV, normalize_style)


        print('origin w')

        w_whole = np.zeros(shape=(Nc, Nc, Order), dtype=np.float)

        for j in range(Nc):
            for i in range(Order):
                w_whole[:, j, i] = best_W_learned[:, 2*j+i]



        return data_predicted, best_Order, best_Nc, best_alpha_scala


def analyze_paras_HFCM(dataset1, ratio=0.7):

    normalize_style = '-01'
    dataset_copy = dataset1.copy()
    dataset, maxV, minV = normalize(dataset1, normalize_style)
    # dataset = dataset1
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
    validation_ratio = 0.2
    len_validation_data = int(len_train_data * validation_ratio)

    small_alpha = [1e-12, 1e-20, 1e-14, 1e-13]
    # small_alpha = np.linspace(1e-15, 0.1, 20)
    # small_alpha = [0, 1e-20, 1e-12, 1e-14, 1e-13]
    Order_list = list(range(1, 7))
    Nc_list = list(range(2, 8))
    # alpha_list = np.hstack((small_alpha, np.linspace(1, 8, 15)))
    alpha_list = small_alpha
    # rmse_total = np.zeros(shape=(len(Nc_list), len(Order_list)))
    best_Order = -1
    best_Nc = -1

    best_alpha_inall = np.zeros(shape=(len(Nc_list), len(Order_list)))
    best_alpha_scala = -1  # 记录最优(Nc, Order)下最优的alpha
    min_nmse = np.inf
    min_rmse_inall = np.inf
    best_W_learned_inall = None
    best_steepness_inall = None
    best_predict_inall = np.zeros(shape=len_train_data)

    # 每个（Nc, Order）下最优alpha 时的误差(Validation dataset)
    rmse_total = np.zeros(shape=(len(Nc_list), len(Order_list)))

    for Oidx, Order in enumerate(Order_list):
        for Nidx, Nc in enumerate(Nc_list):
            # min_rmse 用于记录每个(Order, Nc)下的最小的rmse（优化alpha ）
            min_rmse = np.inf
            best_alpha = -1

            best_W_learned = None
            best_steepness = None
            best_predict = np.zeros(shape=len_train_data)
            # Grid Search for optimizing alpha
            for alpha in alpha_list:
                max_level = Nc - 1
                coffis = wavelet_transform(dataset, max_level)
                # coffis, maxV_wavelet, minV_wavelet = normalize(coffis, normalize_style)
                k = 2 ** max_level
                U_train = coffis[:, :len_train_data - k - len_validation_data]

                # the ridge regression
                tol = 1e-24
                from sklearn import linear_model
                clf = linear_model.Ridge(alpha=alpha, fit_intercept=False, tol=tol)
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
                trainPredict = np.zeros(shape=(Nc, len_train_data - k - len_validation_data))
                for i in range(Nc):
                    trainPredict[i, :Order] = U_train[i, :Order]
                    trainPredict[i, Order:] = predict(samples_train[i], W_learned[i, :], steepness[i], belta)


                # # re-normalize wavelet from [0,1] into real dimension
                # trainPredict = re_normalize(trainPredict, maxV_wavelet, minV_wavelet, normalize_style)

                # # reconstruct part
                new_trainPredict = wavelet_reconstruct(trainPredict)
                new_trainPredict = np.hstack((train_data[:k], new_trainPredict))

                # print('Error is %f' % np.linalg.norm(np.array(train_data)[k:] - new_trainPredict, 2))


                # validation stage for choosing right parameters
                U_validation = coffis[:, len_train_data - k - len_validation_data - Order:len_train_data - k]
                validationPredict = np.zeros(shape=(Nc, len_validation_data))
                samples_validation = {}
                for i in range(Nc):  # solve each node in turn
                    samples = create_dataset(U_validation, belta, Order, i)
                    samples_validation[i] = samples[:-Order, :]  # delete the last "Order' rows(all zeros)
                    # testPredict[i, :Order] = U_test[i, :Order]
                    validationPredict[i, :] = predict(samples_validation[i], W_learned[i, :], steepness[i], belta)
                # validationPredict = re_normalize(validationPredict, maxV_wavelet, minV_wavelet, normalize_style)
                new_validationPredict = wavelet_reconstruct(validationPredict)

                # print(rmse_total)
                if len(dataset) <= 30:
                    data_predicted = best_predict
                    data_predicted = re_normalize(data_predicted, maxV, minV, normalize_style)
                    return data_predicted, rmse, min_nmse, best_Order, best_Nc, best_alpha
                else:
                    # # test data
                    max_level = Nc - 1
                    coffis = wavelet_transform(dataset, max_level)
                    # coffis, maxV_wavelet, minV_wavelet = normalize(coffis, normalize_style)
                    k = 2 ** max_level
                    U_test = coffis[:, len_train_data - k - Order:]  # use last Order data point of train dataset

                    testPredict = np.zeros(shape=(Nc, len_test_data))
                    samples_test = {}
                    for i in range(Nc):  # solve each node in turn
                        samples = create_dataset(U_test, belta, Order, i)
                        samples_test[i] = samples[:-Order, :]  # delete the last "Order' rows(all zeros)
                        # testPredict[i, :Order] = U_test[i, :Order]
                        testPredict[i, :] = predict(samples_test[i], W_learned[i, :],
                                                    steepness[i], belta)


                    # re-normalize wavelet from [0,1] into real dimension
                    # testPredict = re_normalize(testPredict, maxV_wavelet, minV_wavelet, normalize_style)
                    new_testPredict = wavelet_reconstruct(testPredict)
                    # (train, validation, test)
                    data_predicted = np.hstack((new_trainPredict, new_validationPredict, new_testPredict))
                    # data_predicted = re_normalize(data_predicted, maxV, minV, normalize_style)

                # mse, rmse, nmse = statistics(dataset_copy[len_train_data:], data_predicted[len_train_data:])
                                             # new_validationPredict)
                mse, rmse, nmse = statistics(dataset[len_train_data:], data_predicted[len_train_data:])

                # mse, rmse, nmse = statistics(dataset[len_train_data - len_validation_data:len_train_data],
                #                              new_validationPredict)


                print("Nc -> %d, Order -> %d, alpha -> %g: rmse -> %f  | min_rmse is %f, min_rmse_inall is %f(%d, %d)"
                      % (Nc, Order, alpha, rmse, min_rmse, min_rmse_inall, best_Nc, best_Order))
                # use rmse as performance index
                if rmse < min_rmse:
                    min_rmse = rmse

                    best_predict[:] = np.hstack((new_trainPredict, new_validationPredict))
                    best_W_learned = W_learned
                    best_steepness = steepness
                    best_alpha = alpha
            # 记录当前(Nc, Order)下的最优 alpha
            best_alpha_inall[Nidx, Oidx] = best_alpha
            # 记录当前(Nc, Order)下最优rmse, 用于绘图分析
            rmse_total[Nidx, Oidx] = min_rmse
            # 判断当前的(Nc, Order)下，全局rmse是否减小
            if min_rmse < min_rmse_inall:
                min_rmse_inall = min_rmse
                best_Nc = Nc
                best_Order = Order
                best_predict_inall = best_predict
                best_W_learned_inall = best_W_learned
                best_steepness_inall = best_steepness
                best_alpha_scala = best_alpha

    df = pd.DataFrame(rmse_total, index=Nc_list, columns=Order_list)
    return df


# analyze hyper-parameters on the performance on Wavelet-HFCM
def analyze_parameter():
    import seaborn as sns
    plt.style.use(['seaborn-paper'])

    # Analyze sunspot and s&p 500 time series

    # data set : sunspot
    sunspot = pd.read_csv('./datasets/sunspot.csv', delimiter=';').as_matrix()
    dataset = sunspot[:-1, 1]
    ratio = 0.7674
    df1 = analyze_paras_HFCM(dataset, ratio=ratio)


    Nc_list = df1.index.values
    Order_list = df1.columns.values


    sp500_src = "./datasets/sp500.csv"
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    dataset = np.array(sp500[:, 1], dtype=np.float)
    df2 = analyze_paras_HFCM(dataset, ratio=0.6)
    # save df1 & df2 to excel
    writer = pd.ExcelWriter('output_sunspot_sp500.xlsx')
    df1.to_excel(writer, 'df1')

    df2.to_excel(writer, 'df2')
    writer.save()

    # Analyze MG chaos time series
    # # data set 4 : MG chaos( even use 10% data as train data)
    # import scipy.io as sio
    # dataset = sio.loadmat('MG_chaos.mat')['dataset'].flatten()
    # # only use data from t=124 : t=1123  (all data previous are not in the same pattern!)
    # dataset = dataset[123:1123]
    # # time = range(len(dataset))
    # # time = sp500[:, 0]
    # df3 = analyze_paras_HFCM(dataset, ratio)
    # writer = pd.ExcelWriter('output_MG.xlsx')
    # df3.to_excel(writer, 'df1')
    # writer.save()


    # RMSE versus varying level of decomposition
    # sunspot + S&P 500
    import shutil
    import os
    if not os.path.exists('./Outcome_for_papers/impact_parameters/varying_Nc'):
        os.makedirs('./Outcome_for_papers/impact_parameters/varying_Nc')

    if not os.path.exists('./Outcome_for_papers/impact_parameters/varying_Order'):
        os.makedirs('./Outcome_for_papers/impact_parameters/varying_Order')

    for order in Order_list:
        df = pd.DataFrame({r'$N_c$': Nc_list,
                           'S&P500': df2[order].values,
                           'Sunspot time series': df1[order].values})
        df = pd.melt(df, id_vars=r'$N_c$', var_name="Dataset", value_name='RMSE')
        g = sns.factorplot(x=r'$N_c$', y='RMSE', hue='Dataset',
                           hue_order=['Sunspot time series', 'S&P500'], data=df, kind='bar',
                           legend=False, palette=sns.color_palette(["#34495e", "#95a5a6"]))

        # resize figure box to -> put the legend out of the figure
        box = g.ax.get_position()  # get position of figure
        g.ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])  # resize position

        # Put a legend to the right side
        sns.plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)
        # plt.tight_layout()

        plt.savefig(
            r"./Outcome_for_papers/impact_parameters/varying_Nc/k=%d.pdf" % order)
        plt.savefig(
            r"./Outcome_for_papers/impact_parameters/varying_Nc/k=%d.tiff" % order)
        plt.close()

    for Nc in Nc_list:
        # / print(len(df_1.loc[Nc, :]))
        df = pd.DataFrame({'$k$': Order_list,
                           'S&P500': df2.loc[Nc, :].values,
                           'Sunspot time series': df1.loc[Nc, :].values})

        df = pd.melt(df, id_vars='$k$', var_name="Dataset", value_name='RMSE')
        g = sns.factorplot(x='$k$', y='RMSE', hue='Dataset',
                           hue_order=['Sunspot time series', 'S&P500'], data=df, kind='bar',
                           legend=False, palette=sns.color_palette(["#34495e", "#95a5a6"]))

        # resize figure box to -> put the legend out of the figure
        box = g.ax.get_position()  # get position of figure
        g.ax.set_position([box.x0, box.y0, box.width, box.height * 0.9])  # resize position

        # Put a legend to the right side
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

        plt.savefig(
            r"./Outcome_for_papers/impact_parameters/varying_Order/Nc=%d.pdf" % Nc)
        plt.savefig(
            r"./Outcome_for_papers/impact_parameters/varying_Order/Nc=%d.tiff" % Nc)
        plt.close()




def main():
    # load time series data

    ''' New data sets'''

    # dataset 1:monthly-closings-of-the-dowjones.csv  #todo:  good
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
    # sunspot = pd.read_csv(r'./datasets/monthly-closings-of-the-dowjones.csv', delimiter=',', parse_dates=[0],
    #                       date_parser=dateparse).as_matrix()
    #
    # dataset = sunspot[:, 1].astype(np.float)
    #
    # time = sunspot[:, 0]
    # ratio = 0.75

    # # dataset 2: monthly-milk-production-pounds-p.csv   #todo:  good
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
    # sunspot = pd.read_csv(r'./datasets/monthly-milk-production-pounds-p.csv', delimiter=',', parse_dates=[0],
    #                       date_parser=dateparse).as_matrix()
    #
    # dataset = sunspot[:, 1].astype(np.float)
    # time = sunspot[:, 0]
    # ratio = 0.8

    # dataset 3: monthly-critical-radio-frequenci  #todo:  good
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
    # sunspot = pd.read_csv(r'./datasets/monthly-critical-radio-frequenci.csv', delimiter=',', parse_dates=[0],
    #                       date_parser=dateparse).as_matrix()
    # dataset = sunspot[:, 1].astype(np.float)
    # time = sunspot[:, 0]
    # ratio = 0.75

    # # dataset 4:  co2-ppm-mauna-loa-19651980.csv  #todo:  good
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
    sunspot = pd.read_csv(r'./datasets/co2-ppm-mauna-loa-19651980.csv', delimiter=',', parse_dates=[0],
                          date_parser=dateparse).as_matrix()
    dataset = sunspot[:, 1].astype(np.float)
    time = sunspot[:, 0]
    ratio = 0.85


    # dataset 5: monthly-lake-erie-levels-1921-19.csv #todo:  good
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m')  # %Y-%m-%d
    # sunspot = pd.read_csv(r'./datasets/monthly-lake-erie-levels-1921-19.csv', delimiter=',', parse_dates=[0],
    #                       date_parser=dateparse).as_matrix()
    # dataset = sunspot[:, 1].astype(np.float)
    # time = sunspot[:, 0]
    # ratio = 0.7674

    '''old datasets'''

    # data set 1: sunspot
    # sunspot = pd.read_csv('./datasets/sunspot.csv', delimiter=';').as_matrix()
    # dataset = sunspot[:-1, 1]
    # np.savetxt('origin_sunspot.txt', dataset, delimiter=',')
    # time = sunspot[:-1, 0]
    # ratio = 0.7674
    #
    #
    # # # # data set 2 : MG chaos( even use 10% data as train data)
    import scipy.io as sio
    dataset = sio.loadmat('./datasets/MG_chaos.mat')['dataset'].flatten()
    # only use data from t=124 : t=1123  (all data previous are not in the same pattern!)
    dataset = dataset[123:1123]
    time = range(len(dataset))
    ratio = 0.5

    # data set 3 : sp500 index
    # sp500_src = "./datasets/sp500.csv"
    # dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    # sp500 = pd.read_csv(sp500_src, delimiter=',', parse_dates=[0], date_parser=dateparse).as_matrix()
    # dataset = np.array(sp500[:, 1], dtype=np.float)
    # time = sp500[:, 0]
    # ratio = 0.6



    # partition dataset into train set and test set
    length = len(dataset)
    len_train_data = int(length * ratio)

    validation_ratio = 0.2
    len_validation_data = int(len_train_data * validation_ratio)
    len_test_data = length - len_train_data

    # perform prediction
    data_predicted, best_Order, best_Nc, best_alpha = HFCM_ridge(dataset, ratio)


    # Outcomes
    # Error of the whole dataset
    mse, rmse, nmse = statistics(dataset, data_predicted)
    print('*' * 80)
    print('The ratio is %f' % ratio)
    print('best Order is %d, best Nc is %d, best alpha is %g' % (best_Order, best_Nc, best_alpha))
    print('Forecasting on all dataset: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(rmse, 2), rmse, nmse))

    # Error of Train dataset
    mse, rmse, nmse = statistics(dataset[:len_train_data-len_validation_data], data_predicted[:len_train_data-len_validation_data])
    print('Forecasting on train dataset: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(rmse, 2), rmse, nmse))

    # Error of Validation dataset
    mse, rmse, nmse = statistics(dataset[len_train_data-len_validation_data:len_train_data], data_predicted[len_train_data-len_validation_data:len_train_data])
    print('Forecasting on validation dataset: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(rmse, 2), rmse, nmse))

    # Error of Test dataset
    mse, Test_rmse, nmse = statistics(dataset[len_train_data:], data_predicted[len_train_data:])
    print('Forecasting on test dataset: MSE|RMSE|NMSE is : |%f |%f |%f|' % (np.power(Test_rmse, 2), Test_rmse, nmse))

    # print length of each subdatasets

    print('The whole length is %d' % length)
    print('Train dataset length is %d' % (len_train_data - len_validation_data))
    print('Validation dataset length is %d' % len_validation_data)
    print('Test dataset length is %d' % len_test_data)


    # plot time series
    import seaborn as sns
    plt.style.use(['seaborn-paper'])

    fig4 = plt.figure()
    ax41 = fig4.add_subplot(111)

    ax41.plot(time, dataset, 'r-', label='the original data')
    ax41.plot(time, data_predicted, 'go--', label='the predicted data')
    ax41.set_ylabel("Magnitude")
    ax41.set_xlabel('Time')
    # ax41.set_title('time series prediction ')

    # ax41.set_ylim([0.35, 1.4])  # for MG-chaos having a better visualization

    ax41.legend()
    plt.tight_layout()
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
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(origin, predicted)
    rmse = np.sqrt(mse)
    meanV = np.mean(origin)
    dominator = np.linalg.norm(predicted - meanV, 2)
    return mse, rmse, mse / np.power(dominator, 2)


if __name__ == '__main__':
    # analyze hyper-parameters on the performance of Wavelet-HFCM
    analyze_parameter()
    # main function
    # main()

