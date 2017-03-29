import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.rcParams['font.sans-serif'] = ['SimHei']  #
# 指定默认字体

def fcm(x, c=3, threshold=1e-5):

    n = len(x)
    m = 2  # 隶属度的幂, large m will increase the fuzziness of the function
    # generate c * n random numbers in the interval [0, 1] : fuzzy partition matrix U
    U = np.random.rand(c, n)
    # U 的每列 归一化 （和为1）
    col_sum = sum(U)
    U = U / col_sum
    # generate cluster center vector V
    V = np.zeros(c)
    Distance = np.zeros(shape=(c, n), dtype=float)
    newObj = np.inf  # 目标函数值初始化为无穷大
    cnt = 1
    maxCnt = 100
    while cnt < maxCnt:
        # Compute v_i (1 <= i <= c)（ 更新聚类中心）
        for i in range(c):
            top_sumV = 0  # 分子
            btm_sumV = 0  # 分母
            for j in range(n):
                top_sumV += np.power(U[i, j], m) * x[j]
                btm_sumV += np.power(U[i, j], m)
            V[i] = top_sumV / btm_sumV
        # compute all d_{ij}   （更新距离）
        for i in range(c):
            for j in range(n):
                Distance[i][j] = abs(V[i] - x[j])
        # update the fuzzy partition matrix U  （更新模糊矩阵）
        for i in range(c):
            for j in range(n):
                tempSum = 0
                for k in range(c):
                    tempSum += np.power(Distance[i, j] / Distance[k, j], 2 / (m - 1))
                U[i, j] = 1 / tempSum
        # 保留当前目标函数值，并计算新的目标函数值
        previousObj = newObj
        newObj = objFunc(Distance, U, m)
        # 判断是否收敛
        if abs(previousObj - newObj) < threshold:
            break
        # print("FCM 第 %d 次" % cnt)
        cnt += 1

    return U, V


    return  U, V

def objFunc(distance, U,  m=2):
    c, n= distance.shape
    J = 0   # 目标函数值:
    for i in range(c):
        for j in range(n):
            J += np.power(U[i, j], m) * distance[i, j]
    return J


if __name__ == '__main__':
    enrollment = [13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807, 16919,
                  16388, 15433, 15497, 15145, 15163, 15984, 16859, 18150, 18970, 19328,
                  19337, 18876]
    time = list(range(1971, 1992 + 1))
    # 归一化 x
    raw_x = enrollment
    minV = min(raw_x)
    maxV = max(raw_x)
    interval = maxV - minV
    x = [(xi - minV) / interval for xi in raw_x]
    for i in x:
        print("%.3f" % i, end=', ')
    U, V = fcm(x)
    print(U)
    print(V)

    np.savetxt('U.txt', U, fmt='%.4f')
    plt.close()
    for i in range(len(U)):
        plt.plot(U[i, :], '-*')
    plt.xlabel('Time')
    plt.ylabel('隶属度')
    plt.title('采用abs距离度量得到的隶属度(enrollment)')
    plt.show()
    # plt.plot(time, enrollment, marker='o')
    # plt.title('Enrollment time series')
    # plt.xlabel('Year')
    # plt.ylabel('enrollment')
    # plt.show()
