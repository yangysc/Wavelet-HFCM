import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    U = np.loadtxt('U.txt')
    plt.close()
    for i in range(len(U)):
        plt.plot(U[i, :], '-o')
    plt.xlabel('Time')
    plt.ylabel('隶属度')
    plt.title('采用abs距离度量得到的隶属度(enrollment)')
    plt.show()
