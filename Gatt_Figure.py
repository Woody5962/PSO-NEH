import matplotlib.pyplot as plt
import numpy as np


def gatt(g, sequence, mop, S, A):
    """

    :param mop:
    :param g:
    :param sequence:
    :param S:
    :param A:
    :return:
    """
    procedures = np.shape(S)[1]  # amount of procedures
    for job in sequence:  # machine
        for p in range(procedures):
            machine = g[job, p]
            start_time = S[job, p]
            T = A[job, p]
            plt.barh(machine, T, left=start_time, color=[0.2 * job % 1, 0.1*job % 0.8, 0.6])
            plt.text(start_time + T / 6, machine, 'J%s\n%s' % (job, T), color='k', size=10)
    aom = 0
    for i in range(len(mop)):
        aom += mop[i]
    plt.yticks(np.arange(aom), np.arange(aom))
    plt.ylabel('Machine')
    plt.xlabel('Time')
    plt.show()

