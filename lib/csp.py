import numpy as np
import scipy.linalg as la


def CSP(data, label):
    a = np.where(label == 1)
    data_h = [data[i] for i in np.where(label == 0)[0]]
    data_f = [data[i] for i in np.where(label == 1)[0]]

    Rh = np.dot(data_h[0], np.transpose(data_h[0])) / np.trace(np.dot(data_h[0], np.transpose(data_h[0])))
    for i in range(1, len(data_h)):
        Rh += np.dot(data_h[i], np.transpose(data_h[i])) / np.trace(
            np.dot(data_h[i], np.transpose(data_h[i])))  # covariance
    Rh = Rh / len(data_h)

    Rf = np.dot(data_f[0], np.transpose(data_f[0])) / np.trace(np.dot(data_f[0], np.transpose(data_f[0])))
    for i in range(1, len(data_f)):
        Rf += np.dot(data_f[i], np.transpose(data_f[i])) / np.trace(
            np.dot(data_f[i], np.transpose(data_f[i])))  # covariance
    Rf = Rf / len(data_f)

    filter = spatialFilter(Rh, Rf)

    return filter


def spatialFilter(Rh, Rf):
    R = Rh + Rf
    E0, U0 = la.eig(R)

    # sorted descending order
    ord = np.argsort(E0)
    ord = ord[::-1]
    E0 = E0[ord]
    U0 = U0[:, ord]

    # whitening transformation matrix
    P = np.dot(np.sqrt(la.inv(np.diag(E0))), np.transpose(U0))

    # average covariance matrices
    # Sh = np.dot(P, np.dot(Rh, np.transpose(P)))
    # Sf = np.dot(P, np.dot(Rf, np.transpose(P)))
    Sh = np.dot(np.dot(P, Rh), np.transpose(P))
    Sf = np.dot(np.dot(P, Rf), np.transpose(P))

    # generalized eigenvector and eigenvalues
    E, U = la.eig(Sh, Sf)
    ord1 = np.argsort(E)
    ord1 = ord1[::-1]
    E = E[ord1]
    U = U[:, ord1]
    # U = U[:, :3]
    U = np.append(U[:, :3], U[:, 19:], axis=1)

    # the projection matrix
    W = np.dot(np.transpose(U), P)

    return W


def align(data):
    """data alignment"""
    data_align = []
    length = len(data)
    rf_matrix = np.dot(data[0], np.transpose(data[0]))
    for i in range(1, length):
        rf_matrix += np.dot(data[i], np.transpose(data[i]))
    rf_matrix /= length

    rf = la.inv(la.sqrtm(rf_matrix))
    for i in range(length):
        data_align.append(np.dot(rf, data[i]))

    return np.asarray(data_align).squeeze()
