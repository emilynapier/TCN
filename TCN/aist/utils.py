import torch
from os import listdir
import scipy
from scipy.spatial.transform import Rotation as R
import math
from numpy import linalg as LA
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import random
from sklearn.model_selection import train_test_split


bpm_dict = {
    #music file: bpm
    'mBR0': 80,
    'mBR1': 90,
    'mBR2': 100,
    'mBR3': 110,
    'mBR4': 120,
    'mBR5': 130,
    'mPO0': 80,
    'mPO1': 90,
    'mPO2': 100,
    'mPO3': 110,
    'mPO4': 120,
    'mPO5': 130,
    'mLO0': 80,
    'mLO1': 90,
    'mLO2': 100,
    'mLO3': 110,
    'mLO4': 120,
    'mLO5': 130,
    'mMH0': 80,
    'mMH1': 90,
    'mMH2': 100,
    'mMH3': 110,
    'mMH4': 120,
    'mMH5': 130,
    'mLH0': 80,
    'mLH1': 90,
    'mLH2': 100,
    'mLH3': 110,
    'mLH4': 120,
    'mLH5': 130,
    'mHO0':	110,
    'mHO1':	115,
    'mHO2':	120,
    'mHO3':	125,
    'mHO4':	130,
    'mHO5':	135,
    'mWA0': 80,
    'mWA1': 90,
    'mWA2': 100,
    'mWA3': 110,
    'mWA4': 120,
    'mWA5': 130,
    'mKR0': 80,
    'mKR1': 90,
    'mKR2': 100,
    'mKR3': 110,
    'mKR4': 120,
    'mKR5': 130,
    'mJS0': 80,
    'mJS1': 90,
    'mJS2': 100,
    'mJS3': 110,
    'mJS4': 120,
    'mJS5': 130,
    'mJB0': 80,
    'mJB1': 90,
    'mJB2': 100,
    'mJB3': 110,
    'mJB4': 120,
    'mJB5': 130,
}
bpm_labels = {
    #bpm: label
    80: 0,
    90: 1,
    100: 2,
    110: 3,
    115: 4,
    120: 5,
    125: 6,
    130: 7,
    135: 8,
}

def scipy_aa_to_geo(x):
    n, d = x.shape
    j = d//3
    x = x.reshape(n * j, 3)
    rotmat = R.from_rotvec(x).as_matrix()  # should now be (n, j, 3, 3)
    rotmat = rotmat.reshape(n, j, 3, 3)
    A, B = rotmat[1:], rotmat[:-1]
    rotdiff = np.einsum('nmij,nmkj->nmik', B, A)  # batched matmul between A and B.T
    rotdiff = rotdiff.reshape(-1, 3, 3)
    rotdiff = R.from_matrix(rotdiff).as_rotvec()
    rotdiff = rotdiff.reshape(n - 1, j, 3)
    result = LA.norm(rotdiff, axis=2)
    return result  # this should already be the right shape

def data_generator(root, batch_size):

    with open('pose_train.txt') as f:
        train_files = f.read().splitlines()
    with open('pose_val.txt') as f:
        val_files = f.read().splitlines()

    train_input = []
    train_labels = []
    test_input = []
    test_labels = []

    keep = 0.9
    l = 300
    for motion_pkl in listdir(root):
        data = np.load(root / motion_pkl, allow_pickle=True)['smpl_poses']
        data = scipy_aa_to_geo(data)  # convert to geodesic angle
        n = len(data)
        skip = int(n * ((1. - keep) / 2.))
        data = data[skip:n - skip]

        motion_pkl = motion_pkl.split('.')[0]

        label = bpm_labels[bpm_dict[motion_pkl[17:21]]]
        for t in range(len(data)//l):
            for i in range(24):
                if motion_pkl in train_files:
                    train_input.append(data[t * l:(t + 1) * l, i])
                    train_labels.append(label)
                elif motion_pkl in val_files:
                    test_input.append(data[t * l:(t + 1) * l, i])
                    test_labels.append(label)

    train_input = torch.Tensor(np.array(train_input))  # transform to torch tensor
    train_input = train_input.reshape(train_input.shape[0], 1, train_input.shape[1])
    train_labels = torch.LongTensor(train_labels)
    print(train_input.shape, train_labels.shape)

    test_input = torch.Tensor(np.array(test_input))  # transform to torch tensor
    test_input = test_input.reshape(test_input.shape[0], 1, test_input.shape[1])
    test_labels = torch.LongTensor(test_labels)
    print(test_input.shape, test_labels.shape)

    train_set = TensorDataset(train_input, train_labels)
    test_set = TensorDataset(test_input, test_labels)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)
    return train_loader, test_loader

def synthetic_data_generator(batch_size):
    N = 1000
    t = 300
    noise = 0.01
    data = np.zeros((N, t))
    labels = np.zeros(N)

    bpm_list = [80, 90, 100, 110, 115, 120, 125, 130, 135]

    for i in range(N):
        # get bpm labels
        bpm = bpm_list[random.randint(0, 8)]
        labels[i] = bpm_labels[bpm]
        # calculate parameters
        theta_start = random.uniform(0, 2 * np.pi)
        theta_end = theta_start + random.uniform(0, 2 * np.pi)
        frames = int(60 / bpm * 60)  # bps * 60 fps
        beat_frames = int(
            frames // random.randint(3, 6))  # 6 is too high for large joint angles, will have to revisit this
        delay_frames = frames - beat_frames
        # compute joint angles
        beat = [theta + np.random.normal(0, noise) for theta in np.linspace(theta_start, theta_end, beat_frames)]
        delay = [theta + np.random.normal(0, noise) for theta in np.linspace(theta_end, theta_start, delay_frames)]
        # populate data array
        data[i, :beat_frames] = beat
        data[i, beat_frames:frames] = delay
        for f in range(1, (t // frames) - 1):
            data[i, f * frames:(f + 1) * frames] = data[i, :frames]

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

    tensor_x = torch.Tensor(X_train)  # transform to torch tensor
    tensor_x = tensor_x.reshape(tensor_x.shape[0], 1, tensor_x.shape[1])
    tensor_y = torch.LongTensor(y_train)
    train_dataset = TensorDataset(tensor_x, tensor_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    tensor_x = torch.Tensor(X_test)  # transform to torch tensor
    tensor_x = tensor_x.reshape(tensor_x.shape[0], 1, tensor_x.shape[1])
    tensor_y = torch.LongTensor(y_test)
    test_dataset = TensorDataset(tensor_x, tensor_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader
