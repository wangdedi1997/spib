"""
SPIB: A deep learning-based framework for dimension reduction and MSM of MD trajectories.
Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Data Processing
# ------------------------------------------------------------------------------

class DataNormalize(nn.Module):
    def __init__(self, mean, std):
        super(DataNormalize, self).__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def forward(self, x):
        # Assuming x is a tensor with shape (batch_size, features)
        normalized_x = (x - self.mean[None, :]) / self.std[None, :]
        return normalized_x


def prepare_data(data_list, label_list, weight_list, output_dim, train_indices,
                 test_indices, lagtime=1, subsampling_timestep=1, device=torch.device("cpu")):
    r"""This can be used to prepare the data for spib training and validation.
        Parameters
        ----------
        data_list : List of trajectory data
            The data which is wrapped into a dataset
        label_list : List of corresponding labels
            Corresponding label data. Must be of same length.
        weight_list: List of corresponding weights, optional, default=None
            Corresponding weight data. Must be of same length.
        output_dim: int
            The total number of states in label_list
        train_indices: a sequence of indices,
            Indices in the whole set selected for training set
        test_indices: a sequence of indices,
            Indices in the whole set selected for test set
        lagtime: int, default=1
            The lag time used to produce timeshifted blocks.
        subsampling_timestep: int, default=1
            The step size for subsampling
        device: torch device, default=torch.device("cpu")
            The device on which the torch modules are executed.
    """

    train_data = [data_list[i] for i in train_indices]
    test_data = [data_list[i] for i in test_indices]

    train_labels = [label_list[i] for i in train_indices]
    test_labels = [label_list[i] for i in test_indices]

    if weight_list is None:
        train_dataset = TimeLaggedDataset(train_data, train_labels, None, lagtime=lagtime,
                                          subsampling_timestep=subsampling_timestep,
                                          output_dim=output_dim, device=device)
        test_dataset = TimeLaggedDataset(test_data, test_labels, None, lagtime=lagtime,
                                         subsampling_timestep=subsampling_timestep,
                                         output_dim=output_dim, device=device)
    else:
        train_weights = [weight_list[i] for i in train_indices]
        test_weights = [weight_list[i] for i in test_indices]

        train_dataset = TimeLaggedDataset(train_data, train_labels, train_weights, lagtime=lagtime,
                                          subsampling_timestep=subsampling_timestep,
                                          output_dim=output_dim, device=device)
        test_dataset = TimeLaggedDataset(test_data, test_labels, test_weights, lagtime=lagtime,
                                         subsampling_timestep=subsampling_timestep,
                                         output_dim=output_dim, device=device)

    return train_dataset, test_dataset

class TimeLaggedDataset(torch.utils.data.Dataset):
    r""" High-level container for time-lagged time-series data.
    This can be used together with pytorch data tools, i.e., data loaders and other utilities.
    Parameters
    ----------
    data_list : List of trajectory data
        The data which is wrapped into a dataset
    label_list : List of corresponding labels
        Corresponding label data. Must be of same length.
    weight_list: List of corresponding weights, optional, default=None
        Corresponding weight data. Must be of same length.
    lagtime: int, default=1
        The lag time used to produce timeshifted blocks.
    subsampling_timestep: int, default=1
        The step size for subsampling
    device: torch device, default=torch.device("cpu")
        The device on which the torch modules are executed.
    """

    def __init__(self, data_list, label_list, weight_list=None, lagtime=1, subsampling_timestep=1,
                 output_dim=None, device=torch.device("cpu")):
        assert len(data_list) == len(label_list), \
            f"Length of trajectory for data_list and label_list does not match ({len(data_list)} != {len(label_list)})"

        self.lagtime = lagtime
        self.subsampling_timestep = subsampling_timestep
        self.traj_num = len(data_list)

        if weight_list is None:
            # set weights as ones
            weight_list = [np.ones_like(label_list[i]) for i in range(len(label_list))]

        data_init_list = []
        for i in range(len(data_list)):
            data_init_list += [data_init(self.lagtime, self.subsampling_timestep, data_list[i], label_list[i], weight_list[i])]
        self.data_weights = torch.from_numpy(np.concatenate([data_init_list[i][3] for i in range(len(data_init_list))],
                                                            axis=0)).float().to(device)

        self.past_data = torch.from_numpy(np.concatenate([data_init_list[i][0] for i in range(len(data_init_list))],
                                                         axis=0)).float().to(device)
        self.future_data = torch.from_numpy(np.concatenate([data_init_list[i][1] for i in range(len(data_init_list))],
                                                           axis=0)).float().to(device)
        label_data = torch.from_numpy(np.concatenate([data_init_list[i][2] for i in range(len(data_init_list))],
                                                     axis=0)).long()

        # record the lengths of trajectories
        self.split_lengths = [len(data_init_list[i][2]) for i in range(len(data_init_list))]

        if output_dim == None:
            self.future_labels = F.one_hot(label_data).to(device)
        else:
            self.future_labels = F.one_hot(label_data, num_classes=output_dim).to(device)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.past_data)

    def __getitem__(self, index):
        'Generates one sample of data'
        return self.past_data[index], self.future_labels[index], self.data_weights[index]

    def update_labels(self, future_labels):
        self.future_labels = future_labels

def data_init(dt, timestep, traj_data, traj_label, traj_weights):
    assert len(traj_data) == len(traj_label), \
            f"Length of trajectory for traj_data and traj_label does not match ({len(traj_data)} != {len(traj_label)})"

    past_data = traj_data[:(len(traj_data)-dt):timestep]
    future_data = traj_data[(dt):len(traj_data):timestep]
    label = traj_label[(dt):len(traj_data):timestep]

    if traj_weights is not None:
        assert len(traj_data)==len(traj_weights)
        weights = traj_weights[:(len(traj_data)-dt):timestep]
    else:
        weights = None

    return past_data, future_data, label, weights



