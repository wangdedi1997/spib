"""
SPIB: A deep learning-based framework to learn RCs 
from MD trajectories. Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import numpy as np

# Data Processing
# ------------------------------------------------------------------------------

def data_init(t0, dt, traj_data, traj_label, traj_weights):
    assert len(traj_data)==len(traj_label)
    
    # skip the first t0 data
    past_data = traj_data[t0:(len(traj_data)-dt)]
    future_data = traj_data[(t0+dt):len(traj_data)]
    label = traj_label[(t0+dt):len(traj_data)]
    
    # data shape
    data_shape = past_data.shape[1:]
    
    n_data = len(past_data)
    
    # 90% random test/train split
    p = np.random.permutation(n_data)
    past_data = past_data[p]
    future_data = future_data[p]
    label = label[p]
    
    past_data_train = past_data[0: (9 * n_data) // 10]
    past_data_test = past_data[(9 * n_data) // 10:]
    
    future_data_train = future_data[0: (9 * n_data) // 10]
    future_data_test = future_data[(9 * n_data) // 10:]
    
    label_train = label[0: (9 * n_data) // 10]
    label_test = label[(9 * n_data) // 10:]
    
    if traj_weights != None:
        assert len(traj_data)==len(traj_weights)
        weights = traj_weights[t0:(len(traj_data)-dt)]
        weights = weights[p]
        weights_train = weights[0: (9 * n_data) // 10]
        weights_test = weights[(9 * n_data) // 10:]
    else:
        weights_train = None
        weights_test = None
    
    return data_shape, past_data_train, future_data_train, label_train, weights_train,\
        past_data_test, future_data_test, label_test, weights_test

# Train and test model
# ------------------------------------------------------------------------------

def sample_minibatch(past_data, data_labels, data_weights, indices):
    sample_past_data = past_data[indices]
    sample_data_labels = data_labels[indices]
    
    if data_weights == None:
        sample_data_weights = None
    else:
        sample_data_weights = data_weights[indices]
    
    
    return sample_past_data, sample_data_labels, sample_data_weights

