=====
Usage
=====

To run spib, we proposed two scripts in ``scripts``:

For preliminary analyses
------------------------

::

   python test_model.py    -dt # Time delay delta t in terms of # of minimal time resolution of the trajectory data
               -d  # Dimension of RC or bottleneck
               -encoder_type   # Encoder type (Linear or Nonlinear)
               -n1 # Number of nodes in each hidden layer of the encoder
               -n2 # Number of nodes in each hidden layer of the decoder
               -bs # Batch size
               -threshold  # Threshold of the predicted state population change used to measure the convergence of training for each iteration
               -patience   # Number of epochs with the change of the state population smaller than the threshold after which this iteration of training finishes
               -refinements    # Number of refinements
               -lr # Learning rate of Adam optimizer
               -b  # Hyperparameter beta
               -label  # Path to the initial state labels
               -traj   # Path to the trajectory data
               -w  # Path to the weights of the samples
               -seed   # Random seed
               -UpdateLabel    # Whether to refine the labels during the training process
               -SaveTrajResults    # Whether save trajectory results

Example
~~~~~~~

Train and test SPIB on the four-well analytical potential:

::

   python test_model.py -dt 50 -d 1 -encoder_type Nonlinear -bs 512 -threshold 0.01 -patience 2 -refinements 8 -lr 0.001 -b 0.01 -seed 0 -label examples/Four_Well_beta3_gamma4_init_label10.npy -traj examples/Four_Well_beta3_gamma4_traj_data.npy

For advanced analyses
---------------------

::

   python test_model_advanced.py   -config # Input the configuration file 

Here, a configuration file in INI format is supported, which allows a
more flexible control of the training process. A sample configuration
file is shown in the ``scripts/examples`` subdirectory. Two advanced features
are included: 

* It supports simple grid search to tune the hyper-parameters; 
* It also allows multiple trajectories with different weights as the input data;
* It supports the use of learning rate decay to speed up the convergence especially for the first refinement.

Example
~~~~~~~

Train and test SPIB on the four-well analytical potential:

::

   python test_model_advanced.py -config examples/sample_config.ini