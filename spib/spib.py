"""
SPIB: A deep learning-based framework for dimension reduction and MSM of MD trajectories.
Code maintained by Dedi.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/5.0038198
"""
import torch
from torch import nn
import numpy as np
import os
import time
import torch.nn.functional as F

# --------------------
# Model
# --------------------

class SPIB(nn.Module):
    """
    A SPIB model which can be fit to data optimizing for dimension reduction and MSM.
    Parameters
    ----------
    output_dim : int
        Number of initial states.
    data_shape: int...
        A sequence of integers defining the shape of the input data.
    encoder_type: str, default='Nonlinear'
        Encoder type (Linear or Nonlinear)
    z_dim: int, default=2
        Dimension of bottleneck
    lagime : int, default=1
        Time delay delta t in terms of # of minimal time resolution of the trajectory data
    beta: float, default=1e-3
        Hyper-parameter beta makes a trade-off between the predictive capacity and model complexity.
    learning_rate : float, default=1e-3
        The learning rate of the Adam optimizer.
    lr_scheduler_gamma: float, default=1.0
        Multiplicative factor of learning rate decay. lr_scheduler_gamma=1 means no learning rate decay.
    device : torch device, default=torch.device("cpu")
        The device on which the torch modules are executed.
    path : str, default='./SPIB'
        Path to save the training files.
    UpdateLabel : bool, default=True
        Whether to refine the labels during the training process.
    neuron_num1 : int, default=64
        Number of nodes in each hidden layer of the encoder.
    neuron_num2 : int, default=64
        Number of nodes in each hidden layer of the decoder.
    """

    def __init__(self, output_dim, data_shape, encoder_type='Nonlinear', z_dim=2, lagtime=1, beta=1e-3,
                 learning_rate=1e-3, lr_scheduler_gamma=1, device=torch.device("cpu"),
                 path='./spib', UpdateLabel=True, neuron_num1=64, neuron_num2=64, data_transform=None, score_model=None):

        super(SPIB, self).__init__()
        if encoder_type == 'Nonlinear':
            self.encoder_type = 'Nonlinear'
        else:
            self.encoder_type = 'Linear'

        self.z_dim = z_dim
        self.lagtime = lagtime
        self.beta = beta

        self.learning_rate = learning_rate
        self.lr_scheduler_gamma = lr_scheduler_gamma

        self.output_dim = output_dim

        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2

        self.data_transform = data_transform

        self.data_shape = data_shape

        self.UpdateLabel = UpdateLabel

        self.path = path
        self.output_path = self.path + "_d=%d_t=%d_b=%.4f_learn=%f"%(self.z_dim, self.lagtime, self.beta, self.learning_rate)

        self.eps = 1e-10
        self.device = device
        self.score_model = score_model
        if score_model is not None:
            # The collected score. First dimension contains the step, second dimension the score.
            self.score_history = []

        # The collected relative_state_population_change. First dimension contains the step, second dimension the change.
        self.relative_state_population_change_history = []
        # The collected train loss. First dimension contains the step, second dimension the loss. Initially empty.
        self.train_loss_history = []
        # The collected test loss. First dimension contains the step, second dimension the loss. Initially empty.
        self.test_loss_history = []
        # The collected number of states. [ refinement id, number of epoch used for this refinement, number of states ]
        self.convergence_history = []

        # torch buffer, these variables will not be trained
        self.register_buffer('representative_inputs', torch.eye(self.output_dim, np.prod(self.data_shape), device=device, requires_grad=False))

        # create an idle input for calling representative-weights
        # torch buffer, these variables will not be trained
        self.register_buffer('idle_input', torch.eye(self.output_dim, self.output_dim, device=device, requires_grad=False))

        # representative weights
        self.representative_weights = nn.Sequential(
            nn.Linear(self.output_dim, 1, bias=False),
            nn.Softmax(dim=0))

        self.encoder = self._encoder_init()

        if self.encoder_type == 'Nonlinear':
            self.encoder_mean = nn.Linear(self.neuron_num1, self.z_dim)
        else:
            self.encoder_mean = nn.Linear(np.prod(self.data_shape), self.z_dim)

        self.encoder_logvar = nn.Parameter(torch.tensor([0.0]))

        self.decoder = self._decoder_init()

        self.decoder_output = nn.Sequential(
            nn.Linear(self.neuron_num2, self.output_dim),
            nn.LogSoftmax(dim=1))

    def _encoder_init(self):

        modules = []
        if self.data_transform is not None:
            modules += [self.data_transform]

        modules += [nn.Linear(np.prod(self.data_shape), self.neuron_num1)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num1, self.neuron_num1)]
            modules += [nn.ReLU()]

        return nn.Sequential(*modules)

    def _decoder_init(self):
        # cross-entropy MLP decoder
        # output the probability of future state
        modules = [nn.Linear(self.z_dim, self.neuron_num2)]
        modules += [nn.ReLU()]
        for _ in range(1):
            modules += [nn.Linear(self.neuron_num2, self.neuron_num2)]
            modules += [nn.ReLU()]

        return nn.Sequential(*modules)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return eps * std + mu

    def encode(self, inputs):
        enc = self.encoder(inputs)

        if self.encoder_type == 'Nonlinear':
            z_mean = self.encoder_mean(enc)
        else:
            z_mean = self.encoder_mean(inputs)

        z_logvar = self.encoder_logvar

        return z_mean, z_logvar

    def decode(self, z):
        dec = self.decoder(z)
        outputs = self.decoder_output(dec)

        return outputs

    def forward(self, data):
        inputs = torch.flatten(data, start_dim=1)

        z_mean, z_logvar = self.encode(inputs)

        z_sample = self.reparameterize(z_mean, z_logvar)

        outputs = self.decode(z_sample)

        return outputs, z_sample, z_mean, z_logvar

    # Loss function
    # ------------------------------------------------------------------------------
    def calculate_loss(self, data_inputs, data_targets, data_weights):

        # pass through VAE
        outputs, z_sample, z_mean, z_logvar = self.forward(data_inputs)

        # KL Divergence
        log_p = self.log_p(z_sample)
        log_q = -0.5 * torch.sum(z_logvar + torch.pow(z_sample-z_mean, 2) / torch.exp(z_logvar), dim=1)

        # Reconstruction loss is cross-entropy
        # reweighed
        reconstruction_error = torch.sum(data_weights*torch.sum(-data_targets*outputs, dim=1))/data_weights.sum()

        # KL Divergence
        kl_loss = torch.sum(data_weights*(log_q-log_p))/data_weights.sum()

        loss = reconstruction_error + self.beta * kl_loss

        return loss, reconstruction_error.detach().cpu().data, kl_loss.detach().cpu().data

    def log_p (self, z, sum_up=True):
        # get representative_z
        # shape: [output_dim, z_dim]
        representative_z_mean, representative_z_logvar = self.get_representative_z()
        # get representative weights
        # shape: [output_dim, 1]
        w = self.representative_weights(self.idle_input)

        # expand z
        # shape: [batch_size, z_dim]
        z_expand = z.unsqueeze(1)

        representative_mean = representative_z_mean.unsqueeze(0)
        representative_logvar = representative_z_logvar.unsqueeze(0)

        # representative log_q
        representative_log_q = -0.5 * torch.sum(representative_logvar + torch.pow(z_expand-representative_mean, 2)
                                        / torch.exp(representative_logvar), dim=2 )

        if sum_up:
            log_p = torch.sum(torch.log(torch.exp(representative_log_q)@w + self.eps), dim=1)
        else:
            log_p = torch.log(torch.exp(representative_log_q)*w.T + self.eps)

        return log_p

    # the prior
    def get_representative_z(self):
        # calculate representative_means
        # with torch.no_grad():
        X = self.representative_inputs

        # calculate representative_z
        representative_z_mean, representative_z_logvar = self.encode(X)  # C x M

        return representative_z_mean, representative_z_logvar

    def reset_representative(self, representative_inputs):

        # reset the nuber of representative inputs
        self.output_dim = representative_inputs.shape[0]

        # reset representative weights
        self.idle_input = torch.eye(self.output_dim, self.output_dim, device=self.device, requires_grad=False)

        self.representative_weights = nn.Sequential(
            nn.Linear(self.output_dim, 1, bias=False),
            nn.Softmax(dim=0))
        self.representative_weights[0].weight = nn.Parameter(torch.ones([1, self.output_dim], device=self.device))

        # reset representative inputs
        self.representative_inputs = representative_inputs.clone().detach()

    @torch.no_grad()
    def init_representative_inputs(self, inputs, labels):
        state_population = labels.sum(dim=0).cpu()

        # randomly pick up one sample from each initlal state as the initial guess of representative-inputs
        representative_inputs = []

        for i in range(state_population.shape[-1]):
            if state_population[i] > 0:
                index = np.random.randint(0, state_population[i])
                representative_inputs += [inputs[labels[:, i].bool()][index].reshape(1, -1)]
            else:
                # randomly select one sample as the representative input
                index = np.random.randint(0, inputs.shape[0])
                representative_inputs += [inputs[index].reshape(1, -1)]

        representative_inputs = torch.cat(representative_inputs, dim=0)

        self.reset_representative(representative_inputs.to(self.device))

        return representative_inputs

    @torch.no_grad()
    def update_model(self, inputs, input_weights, train_data_labels, test_data_labels, batch_size, threshold=0):
        mean_rep = []
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)

            # pass through VAE
            z_mean, z_logvar = self.encode(batch_inputs)

            mean_rep += [z_mean]

        mean_rep = torch.cat(mean_rep, dim=0)

        state_population = train_data_labels.sum(dim=0).float() / train_data_labels.shape[0]

        # ignore states whose state_population is smaller than threshold to speed up the convergence
        # By default, the threshold is set to be zero
        train_data_labels = train_data_labels[:, state_population > threshold]
        test_data_labels = test_data_labels[:, state_population > threshold]

        # save new guess of representative-inputs
        representative_inputs = []

        for i in range(train_data_labels.shape[-1]):
            weights = input_weights[train_data_labels[:, i].bool()].reshape(-1, 1)
            center_z = ((weights * mean_rep[train_data_labels[:, i].bool()]).sum(dim=0) / weights.sum()).reshape(1, -1)

            # find the one cloest to center_z as representative-inputs
            dist = torch.square(mean_rep - center_z).sum(dim=-1)
            index = torch.argmin(dist)
            representative_inputs += [inputs[index].reshape(1, -1)]

        representative_inputs = torch.cat(representative_inputs, dim=0)

        self.reset_representative(representative_inputs)

        # record the old parameters
        w = self.decoder_output[0].weight[state_population > threshold]
        b = self.decoder_output[0].bias[state_population > threshold]

        # reset the dimension of the output
        self.decoder_output = nn.Sequential(
            nn.Linear(self.neuron_num2, self.output_dim),
            nn.LogSoftmax(dim=1))

        self.decoder_output[0].weight = nn.Parameter(w.to(self.device))
        self.decoder_output[0].bias = nn.Parameter(b.to(self.device))

        return train_data_labels, test_data_labels

    @torch.no_grad()
    def update_labels(self, inputs, batch_size):
        if self.UpdateLabel:
            labels = []

            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i:i+batch_size]

                # pass through VAE
                z_mean, z_logvar = self.encode(batch_inputs)
                log_prediction = self.decode(z_mean)

                # label = p/Z
                labels += [log_prediction.exp()]

            labels = torch.cat(labels, dim=0)
            max_pos = labels.argmax(1)
            labels = F.one_hot(max_pos, num_classes=self.output_dim)

            return labels

    def fit(self, train_dataset, test_dataset, batch_size=128, tolerance=0.001, patience=5, refinements=15,
            mask_threshold=0, index=0):
        """ Fits a SPIB on data.
        Parameters
        ----------
        train_dataset : spib.utils.TimeLaggedDataset
            The data to use for training. Should yield a tuple of batches representing
            instantaneous samples, time-lagged labels and sample weights.
        test_dataset : spib.utils.TimeLaggedDataset
            The data to use for test. Should yield a tuple of batches representing
            instantaneous samples, time-lagged labels and sample weights.
        batch_size : int, default=128
        tolerance: float, default=0.001
            tolerance of loss change for measuring the convergence of the training
        patience: int, default=5
            Number of epochs with the change of the state population smaller than the threshold
            after which this iteration of training finishes.
        refinements: int, default=15
            Number of refinements.
        mask_threshold: float, default=0
            Minimum probability for checking convergence.
        log_interval: int, optional, default=10000
            Number of steps to save the model.
        index: int, optional, default=0

        Returns
        -------
        self : SPIB
            Reference to self.
        """
        self.train()

        # data preparation
        # Specify BatchSampler as sampler to speed up dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(train_dataset), batch_size, False), batch_size=None)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(test_dataset), batch_size, False), batch_size=None)

        # use the training set to initialize the pseudo-inputs
        self.init_representative_inputs(train_dataset.past_data, train_dataset.future_labels)

        # set the optimizer and scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        lr_lambda = lambda epoch: self.lr_scheduler_gamma ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        start = time.time()
        log_path = self.output_path + '_train.log'
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        IB_path = self.output_path + "cpt" + str(index) + "/IB"
        os.makedirs(os.path.dirname(IB_path), exist_ok=True)

        step = 0
        update_times = 0
        unchanged_epochs = 0
        epoch = 0

        # initial state population
        state_population0 = (torch.sum(train_dataset.future_labels, dim=0).float() / train_dataset.future_labels.shape[0]).cpu()
        train_epoch_loss0 = 0

        while True:
            # move to device
            train_epoch_loss = 0
            train_epoch_kl_loss = 0
            train_epoch_reconstruction_error = 0
            for batch_inputs, batch_outputs, batch_weights in train_dataloader:
                step += 1

                loss, reconstruction_error, kl_loss = self.calculate_loss(batch_inputs, batch_outputs, batch_weights)

                # Stop if NaN is obtained
                if(torch.isnan(loss).any()):
                    print("Loss is nan!")
                    raise ValueError

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                weight_sum = batch_weights.sum().cpu()
                train_epoch_loss += loss.detach().cpu().data * weight_sum
                train_epoch_kl_loss += kl_loss * weight_sum
                train_epoch_reconstruction_error += reconstruction_error * weight_sum

                self.train_loss_history += [[step, loss.detach().cpu().data.numpy()]]

            epoch += 1

            with torch.no_grad():
                train_time = time.time() - start

                weight_sum = train_dataset.data_weights.sum().cpu()
                train_epoch_loss /= weight_sum
                train_epoch_kl_loss /= weight_sum
                train_epoch_reconstruction_error /= weight_sum

                print(
                    "Epoch %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                    "Reconstruction loss (train) %f" % (
                        epoch, train_time, train_epoch_loss, train_epoch_kl_loss, train_epoch_reconstruction_error))
                print(
                    "Epoch %i:\tTime %f s\nLoss (train) %f\tKL loss (train): %f\n"
                    "Reconstruction loss (train) %f" % (
                        epoch, train_time, train_epoch_loss, train_epoch_kl_loss,
                        train_epoch_reconstruction_error), file=open(log_path, 'a'))

                test_epoch_loss = 0
                test_epoch_kl_loss = 0
                test_epoch_reconstruction_error = 0
                for batch_inputs, batch_outputs, batch_weights in test_dataloader:

                    loss, reconstruction_error, kl_loss = self.calculate_loss(batch_inputs, batch_outputs, batch_weights)

                    weight_sum = batch_weights.sum().cpu()
                    test_epoch_loss += loss.cpu().data * weight_sum
                    test_epoch_kl_loss += kl_loss * weight_sum
                    test_epoch_reconstruction_error += reconstruction_error * weight_sum

                weight_sum = test_dataset.data_weights.sum().cpu()
                test_epoch_loss /= weight_sum
                test_epoch_kl_loss /= weight_sum
                test_epoch_reconstruction_error /= weight_sum

                print(
                    "Loss (test) %f\tKL loss (test): %f\n"
                    "Reconstruction loss (test) %f" % (
                        test_epoch_loss, test_epoch_kl_loss, test_epoch_reconstruction_error))
                print(
                    "Loss (test) %f\tKL loss (test): %f\n"
                    "Reconstruction loss (test) %f" % (
                        test_epoch_loss, test_epoch_kl_loss, test_epoch_reconstruction_error), file=open(log_path, 'a'))

                self.test_loss_history += [[step, test_epoch_loss.cpu().data.numpy()]]

            # check convergence
            new_train_data_labels = self.update_labels(train_dataset.future_data, batch_size)

            # save the state population
            state_population = (torch.sum(new_train_data_labels, dim=0).float()/new_train_data_labels.shape[0]).cpu()

            print('State population:')
            print('State population:', file=open(log_path, 'a'))
            print(state_population.numpy())
            print(state_population.numpy(), file=open(log_path, 'a'))

            # print the relative state population change
            mask = (state_population0 > mask_threshold)
            relative_state_population_change = torch.sqrt(
                torch.square((state_population - state_population0)[mask] / state_population0[mask]).mean())

            print('Relative state population change=%f' % relative_state_population_change)
            print('Relative state population change=%f' % relative_state_population_change, file=open(log_path, 'a'))

            self.relative_state_population_change_history += [[step, relative_state_population_change.numpy()]]

            # update state_population
            state_population0 = state_population

            scheduler.step()
            if self.lr_scheduler_gamma < 1:
                print("Update lr to %f" % (optimizer.param_groups[0]['lr']))
                print("Update lr to %f" % (optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))

            print('training loss change=%f' % (train_epoch_loss - train_epoch_loss0))
            print('training loss change=%f' % (train_epoch_loss - train_epoch_loss0), file=open(log_path, 'a'))

            # check whether the change of the training loss is smaller than the tolerance
            if torch.abs(train_epoch_loss - train_epoch_loss0) < tolerance:
                unchanged_epochs += 1

                if unchanged_epochs > patience:
                    # save model
                    torch.save({'refinement': update_times+1,
                                'state_dict': self.state_dict()},
                               IB_path + '_%d_cpt.pt' % (update_times+1))
                    torch.save({'optimizer': optimizer.state_dict()},
                               IB_path + '_%d_optim_cpt.pt' % (update_times+1))

                    # check whether only one state is found
                    if torch.sum(state_population>0)<2:
                        print("Only one metastable state is found!")
                        raise ValueError

                    # Stop only if update_times >= min_refinements
                    if self.UpdateLabel and update_times < refinements:

                        train_data_labels = new_train_data_labels
                        test_data_labels = self.update_labels(test_dataset.future_data, batch_size)
                        train_data_labels = train_data_labels.to(self.device)
                        test_data_labels = test_data_labels.to(self.device)

                        update_times += 1
                        print("Update %d\n" % (update_times))
                        print("Update %d\n" % (update_times), file=open(log_path, 'a'))

                        # update the model, and reset the representative-inputs
                        train_data_labels, test_data_labels = self.update_model(train_dataset.past_data,
                                                                                train_dataset.data_weights,
                                                                                train_data_labels, test_data_labels,
                                                                                batch_size, mask_threshold)

                        # save the score
                        if self.score_model is not None:
                            label_numpy = train_data_labels.argmax(1).cpu().numpy()
                            split_indices = np.cumsum(train_dataset.split_lengths)[:-1]
                            split_arrays = np.split(label_numpy, split_indices)
                            self.score_model.fit(split_arrays)
                            train_score = self.score_model.score(split_arrays)

                            label_numpy = test_data_labels.argmax(1).cpu().numpy()
                            split_indices = np.cumsum(test_dataset.split_lengths)[:-1]
                            split_arrays = np.split(label_numpy, split_indices)
                            test_score = self.score_model.score(split_arrays)
                            self.score_history += [[update_times + 1, train_score, test_score]]

                        train_dataset.update_labels(train_data_labels)
                        test_dataset.update_labels(test_data_labels)

                        # initial state population
                        state_population0 = (torch.sum(train_data_labels, dim=0).float() / train_data_labels.shape[0]).cpu()

                        # reset the optimizer and scheduler
                        scheduler.last_epoch = -1

                        # save the history [ refinement id, number of epoch used for this refinement, number of states ]
                        self.convergence_history += [[update_times, epoch, self.output_dim]]

                        # reset epoch and unchanged_epochs
                        epoch = 0
                        unchanged_epochs = 0

                    else:
                        break

            else:
                unchanged_epochs = 0

            train_epoch_loss0 = train_epoch_loss

        # output the saving path
        total_training_time = time.time() - start
        print("Total training time: %f" % total_training_time)
        print("Total training time: %f" % total_training_time, file=open(log_path, 'a'))

        self.eval()

        # label update
        if self.UpdateLabel:
            train_data_labels = self.update_labels(train_dataset.future_data, batch_size)
            test_data_labels = self.update_labels(test_dataset.future_data, batch_size)

            # update the model, and reset the representative-inputs
            train_data_labels, test_data_labels = self.update_model(train_dataset.past_data, train_dataset.data_weights,
                                                                    train_data_labels, test_data_labels, batch_size)

            train_dataset.update_labels(train_data_labels)
            test_dataset.update_labels(test_data_labels)

        # save model
        torch.save({'step': step,
                    'state_dict': self.state_dict()},
                   IB_path + '_final_cpt.pt')
        torch.save({'optimizer': optimizer.state_dict()},
                   IB_path + '_final_optim_cpt.pt')

        # output final result
        self.output_final_result(train_dataset, test_dataset, batch_size, index)
        self.save_representative_parameters(index)
        return self

    @torch.no_grad()
    def output_final_result(self, train_dataset, test_dataset, batch_size, index=0):
        self.eval()

        # data preparation
        # Specify BatchSampler as sampler to speed up dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(train_dataset), batch_size, False), batch_size=None)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, sampler=torch.utils.data.BatchSampler(
            torch.utils.data.SequentialSampler(test_dataset), batch_size, False), batch_size=None)

        summary_path = self.path + '_summary.dat'
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        if os.path.exists(summary_path):
            print("Final Result", file=open(summary_path, 'a'))  # append if already exists
        else:
            print("Final Result", file=open(summary_path, 'w'))

        with torch.no_grad():
            final_result_path = self.output_path + '_final_result' + str(index) + '.npy'
            os.makedirs(os.path.dirname(final_result_path), exist_ok=True)

            final_result = []
            # output the result
            loss, reconstruction_error, kl_loss = [0 for i in range(3)]

            for batch_inputs, batch_outputs, batch_weights in train_dataloader:
                loss1, reconstruction_error1, kl_loss1 = self.calculate_loss(batch_inputs, batch_outputs, batch_weights)

                loss += loss1 * batch_weights.sum()
                kl_loss += kl_loss1 * batch_weights.sum()
                reconstruction_error += reconstruction_error1 * batch_weights.sum()

            # output the result
            weight_sum = train_dataset.data_weights.sum()
            loss /= weight_sum
            kl_loss /= weight_sum
            reconstruction_error /= weight_sum

            final_result += [loss.data.cpu().numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy()]
            print(
                "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f" % (
                    index, loss, kl_loss, reconstruction_error))
            print(
                "Final: %d\nLoss (train) %f\tKL loss (train): %f\n"
                        "Reconstruction loss (train) %f" % (
                    index, loss, kl_loss, reconstruction_error),
                file=open(summary_path, 'a'))

            loss, reconstruction_error, kl_loss = [0 for i in range(3)]

            for batch_inputs, batch_outputs, batch_weights in test_dataloader:
                loss1, reconstruction_error1, kl_loss1 = self.calculate_loss(batch_inputs, batch_outputs, batch_weights)

                loss += loss1 * batch_weights.sum()
                kl_loss += kl_loss1 * batch_weights.sum()
                reconstruction_error += reconstruction_error1 * batch_weights.sum()

            # output the result
            weight_sum = test_dataset.data_weights.sum()
            loss /= weight_sum
            kl_loss /= weight_sum
            reconstruction_error /= weight_sum

            final_result += [loss.cpu().data.numpy(), reconstruction_error.cpu().data.numpy(), kl_loss.cpu().data.numpy()]
            print(
                "Loss (test) %f\tKL loss (test): %f\n"
                "Reconstruction loss (test) %f"
                % (loss, kl_loss, reconstruction_error))
            print(
                "Loss (test) %f\tKL loss (test): %f\n"
                "Reconstruction loss (test) %f"
                % (loss, kl_loss, reconstruction_error), file=open(summary_path, 'a'))

            print("dt: %d\t Beta: %f\t Learning_rate: %f" % (
                self.lagtime, self.beta, self.learning_rate))
            print("dt: %d\t Beta: %f\t Learning_rate: %f" % (
                self.lagtime, self.beta, self.learning_rate),
                file=open(summary_path, 'a'))

            final_result = np.array(final_result)
            np.save(final_result_path, final_result)

    @torch.no_grad()
    def transform(self, data, batch_size=128, to_numpy=False):
        r""" Transforms data through the instantaneous or time-shifted network lobe.
        Parameters
        ----------
        data : numpy ndarray or torch tensor
            The data to transform.
        batch_size : int, default=128
        to_numpy: bool, default=True
            Whether to convert torch tensor to numpy array.
        Returns
        -------
        List of numpy array or torch tensor containing transformed data.
        """
        self.eval()

        if isinstance(data, torch.Tensor):
            inputs = data
        else:
            inputs = torch.from_numpy(data.copy()).float()

        all_prediction = []
        all_z_mean = []
        all_z_logvar = []

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i + batch_size].to(self.device)

            # pass through VAE
            z_mean, z_logvar = self.encode(batch_inputs)

            log_prediction = self.decode(z_mean)

            all_prediction += [log_prediction.exp().cpu()]
            all_z_logvar += [z_logvar.cpu()]
            all_z_mean += [z_mean.cpu()]

        all_prediction = torch.cat(all_prediction, dim=0)
        all_z_logvar = torch.cat(all_z_logvar, dim=0)
        all_z_mean = torch.cat(all_z_mean, dim=0)

        labels = all_prediction.argmax(1)

        if to_numpy:
            return labels.numpy().astype(np.int32), all_prediction.numpy().astype(np.double), \
                   all_z_mean.numpy().astype(np.double), all_z_logvar.numpy().astype(np.double)
        else:
            return labels, all_prediction, all_z_mean, all_z_logvar

    @torch.no_grad()
    def save_representative_parameters(self, index=0):

        # output representative centers
        representative_path = self.output_path + '_representative_inputs' + str(index) + '.npy'
        representative_weight_path = self.output_path + '_representative_weight' + str(index) + '.npy'
        representative_z_mean_path = self.output_path + '_representative_z_mean' + str(index) + '.npy'
        representative_z_logvar_path = self.output_path + '_representative_z_logvar' + str(index) + '.npy'
        os.makedirs(os.path.dirname(representative_path), exist_ok=True)

        np.save(representative_path, self.representative_inputs.cpu().data.numpy())
        np.save(representative_weight_path, self.representative_weights(self.idle_input).cpu().data.numpy())

        representative_z_mean, representative_z_logvar = self.get_representative_z()
        np.save(representative_z_mean_path, representative_z_mean.cpu().data.numpy())
        np.save(representative_z_logvar_path, representative_z_logvar.cpu().data.numpy())







