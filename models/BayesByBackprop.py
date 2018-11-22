import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# SEE:
# https://arxiv.org/pdf/1505.05424.pdf
# https://github.com/nitarshan/bayes-by-backprop/blob/master/Weight%20Uncertainty%20in%20Neural%20Networks.ipynb
# https://www.nitarshan.com/bayes-by-backprop/


class GaussianPosterior():
    def __init__(self, mu, rho):
        self.mu = mu
        self.rho = rho
        self.normal = Normal(0, 1)

    @property
    def sigma(self):
        return torch.log(1 + torch.exp(self.rho))

    def sample(self):
        epsilon = self.normal.sample(self.rho.size())
        return self.mu + self.sigma * epsilon

    def log_prob(self, x):
        return Normal(self.mu.data, self.sigma).log_prob(x).sum()


class ScaleMixturePrior():
    def __init__(self, pi, sigma1, sigma2):
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.normal1 = Normal(0, sigma1)
        self.normal2 = Normal(0, sigma2)

    def log_prob(self, x):
        p1 = torch.exp(self.normal1.log_prob(x))
        p2 = torch.exp(self.normal2.log_prob(x))
        return torch.sum(self.pi * p1 + (1 - self.pi) * p2)


PI = 0.5
SIGMA1 = torch.exp(torch.tensor(0).float())
SIGMA2 = torch.exp(torch.tensor(-6).float())


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Priors
        self.weight_prior = ScaleMixturePrior(PI, SIGMA1, SIGMA2)
        self.bias_prior = ScaleMixturePrior(PI, SIGMA1, SIGMA2)

        # Posterior for weights
        self.weight_mu = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(
            torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.weight_posterior = GaussianPosterior(self.weight_mu,
                                                  self.weight_rho)

        # Posterior for biases
        self.bias_mu = nn.Parameter(
            torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(
            torch.Tensor(out_features).uniform_(-5, -4))
        self.bias_posterior = GaussianPosterior(self.bias_mu, self.bias_rho)

        # log P(w) and log q(w|mu, rho) is saved after every forward pass
        self.log_prior = 0
        self.log_posterior = 0

    def forward(self, x):
        weight = self.weight_posterior.sample()
        bias = self.bias_posterior.sample()

        self.log_prior = self.weight_prior.log_prob(
            weight) + self.bias_prior.log_prob(bias)
        self.log_posterior = self.weight_posterior.log_prob(
            weight) + self.bias_posterior.log_prob(bias)

        return F.linear(x, weight, bias)


class BBBNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = BayesianLinear(1, 20)
        self.fc2 = BayesianLinear(20, 20)
        self.fc3 = BayesianLinear(20, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def log_prior(self):
        return self.fc1.log_prior + self.fc2.log_prior + self.fc3.log_prior

    def log_posterior(self):
        return self.fc1.log_posterior + self.fc2.log_posterior + self.fc3.log_posterior

    def elbo(self, data, target, batch_size):
        n_sample = 50  # number of MC draws
        predictions = torch.zeros(n_sample, batch_size)
        log_priors = torch.zeros(n_sample)
        log_posteriors = torch.zeros(n_sample)

        for i in range(n_sample):
            predictions[i] = self(data)
            log_priors[i] = self.log_prior()
            log_posteriors[i] = self.log_posterior()

        log_prior = torch.mean(log_priors)
        log_posterior = torch.mean(log_posteriors)

        # log likelihood assumes gaussian mode
        # where sigma is parameter of the model
        sigma = 2

        data_log_likelihood = 0

        for i in range(n_sample):
            for j in range(batch_size):
                data_log_likelihood += Normal(predictions[i, j],
                                              sigma).log_prob(target[j])

        loss = log_posterior - log_prior - data_log_likelihood

        return loss
