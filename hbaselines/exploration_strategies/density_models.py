import numpy as np
from sklearn.datasets import make_spd_matrix
import tensorflow as tf
import tensorflow_probability as tfp


class GaussianMixtureModel:
    def __init__(self, dimension, num_components=3, max_training_size=2000000, min_training_size=1000, num_steps=10):
        self.num_components = num_components
        self.dimension = dimension
        self.max_training_size = max_training_size
        self.min_training_size = min_training_size
        self.num_steps = num_steps
        self.training_data = []
        self.mixing_coefficients = np.full(self.num_components, 1/self.num_components)
        self.means = np.random.normal(size=[self.num_components, self.dimension])
        self.covariance = []
        for _ in range(self.num_components):
            self.covariance.append(make_spd_matrix(self.dimension))
        self.covariance = np.array(self.covariance)

    def probability(self, obs):
        prob = 0
        obs = tf.cast(obs, dtype=tf.float64)
        for i in range(self.num_components):
            prob += self.mixing_coefficients[i] * tfp.distributions.MultivariateNormalDiag(loc=self.means[i], scale_diag=self.covariance[i]).prob(value=obs)
        return prob

    def recoding_probability(self, obs):
        """Return the prediction of the density model if another obs is observed."""
        return self.probability(obs)

    def update(self, obs):
        """Train the density model using EM algorithm."""
        self.training_data.append(obs)
        if len(self.training_data) < self.min_training_size:
            return
        elif len(self.training_data) > self.max_training_size:
            self.training_data.pop(0)
        trains_X = np.array(self.training_data)
        eps = 1e-9
        for _ in range(self.num_steps):
            # E-step
            multi_norm_pdf = []
            for i in range(self.num_components):
                multi_norm_pdf.append(tfp.distributions.MultivariateNormalDiag(loc=self.means[i], scale_diag=self.covariance[i]).prob(value=obs))
            multi_norm_pdf = np.array(multi_norm_pdf)
            assert multi_norm_pdf.shape == (self.num_components, len(trains_X))
            multi_norm_pdf = multi_norm_pdf * self.mixing_coefficients.reshape(len(self.mixing_coefficients), 1)
            likelihoods = multi_norm_pdf / (np.sum(multi_norm_pdf, axis=0) + eps)
            assert likelihoods.shape == (self.num_components, len(trains_X))
            # M-step
            updated_weights = np.sum(likelihoods, axis=1)
            self.means = np.matmul(likelihoods, trains_X) / (updated_weights.reshape(len(updated_weights), 1) + eps)
            for i in range(self.num_components):
                difference = trains_X - self.means[i]
                self.covariance[i] = np.matmul((likelihoods[i].reshape(len(difference), 1) * difference).T,
                                               difference) / (updated_weights[i] + eps)
            self.mixing_coefficients = updated_weights / (np.sum(updated_weights) + eps)
