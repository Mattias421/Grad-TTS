# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import numpy as np
from scipy import integrate
from likelihood import utils_sde as mutils

import matplotlib.pyplot as plt

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    rsde = sde.reverse(model, probability_flow=True)
    x = x * sde.mask
    return rsde.sde(x, t)[0] * sde.mask

  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def likelihood_fn(model, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      ode_deltas = []
      ode_t = []
      def ode_func(t, x):
        sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
        logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
        # print(drift)
        # print(sample)
        # print(logp_grad, t)
        ode_deltas.append(logp_grad)
        ode_t.append(t)
        # plt.imshow(sample.cpu().squeeze())
        # plt.title(str(t))
        # plt.savefig(f'../delta_plots/libritts_rk45/{int(t*100)}.png')
        return np.concatenate([drift, logp_grad], axis=0)
      



      def trapezoidal_integral(time_steps, y_values):
        if len(time_steps) != len(y_values):
            raise ValueError("The number of time steps and y values must be the same.")

        integral = 0.0
        for i in range(1, len(time_steps)):
            dt = time_steps[i] - time_steps[i - 1]
            integral += 0.5 * (y_values[i] + y_values[i - 1]) * dt

        return integral
      
      def reverse(data, num_its):

        data = sde.prioir_sampling() # X_T for reverse mode
        data = data * sde.mask
        init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)

        x = init
        h = 1 / num_its # step size
        for i in range(num_its):
          t = (1 - (i + 0.5)*h)
          dxt = ode_func(t, x) * h

          x = (x - dxt)

        plt.clf()
        plt.scatter(ode_t, ode_deltas)
        plt.xlabel('t')
        plt.ylabel('logp_grad')
        plt.title(f'Speech at {num_its} iterations')
        plt.savefig(f'../delta_plots/speech_{num_its}.png')

      def forward(data, num_its=52):
        data = data * sde.mask
        init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)

        x = init
        h = 1 / num_its # step size
        for i in range(num_its):
          t = ((i + 0.5)*h)
          dxt = ode_func(t, x) * h

          x = (x + dxt)

        plt.clf()
        plt.scatter(ode_t, ode_deltas)
        plt.xlabel('t')
        plt.ylabel('logp_grad')
        plt.title(f'Speech at {num_its} iterations')
        plt.savefig(f'../delta_plots/speech_{num_its}.png')

        area = trapezoidal_integral(ode_t, ode_deltas)
        sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        print('logpdelta ', area)
        print('logpxT ', sde.prior_logp(sample))

        logpx0 = area + sde.prior_logp(sample).item()
        print('logpx0 ', area + sde.prior_logp(sample).item())

        return logpx0 / np.log(2) / np.prod(shape[1:])


      # print(forward(data, 50), 'bpd')
      
      # return forward(data, 50)

      # sample
      def ode_sample():
        rand_data = sde.prior_sampling() # X_T for reverse mode
        init = np.concatenate([mutils.to_flattened_numpy(rand_data), np.zeros((shape[0],))], axis=0)

        solution = integrate.solve_ivp(ode_func, (sde.T, eps), init, rtol=rtol, atol=atol, method=method)
        x_0 = solution.y[:, -1][:-shape[0]].reshape(shape)
        # zp = solution.y[:, -1]
        # delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
        # print(delta_logp)
        loss = torch.nn.MSELoss()
        mse = loss(data, torch.tensor(x_0).to(data))
        print('mse', mse)
        plt.imshow(x_0[0])
        plt.savefig('ode_sample.png')

        return mse
      

      # return ode_sample()

      # find likelihood
      data = data * sde.mask
      init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      ode_deltas = []
      ode_t = []
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
      plt.clf()
      plt.scatter(ode_t[1:], ode_deltas[1:])
      plt.savefig(f'../delta_plots/speech_rk45.png')
      # print(solution)
      nfe = solution.nfev
      zp = solution.y[:, -1]


      delta_logp_graph = solution.y[:, :][-1, :]
      plt.clf()
      plt.title('delta logp')
      plt.plot(delta_logp_graph)
      plt.savefig('../delta_plots/delta_logp.png')
      z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      prior_logp = sde.prior_logp(z)

      # print('gaussian')
      # print('mean ', torch.mean(z), 'spread ', torch.std(z))
      # print('text')
      # print('mean ', torch.mean(sde.mu), 'spread ', torch.std(sde.mu))

      # print('change in mean ', (torch.mean(z) - torch.mean(sde.mu)))

      # print('delta logp', delta_logp)
      # print('prior logp', prior_logp)
      bpd = -(prior_logp + delta_logp) / np.log(2)
      N = np.prod(shape[1:])
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      bpd = bpd + offset

      logpx0 = prior_logp + delta_logp

      print('logp(x0) ', logpx0)

      bpd = - logpx0 / np.log(2) / N 
      return bpd.item()

  return likelihood_fn