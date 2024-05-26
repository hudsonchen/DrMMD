from flax import struct
import jax.numpy as jnp
import jax
import time
from functools import partial
from dataclasses import dataclass
from kwgflows.base import DiscreteProbability
from kwgflows.divergences.base import KernelizedDivergence
from kwgflows.rkhs.kernels import base_kernel
from kwgflows.rkhs.rkhs import rkhs_element
from kwgflows.typing import Array, Scalar, Distribution
from typing import Callable


def nystrom_inv(matrix, eps, m):
    rng_key = jax.random.PRNGKey(int(time.time()))
    n = matrix.shape[0]
    matrix_mean = jnp.mean(matrix)
    matrix = matrix / matrix_mean  # Scale the matrix to avoid numerical issues

    # Randomly select m columns
    rng_key, _ = jax.random.split(rng_key)
    idx = jax.random.choice(rng_key, n, (m, ), replace=False)

    W = matrix[idx, :][:, idx]
    U, s, V = jnp.linalg.svd(W)

    U_recon = jnp.sqrt(m / n) * matrix[:, idx] @ U @ jnp.diag(1. / s)
    S_recon = s * (n / m)

    Sigma_inv = (1. / eps) * jnp.eye(n)
    approx_inv = Sigma_inv - Sigma_inv @ U_recon @ jnp.linalg.inv(jnp.diag(1. / S_recon) + U_recon.T @ Sigma_inv @ U_recon) @ U_recon.T @ Sigma_inv
    approx_inv = approx_inv / matrix_mean  # Don't forget the scaling!
    return approx_inv

class mmd(struct.PyTreeNode):
    kernel: base_kernel
    
    def get_witness_function(
        self, z, X, Y
    ) -> Scalar:
        z = z[None, :]
        K_zX = self.kernel.make_distance_matrix(z, X)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (K_zY.mean(1) - K_zX.mean(1)).squeeze()

    def get_first_variation(self, X, Y) -> Callable:
        return partial(self.get_witness_function, X=X, Y=Y)

    def __call__(self, X, Y) -> Scalar:
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()

class mmd_fixed_target:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.g = g
        # kernel: base_kernel
        # lmbda: float
    
    def pre_compute(self, X, Y, lmbda):
        self.X = X
    
    def get_witness_function(
        self, z, Y, lmbda
    ) -> Scalar:
        z = z[None, :]
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_zY = self.kernel.make_distance_matrix(z, Y)
        return (K_zY.mean(1) - K_zX.mean(1)).squeeze()

    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.get_witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y) -> Scalar: # mmd^2
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        return K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()


class drmmd(struct.PyTreeNode):
    kernel: base_kernel
    lmbda: float
    
    def witness_function(
        self, z, X, Y
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, X)
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        inv_K_XX = jnp.linalg.inv(K_XX + N * self.lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_zY.mean(axis=1) - K_zX.mean(axis=1)
        part2 = - (K_zX @ inv_K_XX @ K_XY).mean(axis=1)
        part3 = K_zX @ inv_K_XX @ K_XX.mean(axis=1)
        return (part1 + part2 + part3).squeeze() / self.lmbda * 2 * (1 + self.lmbda)
    
    def get_first_variation(self, X, Y) -> Callable:
        return partial(self.witness_function, X=X, Y=Y)

    def __call__(self, X, Y) -> Scalar: # drmmd
        N, M = Y.shape[0], X.shape[0]
        K_XX = self.kernel.make_distance_matrix(X, X)
        K_XY = self.kernel.make_distance_matrix(X, Y)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        inv_K_XX = jnp.linalg.inv(K_XX + N * self.lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()
        part2 = -(K_XY.T @ inv_K_XX @ K_XY).mean()
        part3 = (K_XX.T @ inv_K_XX @ K_XY).mean() * 2
        part4 = -(K_XX.T @ inv_K_XX @ K_XX).mean()

        return (part1 + part2 + part3 + part4) / self.lmbda * (1 + self.lmbda)


class drmmd_fixed_target:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.g = g

    def pre_compute(self, X, Y, lmbda):
        self.X = X
        K_XX = self.kernel.make_distance_matrix(X, X)
        if self.args.nystrom > 0:
            self.K_XX_inv = nystrom_inv(K_XX, self.lmbda, self.args.nystrom)
        else:
            self.K_XX_inv = jnp.linalg.inv(K_XX + self.X.shape[0] * self.lmbda * jnp.eye(K_XX.shape[0]))
        return
    
    def witness_function(
        self, z, Y, lmbda
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)

        part1 = K_zY.mean(axis=1) - K_zX.mean(axis=1)
        part2 = - (K_zX @ self.K_XX_inv @ K_XY).mean(axis=1)
        part3 = (K_zX @ self.K_XX_inv @ K_XX).mean(axis=1)
        return (part1 + part2 + part3).squeeze() / self.lmbda * 2 * (1 + self.lmbda)
    
    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y) -> Scalar:
        N, M = Y.shape[0], self.X.shape[0]
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        K_YY = self.kernel.make_distance_matrix(Y, Y)

        part1 = K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()
        part2 = -(K_XY.T @ self.K_XX_inv @ K_XY).mean()
        part3 = (K_XX.T @ self.K_XX_inv @ K_XY).mean() * 2
        part4 = -(K_XX.T @ self.K_XX_inv @ K_XX).mean()

        return (part1 + part2 + part3 + part4) / self.lmbda * (1 + self.lmbda)
    

class drmmd_fixed_target_adaptive:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.args = args
        self.g = g

    def pre_compute(self, X, Y, lmbda):
        self.X = X
        self.drmmd = self.__call__(Y, lmbda)
        return
    
    def witness_function(
        self, z, Y, lmbda
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)

        K_XX_inv = jnp.linalg.inv(K_XX + self.X.shape[0] * lmbda * jnp.eye(K_XX.shape[0]))
        part1 = K_zY.mean(axis=1) - K_zX.mean(axis=1)
        part2 = - (K_zX @ K_XX_inv @ K_XY).mean(axis=1)
        part3 = (K_zX @ K_XX_inv @ K_XX).mean(axis=1)
        return (part1 + part2 + part3).squeeze() / lmbda * 2 * (1 + lmbda)
    
    def get_first_variation(self, Y, lmbda) -> Callable:
        return partial(self.witness_function, Y=Y, lmbda=lmbda)

    def __call__(self, Y, lmbda) -> Scalar:
        N, M = Y.shape[0], self.X.shape[0]
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)
        K_YY = self.kernel.make_distance_matrix(Y, Y)
        K_XX_inv = jnp.linalg.inv(K_XX + self.X.shape[0] * lmbda * jnp.eye(K_XX.shape[0]))

        part1 = K_YY.mean() + K_XX.mean() - 2 * K_XY.mean()
        part2 = -(K_XY.T @ K_XX_inv @ K_XY).mean()
        part3 = (K_XX.T @ K_XX_inv @ K_XY).mean() * 2
        part4 = -(K_XX.T @ K_XX_inv @ K_XX).mean()

        return (part1 + part2 + part3 + part4) / lmbda * (1 + lmbda)


class spectral_drmmd_fixed_target:
    def __init__(self, args, kernel, g):
        self.kernel = kernel
        self.lmbda = args.lmbda
        self.args = args
        self.g = g
        # kernel: base_kernel
        # lmbda: float

    def pre_compute(self, X):
        self.X = X
        M = self.X.shape[0]

        # Centering
        one_M = jnp.ones([M, 1])
        Hs = jnp.eye(M) - one_M @ one_M.T / M
        tilde_Hs = M / (M - 1) * Hs
        from jax.scipy.linalg import sqrtm
        self.tilde_Hs_half = sqrtm(tilde_Hs + 0.000 * jnp.eye(M)).real
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        HKH = (self.tilde_Hs_half.T @ K_XX @ self.tilde_Hs_half) / M
        eig_val, eig_vec = jnp.linalg.eigh(HKH + 1e-10 * jnp.eye(M))
        
        self.G = jnp.zeros([M, M])
        for i in range(M):
            self.G += ( (self.g(eig_val[i]) - self.g(0)) * eig_vec[:, i:i+1] @ eig_vec[:, i:i+1].T ) / eig_val[i]
 
        # Uncentered
        # K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        # eig_val, eig_vec = jnp.linalg.eigh(K_XX / M + 1e-10 * jnp.eye(M))
        # self.G = jnp.zeros([M, M])
        # for i in range(M):
        #     self.G += ( (self.g(eig_val[i]) - self.g(0)) * eig_vec[:, i:i+1] @ eig_vec[:, i:i+1].T ) / eig_val[i]
        return self.G
    
    def witness_function(
        self, z, Y
    ) -> Scalar:
        z = z[None, :]
        N, M = Y.shape[0], self.X.shape[0]
        K_zY = self.kernel.make_distance_matrix(z, Y)
        K_zX = self.kernel.make_distance_matrix(z, self.X)
        K_XX = self.kernel.make_distance_matrix(self.X, self.X)
        K_XY = self.kernel.make_distance_matrix(self.X, Y)

        # Centered
        part1 = self.g(0) * K_zY.mean(axis=1) + (K_zX @ self.tilde_Hs_half @ self.G @ self.tilde_Hs_half @ K_XY).mean(axis=1) / M
        part2 = - self.g(0) * K_zX.mean(axis=1) - (K_zX @ self.tilde_Hs_half @ self.G @ self.tilde_Hs_half @ K_XX).mean(axis=1) / M

        # Uncentered
        # part1 = self.g(0) * K_zY.mean(axis=1) + (K_zX @ self.G @ K_XY).mean(axis=1) / M
        # part2 = - self.g(0) * K_zX.mean(axis=1) - (K_zX @ self.G @ K_XX).mean(axis=1) / M

        return (part1 + part2).squeeze() * (1 + self.lmbda)
    
    def get_first_variation(self, Y) -> Callable:
        return partial(self.witness_function, Y=Y)

    def __call__(self, Y) -> Scalar:
        return 0
    


class ula(struct.PyTreeNode):
    kernel: base_kernel
    lmbda: float
    X: Array # Target samples
    target_dist: Distribution

    def witness_function(
        self, z
        # In ULA, Y is not needed.
    ) -> Scalar:
        # log_p = self.X.shape[0] * jnp.log(self.std) + jnp.sum(-0.5 * (z - self.mu) ** 2 / self.std ** 2, axis=1)
        log_p = self.target_dist.log_prob(z).sum()
        return -log_p # Energy is negative log density
    
    def get_first_variation(self, Y) -> Callable:
        return self.witness_function
