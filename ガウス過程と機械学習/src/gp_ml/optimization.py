from __future__ import annotations

import numpy as np
import scipy.optimize as opt
import jax
import jax.numpy as jnp
from jax import Array
from jax.scipy.linalg import solve_triangular

import gp_ml.gp


def unpack_log_params(log_params: Array) -> tuple[Array, Array, Array]:
    """最適化で用いる log パラメータを元のハイパーパラメータ形式へ変換する"""
    theta1: Array = jnp.exp(log_params[0])
    theta2: Array = jnp.exp(log_params[1])
    theta3: Array = jnp.exp(log_params[2])
    return theta1, theta2, theta3


def negative_log_marginal_likelihood(
    log_params: Array,
    X_sample: Array,
    Y_sample: Array,
) -> Array:
    """GP 回帰の負の対数周辺尤度を計算する.

    Args:
        log_params: log(theta1), log(theta2), log(theta3) を並べた配列.
        X_sample: 観測入力. shape は (N, D).
        Y_sample: 観測出力. shape は (N,).

    Returns:
        負の対数周辺尤度.
    """
    theta1, theta2, theta3 = unpack_log_params(log_params)

    K: Array = gp_ml.gp.rbf_with_noise_kernel_matrix(
        X_sample,
        theta1,
        theta2,
        theta3,
    )

    # コレスキー分解
    L: Array = jnp.linalg.cholesky(K)

    # alpha = K^{-1} y を Cholesky 経由で計算する.
    v: Array = solve_triangular(L, Y_sample, lower=True)
    alpha: Array = solve_triangular(L.T, v, lower=False)

    # 式(3.91) を計算
    data_fit: Array = 0.5 * (Y_sample @ alpha)
    complexity_penalty: Array = jnp.sum(jnp.log(jnp.diag(L)))
    normalization: Array = 0.5 * Y_sample.shape[0] * jnp.log(2.0 * jnp.pi)

    return data_fit + complexity_penalty + normalization


def optimize_hyperparameters(
    X_sample: Array,
    Y_sample: Array,
    initial_log_params: Array,
) -> tuple[Array, Array, opt.OptimizeResult]:
    """対数周辺尤度最大化により GP のハイパーパラメータを推定する.

    Args:
        X_sample: 観測入力. shape は (N, D).
        Y_sample: 観測出力. shape は (N,).
        initial_log_params: 最適化の初期値.

    Returns:
        最適化後の log パラメータ，最適化後の負の対数周辺尤度，SciPy の結果.
    """

    # jax で勾配計算
    #   [jax.value_and_grad — JAX documentation](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html#jax.value_and_grad)
    value_and_grad = jax.value_and_grad(negative_log_marginal_likelihood)

    def objective(log_params_np: np.ndarray) -> tuple[float, np.ndarray]:
        log_params: Array = jnp.asarray(log_params_np)
        value, grad = value_and_grad(log_params, X_sample, Y_sample)

        value_np: float = float(value)
        grad_np: np.ndarray = np.asarray(grad, dtype=np.float64)

        # (objective function, gradient) を返す
        return value_np, grad_np

    # Scypy の‘L-BFGS-B’ を使って最適化
    #   [minimize — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize)
    #   [minimize(method=’L-BFGS-B’) — SciPy v1.17.0 Manual](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb)
    result: opt.OptimizeResult = opt.minimize(
        objective,
        x0=np.asarray(initial_log_params, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
    )

    optimized_log_params: Array = jnp.asarray(result.x)
    optimized_nll: Array = jnp.asarray(result.fun)

    return optimized_log_params, optimized_nll, result
