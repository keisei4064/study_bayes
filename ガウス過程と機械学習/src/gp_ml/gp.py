import jax.numpy as jnp
from jax import Array


def rbf_kernel_matrix(
    X1: Array,
    X2: Array,
    theta1: Array,
    theta2: Array,
) -> Array:
    """RBF カーネル行列を計算する.

    Args:
        X1: 入力点集合. shape は (N, D).
        X2: 入力点集合. shape は (M, D).
        theta1: カーネルのスケールパラメータ.
        theta2: カーネルの長さパラメータ.

    Returns:
        RBF カーネル行列. shape は (N, M).
    """
    norm_sq_1: Array = jnp.sum(X1**2, axis=1, keepdims=True)
    norm_sq_2: Array = jnp.sum(X2**2, axis=1, keepdims=True).T

    # broadcasting により (N, 1) + (1, M) -> (N, M)
    dist_sq: Array = norm_sq_1 + norm_sq_2 - 2.0 * (X1 @ X2.T)

    # 数値誤差で生まれ得る微小な負値への対策
    dist_sq = jnp.maximum(dist_sq, 0.0)

    return theta1 * jnp.exp(-dist_sq / theta2)


def rbf_with_noise_kernel_matrix(
    X: Array,
    theta1: Array,
    theta2: Array,
    theta3: Array,
    jitter: float = 1e-8,
) -> Array:
    """観測ノイズ付き RBF カーネル行列を計算する.

    Args:
        X: 観測入力. shape は (N, D).
        theta1: カーネルのスケールパラメータ.
        theta2: カーネルの長さパラメータ.
        theta3: 観測ノイズ分散.
        jitter: 数値安定化用の微小値.

    Returns:
        観測ノイズ付きカーネル行列. shape は (N, N).
    """
    kernel: Array = rbf_kernel_matrix(X, X, theta1, theta2)

    n: int = X.shape[0]
    eye: Array = jnp.eye(n, dtype=X.dtype)

    return kernel + (theta3 + jitter) * eye


def train_and_predict_gp(
    X_sample: Array,
    Y_sample: Array,
    theta1: Array,
    theta2: Array,
    theta3: Array,
    X_test: Array,
) -> tuple[Array, Array]:
    """GP 回帰によりテスト点での予測平均と予測分散を計算する.

    Args:
        X_sample: 観測入力. shape は (N, D).
        Y_sample: 観測出力. shape は (N,).
        theta1: カーネルのスケールパラメータ.
        theta2: カーネルの長さパラメータ.
        theta3: 観測ノイズ分散.
        X_test: テスト入力. shape は (M, D).

    Returns:
        予測平均と予測分散.
        予測平均の shape は (M,).
        予測分散の shape は (M,).
    """
    K: Array = rbf_with_noise_kernel_matrix(
        X_sample,
        theta1,
        theta2,
        theta3,
    )

    K_star: Array = rbf_kernel_matrix(
        X_sample,
        X_test,
        theta1,
        theta2,
    )

    K_star_star: Array = rbf_kernel_matrix(
        X_test,
        X_test,
        theta1,
        theta2,
    )

    alpha: Array = jnp.linalg.solve(K, Y_sample)

    # 予測平均
    mu: Array = K_star.T @ alpha

    # 予測分散
    v: Array = jnp.linalg.solve(K, K_star)
    cov: Array = K_star_star - K_star.T @ v

    # 対角だけ見れば，各点別々に予測した結果と同じ
    var: Array = jnp.diag(cov)
    var = jnp.maximum(var, 0.0)

    return mu, var


if __name__ == "__main__":
    K = rbf_with_noise_kernel_matrix(
        jnp.array([[1.0, -2.0, 5.0]]).T,
        jnp.array([1.0]),
        jnp.array([2.0]),
        jnp.array([0.1]),
    )
    print(K)
