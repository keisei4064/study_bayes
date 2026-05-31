from __future__ import annotations

import gp_ml.data_sample
import gp_ml.gp

import numpy as np
import matplotlib.pyplot as plt
from jax import Array
from jax import random
import jax.numpy as jnp
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_gp_regression(
    x_train: Array,
    y_train: Array,
    x_test: Array,
    y_true: Array,
    mu: Array,
    var: Array,
    title: str = "Gaussian process regression",
) -> tuple[Figure, Axes]:
    """GP 回帰結果を可視化する。

    Args:
        x_train: 訓練入力。shape は (N,)。
        y_train: 訓練出力。shape は (N,)。
        x_test: テスト入力。shape は (M,)。
        y_true: 真の関数値。shape は (M,)。
        mu: GP 予測平均。shape は (M,)。
        var: GP 予測分散。shape は (M,)。
        title: 図のタイトル。

    Returns:
        matplotlib の Figure と Axes。
    """

    x_train_np: np.ndarray = np.asarray(x_train)
    y_train_np: np.ndarray = np.asarray(y_train)
    x_test_np: np.ndarray = np.asarray(x_test)
    y_true_np: np.ndarray = np.asarray(y_true)
    mu_np: np.ndarray = np.asarray(mu)

    # 分散を標準偏差へ
    std_np: np.ndarray = np.sqrt(np.maximum(np.asarray(var), 0.0))

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(x_test_np, y_true_np, label="true function")
    ax.scatter(x_train_np, y_train_np, label="observations", zorder=3, color="green")
    ax.plot(x_test_np, mu_np, label="GP mean", color="orange")

    # 2σ 範囲
    ax.fill_between(
        x_test_np,
        mu_np - 2.0 * std_np,
        mu_np + 2.0 * std_np,
        alpha=0.2,
        label=r"GP mean $\pm 2\sigma$",
        color="orange",
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    return fig, ax


def main() -> None:
    key: Array = random.PRNGKey(0)

    x_train, y_train, x_test, y_true = gp_ml.data_sample.make_dataset(
        key,
        n_train=12,
        noise_std=0.15,
    )

    X_train: Array = x_train[:, None]
    X_test: Array = x_test[:, None]

    # ハイパーパラメータ
    theta1: Array = jnp.array(1.0)
    theta2: Array = jnp.array(1.0)
    theta3: Array = jnp.array(0.15**2)

    mu, var = gp_ml.gp.train_and_predict_gp(
        X_sample=X_train,
        Y_sample=y_train,
        theta1=theta1,
        theta2=theta2,
        theta3=theta3,
        X_test=X_test,
    )

    fig, ax = plot_gp_regression(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_true=y_true,
        mu=mu,
        var=var,
        title="GP regression with fixed hyperparameters",
    )

    plt.show()


if __name__ == "__main__":
    main()
