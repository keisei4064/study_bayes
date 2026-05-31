from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax import random


def true_function(x: Array) -> Array:
    """真の関数（推定対象）"""
    return jnp.sin(x) + 0.3 * jnp.cos(3.0 * x)


def make_dataset(
    key: Array,
    n_train: int = 12,
    noise_std: float = 0.15,
) -> tuple[Array, Array, Array, Array]:
    """データセット作成"""
    key_x, key_noise = random.split(key)

    x_train: Array = random.uniform(
        key_x,
        shape=(n_train,),
        minval=-5.0,
        maxval=5.0,
    )
    x_train = jnp.sort(x_train)

    y_clean: Array = true_function(x_train)
    # 観測ノイズを加える
    noise: Array = noise_std * random.normal(key_noise, shape=(n_train,))
    y_train: Array = y_clean + noise

    x_test: Array = jnp.linspace(-6.0, 6.0, 400)
    y_true: Array = true_function(x_test)

    return x_train, y_train, x_test, y_true


if __name__ == "__main__":
    key = random.PRNGKey(0)
    x_train, y_train, x_test, y_true = make_dataset(key)
    # print(x_train)
    # print(y_train)
    # print(x_test)
    # print(y_true)

    import matplotlib.pyplot as plt

    plt.scatter(x_train, y_train, label="train")
    plt.plot(x_test, y_true, label="true")
    plt.grid()
    plt.legend()
    plt.show()
