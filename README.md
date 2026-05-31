# study_bayes

ベイズ推論関係の勉強  

## ガウス過程と機械学習

参考書: [ガウス過程と機械学習 | 書籍情報 | 株式会社 講談社サイエンティフィク](https://www.kspub.co.jp/book/detail/1529267.html)

### RBFカーネル+観測ノイズ

```math
k(\mathbf{x}, \mathbf{x}')
= \theta_1 \exp\left(
    -\frac{\lVert \mathbf{x}-\mathbf{x}' \rVert^2}{2\theta_2^2}
  \right)
+ \theta_3 \delta(\mathbf{x}, \mathbf{x}')
```

![RBFカーネル+ガウスノイズにおけるハイパーパラメータ比較](ガウス過程と機械学習/gp_hyperparameter_grid.png)

## ベイズ推論による機械学習入門

参考書: [ベイズ推論による機械学習入門 | 書籍情報 | 株式会社 講談社サイエンティフィク](https://www.kspub.co.jp/book/detail/1538320.html)

### 基本的な確率分布

#### 離散確率分布

<details>
<summary>ベルヌーイ分布</summary>

経験分布と理論 PMF  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/bernoulli/empirical_vs_theoretical.png" width="320">

`p` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/bernoulli/parameter_sweep.gif" width="320">

収束確認  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/bernoulli/convergence.png" width="320">

</details>

<details>
<summary>二項分布</summary>

理論 PMF  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/binomial/pmf.png" width="320">

`p` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/binomial/parameter_sweep.gif" width="320">

収束確認  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/binomial/convergence.png" width="320">

</details>

<details>
<summary>カテゴリ分布</summary>

経験分布と理論 PMF  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/categorical/empirical_vs_theoretical.png" width="320">

`p0` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/categorical/parameter_sweep.gif" width="320">

収束確認  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/categorical/convergence.png" width="320">

</details>

<details>
<summary>多項分布</summary>

Plotly 3D 静止図  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/multinomial_plotly/surface.png" width="320">

Plotly 3D アニメーションの静止画像  
[interactive HTML](ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/multinomial_plotly/surface_animation.html)  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/multinomial_plotly/surface_animation_preview.png" width="320">

収束確認  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/multinomial_plotly/convergence.png" width="320">

</details>

<details>
<summary>ポアソン分布</summary>

経験分布と理論 PMF  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/poisson/empirical_vs_theoretical.png" width="320">

`λ` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/poisson/parameter_sweep.gif" width="320">

収束確認  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/poisson/convergence.png" width="320">

</details>

#### 連続確率分布

<details>
<summary>ベータ分布</summary>

経験分布と理論 PDF  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/beta/empirical_vs_theoretical.png" width="320">

`α` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/beta/alpha_sweep.gif" width="320">

`β` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/beta/beta_sweep.gif" width="320">

`α + β` 一定  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/beta/fixed_total.gif" width="320">

`α / β` 一定  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/beta/fixed_ratio.gif" width="320">

収束確認  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/beta/convergence.png" width="320">

</details>

<details>
<summary>ディリクレ分布</summary>

静止図  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/dirichlet/static_example.png" width="320">

`α1` 掃引  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/dirichlet/alpha1_sweep.gif" width="320">

支配頂点の切り替え  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/dirichlet/dominant_vertex_cycle.gif" width="320">

対称 `α` の変化  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/dirichlet/symmetric_alpha_sweep.gif" width="320">

</details>

<details>
<summary>ガンマ分布</summary>

形状母数 `k` の比較  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gamma/shape_by_k.png" width="320">

尺度母数 `θ` の比較  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gamma/shape_by_theta.png" width="320">

サンプリング収束  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gamma/sampling_convergence.gif" width="320">

`k` を小さい範囲で変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gamma/k_small_sweep.gif" width="320">

`k` を大きい範囲で変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gamma/k_large_sweep.gif" width="320">

`θ` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gamma/theta_sweep.gif" width="320">

平均一定  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gamma/constant_mean.gif" width="320">

`k, θ` 同時増加  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gamma/joint_growth.gif" width="320">

</details>

<details>
<summary>1次元ガウス分布</summary>

形状比較  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_1d/shape_comparison.png" width="320">

サンプリング収束  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_1d/sampling_convergence.gif" width="320">

平均を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_1d/mean_sweep.gif" width="320">

標準偏差を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_1d/sigma_sweep.gif" width="320">

平均と標準偏差の同時掃引  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_1d/mean_sigma_sweep.gif" width="320">

</details>

<details>
<summary>多次元ガウス分布</summary>

2D 等高線比較  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_2d/contour_comparison.png" width="320">

2D サンプリング収束  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_2d/sampling_convergence.gif" width="320">

3D Plotly 円運動の静止画像  
[interactive HTML](ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_3d_plotly/circular_motion.html)  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_3d_plotly/circular_motion_preview.png" width="320">

3D Plotly 呼吸変形の静止画像  
[interactive HTML](ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_3d_plotly/breathing_variance.html)  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_3d_plotly/breathing_variance_preview.png" width="320">

3D Plotly 回転の静止画像  
[interactive HTML](ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_3d_plotly/rotation_animation.html)  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_3d_plotly/rotation_animation_preview.png" width="320">

3D Plotly 複合変化の静止画像  
[interactive HTML](ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_3d_plotly/combined_motion.html)  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/gaussian_3d_plotly/combined_motion_preview.png" width="320">

</details>

<details>
<summary>ウィシャート分布</summary>

パラメータ比較  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/wishart/parameter_comparison.png" width="320">

自由度 `ν` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/wishart/nu_sweep.gif" width="320">

スケール行列 `W` を変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/wishart/scale_matrix_sweep.gif" width="320">

`ν` と `W` を同時に変えたときのアニメーション  
<img src="ベイズ推論による機械学習入門/Chapter2-基本的な確率分布/wishart/combined_sweep.gif" width="320">

</details>

### 推論

#### 1次元ガウス分布の学習と予測

![1次元ガウス分布の学習と予測](ベイズ推論による機械学習入門/Chapter3-ベイズ推論による学習と予測/1_dimensional_gaussian_animation.gif)

#### 多次元ガウス分布の学習と予測

![多次元ガウス分布の学習と予測](ベイズ推論による機械学習入門/Chapter3-ベイズ推論による学習と予測/multi_dimensional_gaussian_animation.gif)

#### 線形回帰

![線形回帰](ベイズ推論による機械学習入門/Chapter3-ベイズ推論による学習と予測/linear_regression_animation.gif)

---

#### ポアソン混合モデル

##### ギブスサンプリング

![ポアソン混合モデルに対するギブスサンプリング](ベイズ推論による機械学習入門/Chapter4-混合モデルと近似推論/mixture_poisson_gibbs_sampling_animation.gif)

##### 変分推論

![ポアソン混合モデルに対する変分推論](ベイズ推論による機械学習入門/Chapter4-混合モデルと近似推論/mixture_poisson_variational_inference_animation.gif)

##### 崩壊型ギブスサンプリング

![ポアソン混合モデルに対する崩壊型ギブスサンプリング](ベイズ推論による機械学習入門/Chapter4-混合モデルと近似推論/mixture_poisson_collapsed_gibbs_sampling_animation.gif)

---

#### ガウス混合モデル

##### ギブスサンプリング

![ガウス混合モデルに対するギブスサンプリング](ベイズ推論による機械学習入門/Chapter4-混合モデルと近似推論/mixture_gaussian_gibbs_sampling_animation.gif)

##### 変分推論

![ガウス混合モデルに対する変分推論](ベイズ推論による機械学習入門/Chapter4-混合モデルと近似推論/mixture_gaussian_variational_inference_animation.gif)

##### 崩壊型ギブスサンプリング

![ガウス混合モデルに対する崩壊型ギブスサンプリング](ベイズ推論による機械学習入門/Chapter4-混合モデルと近似推論/mixture_gaussian_collapsed_gibbs_sampling_animation.gif)
