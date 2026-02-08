# SVD

[低秩近似之路（二）：SVD](https://www.spaces.ac.cn/archives/10407)

[SVD分解(一)：自编码器与人工智能](https://www.spaces.ac.cn/archives/4208) 

[白板机器学习-降维](https://www.bilibili.com/video/BV1aE411o7qd)

## 谱定理

对于任意实对称矩阵$\boldsymbol M \in \mathbb{R}^{n \times n}$都存在谱分解（也称特征值分解）

$$
\boldsymbol{M} = \boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^\top
$$

其中$\boldsymbol U,\boldsymbol{\Lambda} \in \mathbb{R}^{n \times n}$，并且$\boldsymbol{U}$是正交矩阵$\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \cdots, \lambda_n)$是对角矩阵

谱定理的几何特性：实对称矩阵代表的线性变换会将单位圆缩放为椭圆，椭圆的缩放方向为特征向量，在 [zhihu](https://zhuanlan.zhihu.com/p/1914024366347388318) 中有二维的可视化结果。这说明实对称矩阵的变换没有旋转（rotation），也没有剪切（shear），只有单纯的缩放，因为任何的旋转和剪切效应都会破坏实对称性质

证明谱定理最简单的方式是数学归纳法。假设对于$n-1$维的实对称矩阵能够被对角化，证明$n$维的实对称矩阵能够被对角化。对于一维的矩阵，显然成立，第一块积木推倒。现在对于实对称矩阵$\boldsymbol M \in \mathbb{R}^{n \times n}$，设$\lambda_1$是其一个非零特征值（可以证明实对称矩阵至少有一个特征值），对应的特征向量为$\boldsymbol{u}_1$。将$\boldsymbol{u}_1$扩展成为一组正交基，其余的基向量为$\boldsymbol{Q} = (\boldsymbol{q}_2, \ldots, \boldsymbol{q}_n)$

现在将矩阵$(\boldsymbol u_1, \boldsymbol Q)^\top \boldsymbol M (\boldsymbol u_1,\boldsymbol Q)$表示为分块矩阵：

$$
(\boldsymbol u_1, \boldsymbol Q)^\top \boldsymbol M (\boldsymbol u_1,\boldsymbol Q) = \begin{pmatrix}
        \lambda_1 & \boldsymbol{0} \\
        \boldsymbol{0} & \boldsymbol{B}
        \end{pmatrix}
$$

其中$\boldsymbol B=\boldsymbol Q^\top \boldsymbol M \boldsymbol Q$，是一个$n-1$维的实对称矩阵。根据数学归纳假设，该矩阵可以被对角化，所以将其表示为$\boldsymbol{B} = \boldsymbol{V} \boldsymbol{\Lambda}_1 \boldsymbol{V}^\top$，其中$\boldsymbol V$是$n-1$维的正交矩阵，通过将$\boldsymbol B$展开，然后把$\boldsymbol V$移到左侧可得到

$$
(\boldsymbol Q \boldsymbol V)^\top \boldsymbol M (\boldsymbol Q \boldsymbol V) = \boldsymbol{\Lambda}_1
$$

于是乎，我们可以得到

$$
(\boldsymbol u_1, \boldsymbol Q \boldsymbol V)^\top \boldsymbol M (\boldsymbol u_1,\boldsymbol Q \boldsymbol V) = \begin{pmatrix}
        \lambda_1 & \boldsymbol{0} \\
        \boldsymbol{0} & \boldsymbol{\Lambda_1}
        \end{pmatrix}
$$

此时我们就证明了 n 维实对称矩阵也是可以被对角化的

## SVD 的存在性证明

有了谱定理过后能够比较轻松地证明 SVD 存在性。为了更符合机器学习的习惯，我这里用$\boldsymbol M \in \mathbb{R}^{n \times k}$来表示，其中可以理解为$n$数据的数量，而$k$则表示数据的维度，这个 notation 在之后讨论 PCA 的时候也会使用。现在我们先拿出 SVD 的结论：

对于任意矩阵$\boldsymbol M \in \mathbb{R}^{n \times k}$，都可以找到如下形式的奇异值分解（SVD，Singular Value Decomposition）

$$
\boldsymbol M=\boldsymbol U \boldsymbol \Sigma \boldsymbol V^\top
$$

其中$\boldsymbol U \in \mathbb R^{n\times n}, \boldsymbol V\in \mathbb R^{k\times k}$都是正交矩阵，$\boldsymbol\Sigma \in \mathbb R^{n\times k}$是非负对角阵

$$
\boldsymbol{\Sigma}_{i, j}=\left\{\begin{array}{ll}
\sigma_{i}, & i=j \\
0, & i \neq j
\end{array}\right.
$$

并且对角线元素从大到小排布：$\sigma_1\ge \sigma_2 \ge \sigma_3\ge...\ge0$，这些对角线元素就称为奇异值

在证明 SVD 的存在性之前，可以直观推倒：如果矩阵是一个实对称矩阵，那么 SVD 结论就退化成为谱定理，即：谱定理是 SVD 中的特例情况。但事实还要更有趣一点，谱定理和 SVD 可以有更复杂的互动：我们可以简单地构造实对称矩阵$\boldsymbol M^\top \boldsymbol M$或者$\boldsymbol M \boldsymbol M^\top$，这样的矩阵就符合谱定理。而如果我们将 SVD 的结论带入到上面构造的对称矩阵当中

$$
\boldsymbol{M}\boldsymbol{M}^{\top}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}\boldsymbol{V}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}=\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}\\
\boldsymbol{M}^{\top}\boldsymbol{M}=\boldsymbol{V}\boldsymbol{\Sigma}^{\top}\boldsymbol{U}^{\top}\boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}=\boldsymbol{V}\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma}\boldsymbol{V}^{\top}
$$

有趣的事情发生了，我们直接得到了$\boldsymbol{M}\boldsymbol{M}^{\top}$和$\boldsymbol{M}^{\top}\boldsymbol{M}$的谱分解结果！换句话说：SVD 分解中的$\boldsymbol U, \boldsymbol V, \boldsymbol \Sigma$其实就是谱分解中对应的特征向量、特征值的平方根

其实以上互动也给我们证明 SVD 的存在性提供了很好的方向：从某一个实对称矩阵的谱分解出发（e.g.$\boldsymbol{M}^{\top}\boldsymbol{M}$），获得其特征向量（e.g.$\boldsymbol V$），利用 SVD 的形式直接构造出另外一组向量$\boldsymbol U$，我们只需要证明这一组向量是相互正交的，即可证明 SVD 的存在性

我们设$\boldsymbol{M}^{\top}\boldsymbol{M}$的谱分解为$\boldsymbol{M}^{\top}\boldsymbol{M} =\boldsymbol V \boldsymbol \Lambda\boldsymbol V^\top$，并且不失一般性设$\boldsymbol M$的 rank 为$r$，$\boldsymbol \Lambda$的特征值排列为降序排列，并且由于$\boldsymbol{M}^{\top}\boldsymbol{M}$是半正定矩阵，其特征值都为非负数

OK，一切准备就绪，开始构造$\boldsymbol U$

$$
\boldsymbol{\Sigma}_{[:r,:r]}=(\boldsymbol{\Lambda}_{[:r,:r]})^{1/2},\quad \boldsymbol U_{[:n,:r]}=\boldsymbol M\boldsymbol V_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1}
$$

由于 rank 的限制，只构造了$r$个向量。现在我们来证明这 r 个向量是相互正交的

$$
\begin{aligned}
\boldsymbol{U}_{[:n,:r]}^{\top}\boldsymbol{U}_{[:n,:r]} & =\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{V}_{[:k,:r]}^{\top}\boldsymbol{M}^{\top}\boldsymbol{M}\boldsymbol{V}_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1} \\
& =\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{V}_{[:k,:r]}^{\top}\boldsymbol{V}\boldsymbol{\Lambda}\boldsymbol{V}^{\top}\boldsymbol{V}_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1} \\
& =\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{I}_{[:r,:k]}\boldsymbol{\Lambda}\boldsymbol{I}_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1} \\
& =\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{\Lambda}_{[:r,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1} \\
& =\boldsymbol{I}_{r}
\end{aligned}
$$

可以看到，基于谱分解的结论，很快就消除了各个部分，得到了正交结论。此时可以直接把$\boldsymbol U_{[:n,:r]}$扩充成为完整的正交基即可得到完成的$\boldsymbol U$矩阵。但是我们其实还没有直接证明 SVD 的存在，因为不管是我们构造的$\boldsymbol U_{[:n,:r]}$还是扩充的$\boldsymbol U$都没有直接地以 SVD 的形式给出，$\boldsymbol M$始终在等式的右侧并且与其他矩阵相乘。不过我们现在手上的材料足够多，要直接证明 SVD 的形式已经不难了

我们首先从未扩充版本的 SVD 形式开始推，直接带入我们之前的$\boldsymbol U_{[:n,:r]}$结论

$$
\boldsymbol{U}_{[:n,:r]}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:k,:r]}^{\top}=\boldsymbol{M}\boldsymbol{V}_{[:k,:r]}\boldsymbol{\Sigma}_{[:r,:r]}^{-1}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:k,:r]}^{\top}=\boldsymbol{M}\boldsymbol{V}_{[:k,:r]}\boldsymbol{V}_{[:k,:r]}^{\top}
$$

现在只需要证明$\boldsymbol{M}\boldsymbol{V}_{[:k,:r]}\boldsymbol{V}_{[:k,:r]}^{\top} = \boldsymbol M$然后扩充我们的结论为完整的 SVD 即可

$$
\begin{aligned}
\boldsymbol{M} & = \boldsymbol{M} \boldsymbol{V} \boldsymbol{V}^{\top} \\
    & = \begin{pmatrix} \boldsymbol{M} \boldsymbol{V}_{[:k,:r]} & \boldsymbol{M} \boldsymbol{V}_{[:k, r:]} \end{pmatrix} \begin{pmatrix} \boldsymbol{V}_{[:k,:r]}^{\top} \\ \boldsymbol{V}_{[:k, r:]}^{\top} \end{pmatrix} \\
    & = \begin{pmatrix} \boldsymbol{M} \boldsymbol{V}_{[:k,:r]} & \boldsymbol{0}_{k \times (k-r)} \end{pmatrix} \begin{pmatrix} \boldsymbol{V}_{[:k,:r]}^{\top} \\ \boldsymbol{V}_{[:k, r:]}^{\top} \end{pmatrix} \\
    & = \boldsymbol{M} \boldsymbol{V}_{[:k,:r]} \boldsymbol{V}_{[:k,:r]}^{\top}
\end{aligned}
$$

上述证明的关键就是在第三个等式，利用了特征值为0的特征向量的性质：$\boldsymbol M \boldsymbol v_i=0$

最后将$\boldsymbol{U}_{[:n,:r]}\boldsymbol{\Sigma}_{[:r,:r]}\boldsymbol{V}_{[:k,:r]}^{\top}=\boldsymbol M$进行扩充，即可得到完整的 SVD 形式

$$
\boldsymbol M=\boldsymbol U \boldsymbol \Sigma \boldsymbol V^\top
$$

## 降维

废了不少劲证明了 SVD 的存在性，但是还是没体会到其妙处。现在我们通过机器学习中的降维为例子，来一探 SVD 的妙处。

参考 [zhihu-白板机器学习-降维](https://zhuanlan.zhihu.com/p/326074168)

首先我们可能希望了解：我们为什么需要降维？我们从一个高维的反直觉思维开始：在高维当中单位球体的体积趋近于 0，而单位立方体的体积永远都是1，高维球体体积的计算公式如下：

$$
V_n = \frac{\pi^{n/2}}{\Gamma(n/2+1)}
$$

分母伽马函数的增长速度远超分子，所以随着维度的增加，体积迅速减少。这体现了高维空间的数据稀疏性。稀疏性的本质来源于维度灾难，这里有另一个直观的解释

> From DeepSeek
>
> 1. **一维空间（一条线）**：
>
>    假设我们有一条长度为1米的线段。我们在这条线上均匀地撒上100个点。这些点会非常密集，相邻点之间的距离大约是1厘米。空间感觉很“满”。
>
> 2. **二维空间（一个正方形）**：
>
>    现在，我们扩展到一个1米 x 1米的正方形平面。为了保持和线上同样的“密度”（即单位面积内的点数），我们需要多少点？我们需要 10,000 个点！因为现在点不仅要覆盖长度，还要覆盖宽度。如果还是只有100个点，它们在这个平面上就会显得非常稀疏、孤零零的。
>
> 3. **三维空间（一个立方体）**：
>
>    再进一步，到一个1米 x 1米 x 1米的立方体。要保持同样的密度，我们需 1,000,000 个点！如果还是只有100个点，那么这些点就像宇宙中寥寥无几的星星，彼此之间的距离非常遥远。

球壳上的体积变得异常的大，你可以想象将多维球壳向外扩展 10%，将会带来体积的爆炸性增长，这就是因为每一个维度都要向前扩展 10% 形成了维度灾难，此时增加的球壳体积会远远超过球体的体积

维度灾难带来的部分挑战：

1. 距离度量失效：高维空间中的各个点的距离都差不多（e.g. 数据都集中在球壳上，距离原点的距离都是一样的），导致了以距离为基础的机器学习方法直接失效（KNN、聚类）
2. 采样困难：在高维空间中的采样随着维度增加指数上升

所以降维能够带来计算和度量上的便利，并且我们常常希望找到事物中的核心影响因素，保留最重要的维度，所以降维的好处是不言而喻的。不过我觉得仍要从两面性来看待维度灾难：

1. 希望简化计算复杂度时，使用降维能够极其有效地带来收益
2. 希望增加模型能力/容量时，利用维度的上升也可带来模型容量的快速提升

### 最大投影方差

最大投影方差的思路是：寻找一个方向，数据点在该方向上的投影方差最大（i.e. 该方向是最能区分数据之间差别的方向）。接下来做一些 Notation：

1. 数据样本$X \in \mathbb R^{n\times k}$，数据样本个数为$n$，维度为$k$

2. 中心矩阵$H \in \mathbb R^{n\times n}$，其作用是将数据进行中心化，等价于 `X - torch.mean(X, dim=0, keep_dim=True)`
    
    $$
    H = I_n - \frac{1}{n}1_n1_n^\top
    $$
    
    其中$1_n$为一个全 1 的向量$1_n \in \mathbb R^{n\times1}$。中心化过后的数据样本即为$HX = X-\overline X$。中心矩阵有两个常用的性质$H^T=H$，$H^2=H$，其中第二个性质也比较好理解：对中心化过后的数据再做中心化是不变的

3. 数据的方差矩阵$S \in \mathbb R^{k\times k}$表示
    
    $$
    S=(X-\overline X)^\top(X-\overline X)=(HX)^\top(HX)=X^\top H X
    $$
    
    真正的方差就是求对$S$求和，然后除以样本个数

4. 

接下来可以直接根据思路写出优化目标：投影的方差最小。我们先写出投影的方差矩阵

$$
L=(Xu)^TH(Xu)=u^\top X^\top H Xu=u^\top Su
$$

接着可以写出优化目标，找到最优的方向$u$使得方差最大

$$
L=\operatorname*{argmax}_{u} u^{T}Su \\ \text{s.t.} \ u^{T}u=1
$$

解这个方法直接用 lagrange 乘子法

$$
Lagrange(u, \lambda) = u^{T} S u + \lambda(1 - u^{T} u) \\
\frac{\partial L}{\partial u} = 2 S u - 2 \lambda u = 0 \\
S u = \lambda u
$$

**最终发现，我们要找到的$u$就是矩阵$S$的特征向量，$\lambda$就是对应的特征值！此时最优的$L=\lambda$，那么我们就选择特征值最大的特征向量即可**。此时我们可以根据特征值的大小，选择最重要的 topk 个特征向量作为投影方向

### 最小重构代价

我认为最小重构代价更为直观一点，更能贴合我们降维的目的：维度降低，但是信息量减少最少。在 [zhihu-白板机器学习-降维](https://zhuanlan.zhihu.com/p/326074168) 中使用了构造一个正交基的方法来证明最小重构代价，最后只需要放弃特征值最小的特征向量即可。这里我整理我自己的思路，最终推导出和最大投影方差相同的结论

我同样聚焦于一个向量：把（已中心化）数据投影到向量上$u$，计算得到投影过后的新数据坐标，要求新数据与原始数据的 F-范数最小

$$
L=\operatorname*{argmin}_u ||X-(Xu)u^\top||_F^2
$$

接下来的证明求助了 DeepSeek，我自己是没证出来的，其中运用了 trace 的性质

> From DeepSeek
>
> 由于Frobenius范数的平方等于矩阵迹的形式，有：
> 
> $$
> J(\mathbf{u}) = \operatorname{tr} \left( (\mathbf{X} - \mathbf{X} \mathbf{u} \mathbf{u}^T)^T (\mathbf{X} - \mathbf{X} \mathbf{u} \mathbf{u}^T) \right)
> $$
> 
> 展开括号内的表达式
> 
> $$
> (\mathbf{X} - \mathbf{X} \mathbf{u} \mathbf{u}^T)^T (\mathbf{X} - \mathbf{X} \mathbf{u} \mathbf{u}^T) = \mathbf{X}^T \mathbf{X} - \mathbf{X}^T \mathbf{X} \mathbf{u} \mathbf{u}^T - \mathbf{u} \mathbf{u}^T \mathbf{X}^T \mathbf{X} + \mathbf{u} \mathbf{u}^T \mathbf{X}^T \mathbf{X} \mathbf{u} \mathbf{u}^T
> $$
> 
> 令$\mathbf{S} = \mathbf{X}^T \mathbf{X}$，带入上式得到
> 
> $$
> J(\mathbf{u}) = \operatorname{tr} \left( \mathbf{S} - \mathbf{S} \mathbf{u} \mathbf{u}^T - \mathbf{u} \mathbf{u}^T \mathbf{S} + \mathbf{u} \mathbf{u}^T \mathbf{S} \mathbf{u} \mathbf{u}^T \right)
> $$
> 
> 由 trace 性质可以将其展开 [wiki-trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) 为四项
>
> -$\operatorname{tr}(\mathbf{S})$是常数，与$\mathbf{u}$无关。
> -$\operatorname{tr}(\mathbf{S} \mathbf{u} \mathbf{u}^T) = \operatorname{tr}(\mathbf{u}^T \mathbf{S} \mathbf{u}) = \mathbf{u}^T \mathbf{S} \mathbf{u}$（因为$\mathbf{u}^T \mathbf{S} \mathbf{u}$是标量）。参考 [wiki-trace](https://en.wikipedia.org/wiki/Trace_(linear_algebra)) 有性质：$\operatorname tr(AB)=\operatorname tr(BA)$
> -$\operatorname{tr}(\mathbf{u} \mathbf{u}^T \mathbf{S}) = \operatorname{tr}(\mathbf{u}^T \mathbf{S} \mathbf{u}) = \mathbf{u}^T \mathbf{S} \mathbf{u}$。
> -$\operatorname{tr}(\mathbf{u} \mathbf{u}^T \mathbf{S} \mathbf{u} \mathbf{u}^T)$：注意$\mathbf{u}^T \mathbf{S} \mathbf{u}$是标量，记为$c$，则$\mathbf{u} \mathbf{u}^T \mathbf{S} \mathbf{u} \mathbf{u}^T = c \mathbf{u} \mathbf{u}^T$，所以$\operatorname{tr}(c \mathbf{u} \mathbf{u}^T) = c \operatorname{tr}(\mathbf{u} \mathbf{u}^T) = c \mathbf{u}^T \mathbf{u} = c$（因为$\mathbf{u}^T \mathbf{u} = 1$）。因此，这一项也是$\mathbf{u}^T \mathbf{S} \mathbf{u}$.
>
> 所以整个优化方程化简为：
> 
> $$
> J(\mathbf{u}) = \operatorname{tr}(\mathbf{S}) - \mathbf{u}^T \mathbf{S} \mathbf{u} - \mathbf{u}^T \mathbf{S} \mathbf{u} + \mathbf{u}^T \mathbf{S} \mathbf{u} = \operatorname{tr}(\mathbf{S}) - \mathbf{u}^T \mathbf{S} \mathbf{u}
> $$
> 
> 由于第一项为常数，所以最终的优化目标变为
> 
> $$
> J(\mathbf{u}) = \operatorname*{argmin}_u- \mathbf{u}^T \mathbf{S} \mathbf{u}
> $$
> 

把优化目标再一转换，马上就得到了和最大投影方差相同的结论。我们只要选择特征值最大的特征向量，就能够最好地减少重构前后矩阵的误差

### SVD 与降维

OK，又绕了一大圈，讲了降维。那么 SVD 与降维之间有什么关系？现在提出 intuition：**对数据做 SVD 就是在对其进行降维**。可以看到 SVD 当中的特征向量矩阵$\boldsymbol V$，其实就是$\boldsymbol{M}^{\top}\boldsymbol{M}$的特征向量矩阵，而如果我们把$\boldsymbol M \in \mathbb R^{n\times k}$看做中心化过后的数据$X \in \mathbb R^{n\times k}$，**此时$\boldsymbol V$就是 PCA 中一直寻找的主成分！**而$\boldsymbol U \boldsymbol \Sigma$就是投影过后的坐标，因为$\boldsymbol M \boldsymbol V =\boldsymbol U \boldsymbol \Sigma$。另外由于矩阵$\boldsymbol U$的正交性，其坐标向量是相互正交的

### SVD 与低秩近似

虽然通过以上降维的论证，我们知道如何计算主成分，但是低秩近似的证明与降维当中的优化目标并不一样，不过神奇的是二者的答案都指向了 SVD。**这更一步加深了 SVD 与低秩相关的 intuition是**

定义低秩近似的优化问题：

$$
\underset{A,B}{argmin}\|AB-M\|_{F}^{2}
$$

其中$A\in\mathbb{R}^{n\times r},B\in\mathbb{R}^{r\times m},M\in\mathbb{R}^{n\times m},r<\min(n,m)$，说白了，这就是要寻找矩阵$M$的“最优$r$秩近似”。这里直接给出结论，最优解和最小值分别为

$$
A=U\Sigma,B=V^T\\
\min_{A,B}\|AB-M\|_{F}^{2}=\sum_{i=1}^r{\sigma_i^2}
$$

其中$U,\Sigma,V,\sigma$就是$M$的奇异值分解中对应的各个矩阵和奇异值，为了满足 rank 的要求，我们只取前$r$个特征值/特征向量即可，即：$U\in \mathbb R^{n\times r}, \Sigma \in \mathbb R^{r\times r}, V \in \mathbb R^{m\times r}$。注意：该解为最优解之一，不保证唯一性，可能存在其他矩阵也能达到最小值

这一节本质就是对 **Eckart-Young-Mirsky定理** 的在F-范数下的证明。在这个过程中也自然地引出了伪逆的定义。证明过程还是比较复杂，这里我只简单记录两个知识点

1. Moore-Penrose 伪逆的定义，[低秩近似之路（一）：伪逆](https://spaces.ac.cn/archives/10366)

    伪逆定义的出发点其实是寻找最优解
    
    $$
    \underset{B}{argmin}\|AB-I\|_{F}^{2}
    $$
    
    通过矩阵求导的方式最终我们可以推理得到
    
    $$
    A^\dagger = \lim_{\epsilon\to0} (A^\top A + \epsilon I_r)^{-1} A^\top
    $$
    
    如果$A^\top A$可逆，可以证明伪逆等价于$A^{-1}$。不过这个形式有点难看，因为有极限的存在，该极限保证了可逆操作的合法性。实际上我们可以利用谱分解写一个更好看的形式，首先我们有谱分解
    
    $$
    A^TA=U\Lambda U^\top
    $$
    
    带入到伪逆公式当中
    
    $$
    \begin{aligned} \left(\boldsymbol{A}^{\top} \boldsymbol{A}+\epsilon \boldsymbol{I}_{r}\right)^{-1} \boldsymbol{A}^{\top} & =\left(\boldsymbol{U} \boldsymbol{\Lambda} \boldsymbol{U}^{\top}+\epsilon \boldsymbol{I}_{r}\right)^{-1} \boldsymbol{A}^{\top} \\ & =\left[\boldsymbol{U}\left(\boldsymbol{\Lambda}+\epsilon \boldsymbol{I}_{r}\right) \boldsymbol{U}^{\top}\right]^{-1} \boldsymbol{A}^{\top} \\ & =\boldsymbol{U}\left(\boldsymbol{\Lambda}+\epsilon \boldsymbol{I}_{r}\right)^{-1} \boldsymbol{U}^{\top} \boldsymbol{A}^{\top} \end{aligned}
    $$
    
    可以看到，这里的可逆符号转移仅作用于对角矩阵，不过由于$UA$矩阵在 rank > r 过后的向量也全部为零（可由$(UA)^T(UA)=\Lambda$证明），所以把可逆矩阵中的零分之一的情况完全抵消（零乘以任何数都是零），所以直接把极限给拿掉得到形式
    
    $$
    A^\dagger =  U\Lambda^{-1} U^\top A^\top
    $$
    
    现在有了 SVD 分解，我们可以把$UA$也简化表达。注意上面讨论的所有$U$其实是 SVD 分解中的$V$，而我们下面所写的式子按照$A=U\Sigma V^\top$来写伪逆形式：
    
    $$
    A^\dagger =  V\Sigma^\dagger U^\top
    $$
    
    其中$\Sigma^\dagger$就是$\Sigma$中的非零元素取倒数然后转置

2. 单位正交矩阵不改变矩阵范数

    这很好理解：旋转不改变长度。也可以通过 trace 性质轻松证明

同时在 [SVD分解(一)：自编码器与人工智能](https://www.spaces.ac.cn/archives/4208)  提到了一种观点：SVD 与自编码器的定价性。优化一个没有激活函数的3层 MLP，等价于 SVD 的求解

$$
M_{m\times n}≈M_{m\times n}C_{n\times r}D_{r\times n}
$$

而进行 SVD 分解可以得到，此时的最优性是有保证的

$$
M_{m\times n}≈U_{m\times r} \Sigma_{r\times r} V_{n\times r}^T
$$

如果我们利用反向传播去优化矩阵$C_{n\times r}$和$D_{r\times n}$最终的估计误差一定会收敛于 SVD 分解结果的估计误差，在此可以“近似”地认为：$M_{m\times n}C_{n\times r}=U_{m\times r} \Sigma_{r\times r}$，以及$D_{r\times n}= V_{n\times r}^T$

**Review Eigen & Eigen Value**

从以上的分析中，我们其实并没有刻意地去寻找特征向量，而特征值和特征向量的形式，非常自然地从我们的推导之中出现了。这意味着特征值和特征向量在最优化中是具有显著的意义的。而“特征”一词的意义，在其中显得更加明显：如果我们选择这些特征值大的特征向量，对矩阵进行重构，那么重构前后矩阵的信息获得了最优的保留，也就是说：我们保留了矩阵的“特征”

# Question

- 为什么实对称矩阵一定可以被对角化？这其中有没有什么深层次的原因？

    询问了 Kimi & DeepSeek，二者都给出了两个过程：

    1. 证明实对称矩阵可对角化的三个步骤

    2. 引出更一般的谱定理（[Spectral Theorem](https://en.wikipedia.org/wiki/Spectral_theorem)）

        **谱定理**大致指出：**自伴算子（或矩阵）可以在某个标准正交基下被对角化**。所以，实对称矩阵可对角化是谱定理的一个直接推论。谱定理是泛函分析、量子力学等领域的基石，它将算子（或矩阵）的“结构”完全由其“谱”（即特征值的集合）来描述。

    这确实很神奇🧐
    
- 在学习 SVD 的过程中还在科学空间中看到了激活函数以及 EM 算法的相关解读，包含了 silu 为什么会 work，EM 算法与梯度下降算法之间的联系

    [浅谈神经网络中激活函数的设计](https://spaces.ac.cn/archives/4647)

    [梯度下降和EM算法](https://www.spaces.ac.cn/archives/4277)
    
- 矩阵分块与矩阵求导的基础

    [矩阵求导术-上](https://zhuanlan.zhihu.com/p/24709748) 我应该在研究生时期就看过这一篇，不过当时完全没看明白。当我接触了过一些 trace 相关的技巧后，再来看感觉轻松很多，很多思想都非常值得学习，而且配合了不少例子，绝对是一篇不可多得的高质量博客。**这里仅讨论标量对矩阵的求导，不讨论向量/矩阵对矩阵的求导。**矩阵对矩阵的求导遵循另外的求导法则
    
    向量对矩阵求导从形式上来说非常简单，就是对每一个元素逐个求导，最后形成矩阵
    
    $$
    \frac{\partial f}{\partial X}=\left[\frac{\partial f}{\partial X_{ij}}\right]
    $$
    
    然而，这个定义在计算中并不好用，实用上的原因是对函数较复杂的情形难以逐元素求导；哲理上的原因是逐元素求导破坏了**整体性**。所以我们希望在求导时不拆开矩阵，而是直接通过矩阵运算，从整体出发推导出结果。首先我们复习一下一元微分和多元微分
    
    $$
    df = f'(x)dx\\
    df = \sum_{i=1}^{n} \frac{\partial f}{\partial x_i} dx_i = \frac{\partial f}{\partial x}^T d\boldsymbol{x}
    $$
    
    多元微分中，我们可以把其看待成为导数与微分之间的内积。现在我们来看矩阵微分
    
    $$
    df = \sum_{i=1}^{m} \sum_{j=1}^{n} \frac{\partial f}{\partial X_{ij}} dX_{ij} = \operatorname{tr} \left( \frac{\partial f}{\partial X}^T dX \right)
    $$
    
    我们把矩阵微分也用矩阵乘法 + trace 的形式表达出来了。其中使用了一个非常重要的 trace trick：$\sum_{i,j} A_{ij}B_{ij}=\operatorname {tr}(A^TB)$，可以看到 trace 在矩阵求导的过程中是非常常见的形式，所以会频繁使用到一些 trace trick，所以掌握常用的 trace trick 是非常有必要的（好在这些 trick 都不是很复杂，也很好理解）
    
    在进行矩阵求导之前，回顾我们是如何对一元函数求导的：我们首先建立了初等函数的求导结果、推导出求导的四则运算、建立复合函数的求导法则（i.e. 链式求导法则），通过利用链式求导法则和四则运算法则将函数求导进行逐渐的拆分，直到只对初等函数进行求导，最终将结果整合起来，获得最终的导数表达。**所以，现在我们建立了矩阵求导的矩阵形式，我们还需要建立矩阵微分的运算法则来拆解复杂的矩阵函数**
    
    1. 基础运算
        - 加减法：$d(X\pm Y)=dX\pm dY$
        - 矩阵乘法：$d(XY)=(dX)Y+XdY$
        - 转置：$d(X^{T})=(dX)^{T}$
        - 迹：$d\operatorname{tr}(X)=\operatorname{tr}(dX)$
    
    2. 逆
        -$dX^{-1}=-X^{-1}dXX^{-1}$。此式可在$XX^{-1}=I$两侧求微分来证明。
    
    3. 行列式
        -$d|X|=\operatorname{tr}(X^{\#}dX)$，其中$X^{\#}$表示$X$的伴随矩阵，在X可逆时又可以写作$d|X|=|X|\operatorname{tr}(X^{-1}dX)$。此式可用 Laplace 展开来证明，详见张贤达《矩阵分析与应用》第279页。
    
    4. 逐元素乘法
        -$d(X\odot Y)=dX\odot Y+X\odot dY$，$\odot$表示尺寸相同的矩阵逐元素相乘。
    
    5. 逐元素函数
    
        -$d\sigma(X)=\sigma^{\prime}(X)\odot dX$，$\sigma(X)=[\sigma(X_{ij})]$是逐元素标量函数运算，$\sigma^{\prime}(X)=[\sigma^{\prime}(X_{ij})]$是逐元素求导数
    
    6. 复合函数
    
        无法直接套用一元的复合函数公式，因为我们没有矩阵对矩阵求导的定义。所以唯一的方法是将符合函数中的微分形式进行带入
        
        $$
        df=\operatorname{tr}(\frac{\partial f}{\partial Y}^TdY)=\operatorname{tr}(\frac{\partial f}{\partial Y}^Tdg(X))
        $$
    
    法则看上去很多，但是基本都符合我们之前的一元情况，除了逆和行列式。也如之前所说，**要完成矩阵求导，我们还需要一些 trace trick**
    
    1. 矩阵乘法的 trace 展开：$\operatorname{tr}(A^\top B)=\operatorname{tr}(B^\top A)=\Sigma_{i,j}A_{ij}B_{ij}$
    
    2. 标量套上迹：$a = \operatorname{tr}(a)$
    
    3. trace 转置性质：$\operatorname{tr}(A^{T}) = \operatorname{tr}(A)$
    
    4. trace 线性性质：$\operatorname{tr}(A \pm B) = \operatorname{tr}(A) \pm \operatorname{tr}(B)$
    
    5. trace 中矩阵乘法可交换：$\operatorname{tr}(AB) = \operatorname{tr}(BA)$，需要保证$AB, BA$是可行的矩阵乘法。该性质还可拓展为 trace 循环
        
        $$
        \operatorname{tr}(ABC)=\operatorname{tr}(BCA)=\operatorname{tr}(CAB)
        $$
    
    6. trace 中矩阵乘法/逐元素乘法可交换：$\operatorname{tr}(A^{T}(B \odot C)) = \operatorname{tr}((A \odot B)^{T} C)$两侧都等于$\sum_{i,j} A_{ij} B_{ij} C_{ij}$
    
        左侧：$\operatorname{tr}(A^T(B \odot C)) = \sum_{i,j} A_{ij} (B \odot C)_{ij} = \sum_{i,j} A_{ij} B_{ij} C_{ij}$
    
        右侧：$\operatorname{tr}((A \odot B)^T C) = \sum_{i,j} (A \odot B)_{ij} C_{ij} = \sum_{i,j} A_{ij} B_{ij} C_{ij}$
    
    由于 trace 的引入，我们可以对矩阵的乘法顺序方便地进行交换，从而将$dX$放到最右侧，所以我们通常用的技巧是：把标量导数套上 trace，通过 trace trick 和微分符号与 trace 可互换的性质，调整$dX$到所需位置，最后对比$\operatorname{tr}(\frac{\partial f}{\partial X}^TdX)$和$\operatorname{tr}(g(X)dX)$，我们即可确定$g(X)=\frac{\partial f}{\partial X}$
    
    我们以最常见的两个例子来说明
    
    1. **Example 1:**$f=a^TXb$，其中$a\in \mathbb R^{m\times1}, b\in \mathbb R^{n\times 1}, X\in \mathbb R^{m\times n}$
    
        思路：先利用标量套上 trace 不变的性质，变为 trace 形式，然后利用 trace 矩阵乘法可交换性质把$dX$换到最右侧
        
        $$
        df=\operatorname{tr}(a^{T}dXb)=\operatorname{tr}(ba^{T}dX)=\operatorname{tr}((ab^{T})^{T}dX)
        $$
        
        按照上述思路与$\operatorname{tr}(\frac{\partial f}{\partial X}^TdX)$形式对比，我们可以立刻得到导数为
        
        $$
        \frac{\partial f}{\partial X}=ab^{T}
        $$
    
    2. **Example 2:**$f = \operatorname{tr}(W^T X^TX W)$，其中$X \in \mathbb R^{m\times k}, W \in \mathbb R^{k\times n}$，这其实就是求解 F-范数平方的导数$f=||XW||_F^2$，其中我们关注的是$W$的导数
    
        思路：既然已经套上 trace 了，直接利用四则运算中的矩阵乘法微分进行拆分，然后逐个求解。为了简便我们用$M=X^TX$来表示一个对称矩阵
        
        $$
        df = \operatorname{tr}((dW)^T M W) + \operatorname{tr}(W^T M dW) = \operatorname{tr}(W^T M^T dW) + \operatorname{tr}(W^T M dW)
        $$
        
        上述第二个等号利用了 trace 的转置性质，现在结果已经明了
        
        $$
        \frac{\partial f}{\partial W}=(M^T+M)W=2MW=2X^TXW
        $$
