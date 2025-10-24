import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium", app_title="LLMs-编写注意力机制")

with app.setup(hide_code=True):
    # Initialization code that runs before all other cells
    import marimo as mo
    import torch
    import torch.nn as nn


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    # 2. 编写注意力机制
    > 本章介绍数 `Transformer` 模型的注意力机制（`Attention Mechanisms`）。

    ---

    ![2-1](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-1.svg)


    实现的具体过程如下：

    ![2-2](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-2.svg)

    ---

    ## 2.1 长序列模型的问题

    > 由于源语言和目标语言的语法结构存在差异，逐字翻译文本是不可行的。

    ![2-3](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-3.svg)

    在 `Transformer` 模型出现之前，编码器-解码器 RNN 通常用于机器翻译任务。

    在这种设置中，编码器处理源语言的一系列标记，并使用隐藏状态（神经网络中的一种中间层）生成整个输入序列的精简表示：

    ![2-4](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-4.svg)

    ## 2.2 注意力机制介绍

    > 通过以上翻译过程可以看出，再一次翻译中，句子中的每个单词重要程度是不同，对于某些单词我们需要格外关注。这就是注意力机制所完成的任务。

    ![2-5](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-5.svg)

    `Transformer` 中的自注意力机制是一种旨在增强输入表示的技术，它使序列中的每个位置能够与同一序列中其他每个位置互动并确定其相关性

    ![2-6](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-6.svg)

    ## 2.3 简单自注意力机制

    ### 2.3.1 自注意力计算过程

    > 探索一个不可训练的自注意力机制权重计算过程。

    :cloud: 假如现在要计算 `journey` 单词这这句话中的重要程度：
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.image(
        src="https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-07.svg",
        width=1440,
    )
    return


@app.function(hide_code=True)
def inputs_table_data(inputs_words, inputs):
    data = []
    words = inputs_words.split(" ")
    for i, item in enumerate(inputs):
        npv = inputs[i].numpy()
        vector = f"[{str(npv[0])}, {str(npv[1])}, {str(npv[2])}]"
        data.append({"ID": i, "Word": words[i], "Vector": vector})
    return data


@app.cell(hide_code=True)
def _():
    inputs_words = "Your journey starts with one step"

    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],  # Your     (x^1)
            [0.55, 0.87, 0.66],  # journey  (x^2)
            [0.57, 0.85, 0.64],  # starts   (x^3)
            [0.22, 0.58, 0.33],  # with     (x^4)
            [0.77, 0.25, 0.10],  # one      (x^5)
            [0.05, 0.80, 0.55],  # step     (x^6)
        ]
    )

    mo.ui.table(data=inputs_table_data(inputs_words, inputs))
    return (inputs,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    图中展示了该过程的初始步骤，即通过点积运算计算 $x^{(2)}$ 与所有其他输入元素之间的注意力得分 $\omega$

    ![2-8](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-8.svg)
    """
    )
    return


@app.cell
def _():
    0.55 * 0.43 + 0.87 * 0.15 + 0.66 * 0.89
    return


@app.cell
def _():
    def attn_scores_2v(inputs, query):
        scores = torch.empty(inputs.shape[0])
        for i, x_i in enumerate(inputs):
            scores[i] = torch.dot(
                x_i, query
            )  # dot product (transpose not necessary here since they are 1-dim vectors)
        return scores


    def attn_scores_2(inputs, query):
        result = []
        scores = torch.empty(inputs.shape[0])
        for i, x_i in enumerate(inputs):
            scores[i] = torch.dot(
                x_i, query
            )  # dot product (transpose not necessary here since they are 1-dim vectors)
            value = scores[i].numpy()
            result.append(
                f":radio_button: $Score_{{1{i}}} = Q_1 \cdot X_{i}$ = {value:.4}"
            )
        return "<br/>".join(result)
    return attn_scores_2, attn_scores_2v


@app.cell(hide_code=True)
def _(attn_scores_2, inputs):
    query = inputs[1]  # 2nd input token is the query "journey"

    mo.md(
        f"""
    {attn_scores_2(inputs, query)}

    :fire: 点积本质上是将两个向量元素相乘，然后对所得乘积求和的简写
    """
    )
    return (query,)


@app.cell
def _(inputs, query):
    def test_dot(qidx):
        res = 0.0
        for idx, element in enumerate(inputs[qidx]):
            res += inputs[qidx][idx] * query[idx]
        return res


    _tmp_r = []
    for qidx in range(len(inputs)):
        _tmp_r.append(test_dot(qidx))

    mo.md(
        f"""
    "journey"在每个单词上的注意力分数：
    ```python
    {torch.tensor(_tmp_r)}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    可以看到每个词汇相对于输入的查询 `"journey"` 的注意力分数被计算出来，但是有一个问题：

    /// attention | 问题出现了
    计算出来的值没有一个固定范围，**和**页是不固定的。
    ///

    :hammer: 为了便于计算每个词汇的注意力占比，需要将其归一化

    ![2-9](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-9.svg)
    """
    )
    return


@app.cell
def _(attn_scores_2v, inputs, query):
    attn_scores_2_value = attn_scores_2v(inputs, query)
    attn_weights_2_tmp = attn_scores_2_value / attn_scores_2_value.sum()

    mo.md(
        f"""
    归一化结果为：
    ```python
    {attn_weights_2_tmp}
    ```

    张量总和为 `{attn_weights_2_tmp.sum():.4}`
    """
    )
    attn_scores_2_value = attn_scores_2v(inputs, query)
    attn_weights_2_tmp = attn_scores_2_value / attn_scores_2_value.sum()

    mo.md(
        f"""
    归一化结果为：
    ```python
    {attn_weights_2_tmp}
    ```

    张量总和为 `{attn_weights_2_tmp.sum():.4}`
    """
    )
    return (attn_scores_2_value,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r""":rocket: 以上代码是一种最简单的归一化方式，但是在实践中，使用 `softmax` 函数进行归一化是常见的，也是推荐的做法，因为它更擅长处理极值，并且在训练过程中具有更理想的梯度特性。"""
    )
    return


@app.cell
def _(attn_scores_2_value):
    def softmax_naive(x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)


    attn_weights_2_naive = softmax_naive(attn_scores_2_value)

    mo.md(
        f"""
    `Softmax` 归一化结果为：
    ```python
    {attn_weights_2_naive}
    ```

    张量总和为 `{attn_weights_2_naive.sum():.4}`
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :cloud: 但是以上方式仍然会有问题。

    /// attention | 问题出现了
    由于溢出和下溢问题，上述简单的实现可能会在输入值较大或较小时出现数值不稳定的问题。
    ///

    :rocket: 因此，在实践中，建议使用 PyTorch 的 softmax 实现，该实现已针对性能进行了高度优化。
    """
    )
    return


@app.cell(hide_code=True)
def _(attn_scores_2_value):
    attn_weights_2 = torch.softmax(attn_scores_2_value, dim=0)

    mo.md(
        f"""
    `torch.softmax` 归一化结果为：
    ```python
    {attn_weights_2}
    ```

    张量总和为 `{attn_weights_2.sum():.4}`
    """
    )
    return (attn_weights_2,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    计算上下文向量 $𝑧^{(2)}$ 是通过将嵌入的输入标记 $𝑥^{(i)}$ 与注意力权重相乘，并将得到的向量相加：

    ![2-10](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-10.svg)
    """
    )
    return


@app.cell
def _(attn_weights_2, inputs):
    _tmp_query = inputs[1]  # 2nd input token is the query

    _tmp_context_vec_2 = torch.zeros(_tmp_query.shape)
    for _tmp_i, _tmp_x_i in enumerate(inputs):
        print(f"{attn_weights_2[_tmp_i].numpy():.4} * {_tmp_x_i.numpy()}")
        _tmp_context_vec_2 += attn_weights_2[_tmp_i] * _tmp_x_i

    print(_tmp_context_vec_2)
    return


@app.cell
def _():
    (
        0.1385 * 0.43
        + 0.2379 * 0.55
        + 0.2333 * 0.57
        + 0.124 * 0.22
        + 0.1082 * 0.77
        + 0.1581 * 0.05
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### 2.3.2 计算注意力权重

    > 接下来，我们将此计算推广到计算所有注意力权重和上下文向量。

    ![2-11](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-11.svg)

    /// details | :fire: 计算上下文向量的过程

    :radio_button: **(1/3)** 在自注意力机制中，首先计算注意力分数。<br/>
    :radio_button: **(2/3)** 然后对其进行归一化，得出总注意力权重为 1 的注意力权重。<br/>
    :radio_button: **(3/3)** 最后通过对输入进行加权求和，生成上下文向量。
    ///
    """
    )
    return


@app.cell
def _(inputs):
    attn_scores_native = torch.empty(6, 6)

    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores_native[i, j] = torch.dot(x_i, x_j)

    mo.md(
        f"""
    **(1/3)** 所有单词的注意力分数计算结果：
    ```python
    {attn_scores_native}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r""":hammer: 我们可以通过矩阵乘法更有效地实现上述目标：""")
    return


@app.cell(hide_code=True)
def _(inputs):
    attn_scores = inputs @ inputs.T

    mo.md(
        f"""
    **(1/3)** 所有单词的注意力分数计算结果：
    ```python
    {attn_scores}
    ```
    """
    )
    return (attn_scores,)


@app.cell
def _(attn_scores):
    attn_weights = torch.softmax(attn_scores, dim=-1)

    mo.md(
        f"""
    **(2/3)** 将计算结果归一化的结果：
    ```python
    {attn_weights}
    ```

    可以计算得出每个单词的注意力权重总和为 `1`：
    ```python
    {attn_weights.sum(dim=-1)}
    ```
    """
    )
    return (attn_weights,)


@app.cell(hide_code=True)
def _(attn_weights, inputs):
    all_context_vecs = attn_weights @ inputs

    mo.md(
        f"""
    **(3/3)** 加权求和，生成上下文向量：
    ```python
    {all_context_vecs}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.hstack(
        [
            mo.mermaid(
                """
    graph TB;
    S1("确定$$Query(Q)$$") --> S2("计算点积$$(\\omega = Q \\cdot X)$$") --> S3("归一化$$(\\alpha = Softmax(\\omega))$$") --> S4("计算上下文向量$$(Z = \\sum_{i=0}^{T} \\alpha_i X^{i})$$")
    """
            ),
            mo.mermaid(
                """
    graph TB;
    S1("$$Q\\in\\mathbb{R}^{1 \\times 3}$$") --> S2("$$\\omega = Q \\cdot X, X\\in\\mathbb{R}^{1 \\times 3}, \\omega\\in\\mathbb{R}$$") --> S3("$$\\omega_{0-T}=\\{\\omega_0, \\omega_1, ..., \\omega_T\\},\\omega_{0-T}\\in\\mathbb{R}^{1 \\times 6}$$") --> S4("$$\\alpha_{0-T} = Softmax(\\omega_{0-T}), \\alpha_{0-T}\\in\\mathbb{R}^{1 \\times 6}$$") --> S5("$$Z_j = \\sum_{i=0}^{T}\\alpha_i X^{(i)},X\\in\\mathbb{R}^{1 \\times 3},Z\\in\\mathbb{R}^{1 \\times 3}$$") --> S6("$$Z_{0-T}=\\{Z_0,Z_1,...,Z_T\\},Z\\in\\mathbb{R}^{6 \\times 3}$$")
    """
            ),
        ],
        justify="center",
        gap=5,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 2.4 实现可训练的注意力权重

    > 本节内容介绍如何实现一个可以训练的注意力机制

    ![2-12](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-12.svg)

    ### 2.4.1 逐步计算注意力权重

    > 在本节中，我们将实现原始 Transformer 架构、GPT 模型和大多数其他流行 LLM 中使用的自注意力机制。这种自注意力机制也被称为 "缩放点积注意力" （`scaled dot-product attention`）

    + :cloud: 总体思路与之前类似：我们希望将上下文向量计算为特定输入元素的输入向量的加权和。为此，我们需要注意力权重。
    + :cloud: 与之前介绍的基本注意力机制相比，只有细微的差别：最显着的区别是引入了在模型训练期间更新的权重矩阵。
    + :cloud: 这些可训练的权重矩阵至关重要，可以帮助模型学习生成“**良好**”的上下文向量。

    ![2-13](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-13.svg)

    :rocket: 要逐步实现自注意力机制，我们需要先来介绍一下三个可训练的权重矩阵$W_q, W_k, W_v$。这三个矩阵用于将嵌入的输入标记 $x^{(i)}$通过矩阵乘法投影到查询、键和值向量中：

    /// admonition | $QKV矩阵$

    在深度学习，尤其是在自然语言处理和计算机视觉领域，QKV 矩阵通常与自注意力机制（Self-Attention）相关联。以下是 Q、K 和 V 矩阵各自的含义及其作用：

    #### 1. Q 矩阵（查询矩阵）

    - **含义**：Q 代表查询（Query），它是输入序列中每个位置的表示，通常通过对输入特征进行线性变换得到。
    - **作用**：Q 矩阵用于计算输入序列中各个位置之间的相关性。具体来说，它通过与 K 矩阵进行点积来生成注意力权重，决定了每个位置对其他位置的关注程度。

    #### 2. K 矩阵（键矩阵）

    - **含义**：K 代表键（Key），它同样是通过对输入特征进行线性变换得到的。K 矩阵提供了每个位置的“标识”或“标签”。
    - **作用**：K 矩阵用于和 Q 矩阵进行匹配，以计算注意力分数。通过计算 Q 和 K 的点积，模型可以评估查询与各个键之间的相似度，从而确定注意力的分配。

    #### 3. V 矩阵（值矩阵）

    - **含义**：V 代表值（Value），它是输入序列中每个位置的实际信息表示，也是通过线性变换得到的。
    - **作用**：V 矩阵包含了实际需要传递的信息。在计算注意力时，Q 和 K 的点积结果（注意力权重）会用于加权 V 矩阵中的值，生成最终的输出表示。换句话说，V 矩阵提供了根据注意力权重加权后的信息。

    #### 总结

    - **Q 矩阵**：用于查询，计算与 K 的相关性。
    - **K 矩阵**：用于匹配查询，提供每个位置的标识。
    - **V 矩阵**：包含实际的信息，根据注意力权重加权后生成输出。

    #### 自注意力机制的流程

    1. **输入**：输入序列经过线性变换得到 Q、K、V 矩阵。
    2. **计算注意力分数**：通过点积计算 Q 和 K 的相似度，并通过 Softmax 函数得到注意力权重。
    3. **加权求和**：将注意力权重应用于 V 矩阵，得到最终的输出。

    这种机制使得模型能够在处理输入序列时，自适应地关注不同位置的相关信息，从而提高了上下文理解能力。

    + $Query^{(i)} = x^{(i)} \times W_{q}^{(i)}$
    + $Key^{(i)} = x^{(i)} \times W_{k}^{(i)}$
    + $Value^{(i)} = x^{(i)} \times W_{v}^{(i)}$

    + 算法角度

    ![2-26](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/Example-01.png)

    + 应用角度

    ![2-26](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-27.webp)

    ///

    :hammer: 输入和查询向量的嵌入维度可以相同或不同，这取决于模型的设计和具体实现。在 GPT 模型中，输入和输出维度通常相同，但为了便于说明，为了更好地跟踪计算，我们在这里选择不同的输入和输出维度：
    """
    )
    return


@app.cell(hide_code=True)
def _(inputs):
    x_2 = inputs[1]  # second input element
    d_in = inputs.shape[1]  # the input embedding size, d=3
    d_out = 2  # the output embedding size, d=2

    print(x_2)

    mo.show_code()
    return d_in, d_out, x_2


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    $$
    A(A\in\mathbb{R}^{M \times N}) \times B(B\in\mathbb{R}^{N \times P}) = C(C\in\mathbb{R}^{M \times P})
    $$

    考虑两个矩阵 \( A \) 和 \( B \)：

    \[
    A = \begin{pmatrix}
    1 & 2 & 3 \\
    4 & 5 & 6
    \end{pmatrix}, \quad
    B = \begin{pmatrix}
    7 & 8 \\
    9 & 10 \\
    11 & 12
    \end{pmatrix}
    \]

    计算过程如下：

    \[
    C = A \times B = \begin{pmatrix}
    (1 \cdot 7) + (2 \cdot 9) + (3 \cdot 11) = 7 + 18 + 33 = 58 & (1 \cdot 8) + (2 \cdot 10) + (3 \cdot 12) = 8 + 20 + 36 = 64 \\
    (4 \cdot 7) + (5 \cdot 9) + (6 \cdot 11) = 28 + 45 + 66 = 139 & (4 \cdot 8) + (5 \cdot 10) + (6 \cdot 12) = 32 + 50 + 72 = 154
    \end{pmatrix}
    \]

    ---

    :hammer: 下面，我们初始化三个权重矩阵；请注意，为了便于说明，我们设置了 `require_grad=False` 以减少输出中的混乱，但如果我们要使用权重矩阵进行模型训练，我们将设置 `require_grad=True` 以在模型训练期间更新这些矩阵。
    """
    )
    return


@app.cell(hide_code=True)
def _(d_in, d_out):
    torch.manual_seed(123)

    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    mo.md(
        f"""
    + $W_{{q}}$
    ```python
    {W_query}
    ```

    + $W_{{k}}$
    ```python
    {W_key}
    ```

    + $W_{{v}}$
    ```python
    {W_value}
    ```
    """
    )
    return W_key, W_query, W_value


@app.cell(hide_code=True)
def _():
    mo.md(r""":hammer: 接下来我们计算`Query`、`Key`和`Value`向量：""")
    return


@app.cell(hide_code=True)
def _(W_key, W_query, W_value, inputs, x_2):
    query_2 = (
        x_2 @ W_query
    )  # _2 because it's with respect to the 2nd input element
    keys = inputs @ W_key
    values = inputs @ W_value

    mo.md(
        f"""
    $X^{{(2)}}:$
    ```python
    {x_2}
    ```

    $Query^{{(2)}}:$
    ```python
    {query_2}
    ```

    $Keys:$
    ```python
    {keys}

    # Shape:
    {keys.shape}
    ```

    $Values:$
    ```python
    {values}

    #Shape:
    {values.shape}
    ```
    """
    )
    return keys, query_2, values


@app.cell
def _():
    0.5500 * 0.2961 + 0.8700 * 0.2517 + 0.6600 * 0.0740
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :hammer: 通过计算`Query`和每个`Key`向量之间的点积来计算非标准化（未归一化）注意力分数：

    ![2-14](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-14.svg)
    """
    )
    return


@app.cell(hide_code=True)
def _(keys, query_2):
    keys_2 = keys[1]  # Python starts index at 0
    attn_score_22 = query_2.dot(keys_2)

    mo.md(
        f"""
    Code:
    ```python
    keys_2 = keys[1]  # Python starts index at 0
    attn_score_22 = query_2.dot(keys_2)
    ```

    既：
    $W_{{k^{{(1)}}}} \\cdot Q^{{(2)}}$

    ---

    + $W_{{k^{{(1)}}}}$
    ```python
    {keys_2}
    ```

    + $Q^{{(2)}}$
    ```python
    {query_2}
    ```

    + Result:
    ```python
    {attn_score_22}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r""":hammer: 由于我们有 6 个输入，因此对于给定的查询向量我们有 6 个注意力分数："""
    )
    return


@app.cell(hide_code=True)
def _(keys, query_2):
    attn_scores_u2 = query_2 @ keys.T  # All attention scores for given query

    mo.md(
        f"""
    Code:
    ```python
    attn_scores_u2 = query_2 @ keys.T  # All attention scores for given query
    ```

    Result:
    ```python
    {attn_scores_u2}
    ```
    """
    )
    return (attn_scores_u2,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :hammer: 现在得到的是未归一化的注意力权重，接下来需要对其进行归一化操作。

    ![2-15](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-15.svg)
    """
    )
    return


@app.cell(hide_code=True)
def _(attn_scores_u2, keys):
    d_k = keys.shape[1]
    attn_weights_s2 = torch.softmax(attn_scores_u2 / d_k**0.5, dim=-1)


    mo.md(
        f"""
    Code:
    ```python
    d_k = keys.shape[1]
    attn_weights_s2 = torch.softmax(attn_scores_u2 / d_k**0.5, dim=-1)
    ```

    Result:
    ```python
    {attn_weights_s2}
    ```
    """
    )
    return (attn_weights_s2,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :hammer: 计算上下文向量：

    ![2-16](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-16.svg)
    """
    )
    return


@app.cell(hide_code=True)
def _(attn_weights_s2, values):
    context_vec_2 = attn_weights_s2 @ values

    mo.md(
        f"""
    $Attn$
    ```python
    {attn_weights_s2}
    ```

    $Values$
    ```python
    {values}
    ```

    $Context$
    ```python
    {context_vec_2}
    ```

    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""### 2.4.2 封装自注意力机制类""")
    return


@app.cell(hide_code=True)
def _(d_in, d_out, inputs):
    class SelfAttention_v1(nn.Module):

        def __init__(self, d_in, d_out):
            super().__init__()
            self.W_query = nn.Parameter(torch.rand(d_in, d_out))
            self.W_key = nn.Parameter(torch.rand(d_in, d_out))
            self.W_value = nn.Parameter(torch.rand(d_in, d_out))

        def forward(self, x):
            keys = x @ self.W_key
            queries = x @ self.W_query
            values = x @ self.W_value

            attn_scores = queries @ keys.T  # omega
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5, dim=-1
            )

            context_vec = attn_weights @ values
            return context_vec


    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print(sa_v1(inputs))
    mo.show_code()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :rocket: 整体计算过程如下：

    ![2-17](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-17.svg)

    :hammer: 我们可以使用`PyTorch`的`Linear`层简化`SelfAttention`的代码：
    """
    )
    return


@app.cell(hide_code=True)
def _(d_in, d_out, inputs):
    class SelfAttention_v2(nn.Module):

        def __init__(self, d_in, d_out, qkv_bias=False):
            super().__init__()
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        def forward(self, x):
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5, dim=-1
            )

            context_vec = attn_weights @ values
            return context_vec


    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print(sa_v2(inputs))

    mo.show_code()
    return (sa_v2,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :pushpin: 请注意，`SelfAttention_v1` 和 `SelfAttention_v2` 给出不同的输出，因为它们对权重矩阵使用不同的初始权重。

    ## 2.5 用因果注意力隐藏未来的词语

    > 在因果注意力机制中，对角线上方的注意力权重被掩盖，确保对于任何给定的输入，LLM 在利用注意力权重计算上下文向量时无法利用未来的`Token`。

    ![2-18](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-18.svg)

    ### 2.5.1 应用因果注意力掩码

    > 因果自注意力机制确保模型对序列中某个位置的预测仅依赖于先前位置的已知输出，而不依赖于未来位置。简而言之，这确保每个下一个单词的预测仅依赖于前面的单词。
    > 
    > 为了实现这一点，对于每个给定的`Token`，我们屏蔽掉未来的`Token`（即输入文本中当前`Token`之后的`Token`）：

    ![2-19](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-19.svg)

    :cloud: 为了说明和实现因果自注意力，让我们使用上一节中的注意力分数和权重：
    """
    )
    return


@app.cell(hide_code=True)
def _(inputs, keys, sa_v2):
    queries_v2 = sa_v2.W_query(inputs)
    keys_v2 = sa_v2.W_key(inputs)
    attn_scores_v2 = queries_v2 @ keys_v2.T
    attn_weights_v2 = torch.softmax(attn_scores_v2 / keys.shape[-1] ** 0.5, dim=-1)

    print(attn_weights_v2)

    mo.show_code()
    return (attn_weights_v2,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r""":hammer: 掩盖未来注意力权重的最简单方法是通过 `PyTorch` 的 `tril` 函数创建一个掩码，将主对角线下方的元素（包括对角线本身）设置为 `1`，将主对角线上方的元素设置为 `0`："""
    )
    return


@app.cell(hide_code=True)
def _():
    get_tril_width, set_tril_width = mo.state(2)
    get_tril_height, set_tril_height = mo.state(2)
    import numpy as np


    def width_slider_on_change(value):
        set_tril_width(value)


    def height_slider_on_change(value):
        set_tril_height(value)


    width_slider = mo.ui.slider(
        start=2,
        stop=12,
        step=1,
        on_change=width_slider_on_change,
        label="Tensor Rows:",
    )
    height_slider = mo.ui.slider(
        start=2,
        stop=12,
        step=1,
        on_change=height_slider_on_change,
        label="Tensor Columns",
    )


    def generate_latex_tril_matrix(width, height):
        # 创建一个全零的矩阵
        matrix = np.zeros((height, width))

        # 填充下三角部分
        for i in range(min(height, width)):
            for j in range(i + 1):
                matrix[i][j] = 1

        # 生成 LaTeX 矩阵格式
        latex_matrix = "$$\n\\left[\\begin{matrix}\n"
        for row in matrix:
            latex_matrix += " & ".join(map(str, row.astype(int))) + " \\\\\n"
        latex_matrix += "\\end{matrix}\\right]\n$$"

        return latex_matrix
    return (
        generate_latex_tril_matrix,
        get_tril_height,
        get_tril_width,
        height_slider,
        width_slider,
    )


@app.cell(hide_code=True)
def _(
    generate_latex_tril_matrix,
    get_tril_height,
    get_tril_width,
    height_slider,
    width_slider,
):
    mo.vstack(
        [
            mo.md("`torch.tril` Example:"),
            width_slider,
            height_slider,
            mo.md(generate_latex_tril_matrix(get_tril_width(), get_tril_height())),
        ]
    )
    return


@app.cell(hide_code=True)
def _(attn_scores):
    context_length = attn_scores.shape[0]
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print(mask_simple)

    mo.show_code()
    return context_length, mask_simple


@app.cell(hide_code=True)
def _():
    mo.md(
        r""":hammer: 将 `mask_simple` 与 `attn_weights_v2` 相乘即可得到掩码后的注意力权重："""
    )
    return


@app.cell(hide_code=True)
def _(attn_weights_v2, mask_simple):
    masked_simple = attn_weights_v2 * mask_simple

    print(masked_simple)
    mo.show_code()
    return (masked_simple,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r""":fire: 现在虽然得到了我们想要的矩阵，但是`Softmax`的结果被破坏了，无法满足每行的总和为`1`，因此我们可以用如下方式解决："""
    )
    return


@app.cell(hide_code=True)
def _(masked_simple):
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums

    print(masked_simple_norm)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :fire: 虽然我们现在从技术上已经完成了因果注意力机制的编码，但让我们简要地看一下实现上述相同目标的更有效的方法。

    :hammer: 我们不是将对角线上方的注意力权重清零并重新规范化结果，而是可以在对角线上方未规范化的注意力分数进入 `softmax` 函数之前用负无穷大进行屏蔽：

    ![2-20](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-20.svg)
    """
    )
    return


@app.cell(hide_code=True)
def _(attn_scores, context_length):
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print(mask)
    print(masked)
    mo.show_code()
    return (masked,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    因为 `mask` 是在 `softmax` 之前被**加到注意力得分上**的。

    + 1️⃣ softmax 的数学性质：

    $$
    \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
    $$

    如果掩码位置填的是 **0**，那么：

    * 被 `mask` 的 `token` 的分数不受影响；
    * `softmax` 仍然会分配一部分概率给这些位置；
    * 模型仍然可能“偷看”未来的 `token` —— 因果性被破坏。

    ---

    + 2️⃣ 如果掩码位置是 **−∞**（或在数值实现中一个非常大的负数，如 −1e9）：

    $$
    e^{-\infty} = 0
    $$

    则 `softmax` 输出中对应位置的概率严格为 0，
    即该位置被完全屏蔽，不会对注意力结果产生任何影响。

    ---

    初始化注意力掩码矩阵时，对角线上方（未来 token 区域）使用 **−∞** 而不是 0，是为了在 `softmax` 中将这些位置的注意力概率严格压制为 0，从而确保模型的因果性（即当前 `token` 只能看见过去和自己，不能看见未来）。
    """
    )
    return


@app.cell(hide_code=True)
def _(keys, masked):
    attn_weights_v3 = torch.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)

    print(attn_weights_v3)
    mo.show_code()
    return (attn_weights_v3,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### 2.5.2 使用Dropout掩盖注意力权重

    > 我们经常使用`Dropout`的方式来避免训练过程中的过度拟合，其核心原理就是在训练时会随机屏蔽掉一部分注意力权重，以防止训练时过度依赖某个参数。

    ![2-21](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-21.svg)

    :rocket: 如果我们应用 `0.5（50%` 的 `dropout` 率，则未丢弃的值将相应缩放 $\frac{1}{0.5} = 2$。

    这是一个非常核心但容易被忽略的细节，下面我给出正式、系统的解释。

    ---

    /// admonition | Dropout 的本质

    Dropout 的目的是在训练时随机丢弃（置零）部分神经元的输出，以防止过拟合。
    设一个神经元输出为$x$，`Dropout`以概率 $p$ 丢弃该神经元（即令输出为 0），以概率 $1-p$ 保留。

    于是我们定义掩码（mask）：

    $$
    m_i \sim \text{Bernoulli}(1-p)
    $$

    > “随机变量 $m_i$ 服从参数为 $1-p$ 的伯努利分布（Bernoulli distribution）。”

    Dropout 后的输出：

    $$
    y_i = m_i \cdot x_i
    $$

    ---

    没有被剔除的值会翻倍的关键在于 **保持期望一致（expected value consistency）**。

    > 我们希望在 **训练时使用 Dropout** 和 **推理时不使用 Dropout**，两者的输出期望一致，否则网络在推理阶段的激活分布会和训练时不一致，导致性能下降。

    设神经元原始输出为$x$，丢弃概率为$p$。

    * **训练时（使用 Dropout）**

    $$
    E[y] = E[m \cdot x] = (1-p) \cdot x
    $$

    * **推理时（不使用 Dropout）**

    $$
    y_{\text{infer}} = x
    $$

    可以看到期望值变了，从 $(1-p)x$ 变成 $p$。

    为保持一致，我们在训练时将未丢弃的神经元输出除以 $1-p$，即：

    $$
    y = \frac{m \cdot x}{1-p}
    $$

    此时：

    $$
    E[y] = (1-p) \cdot \frac{x}{1-p} = x
    $$

    这样训练和推理的输出分布就一致了。

    ---

    :fire: Dropout 时未被丢弃的值变大，是因为在训练阶段为了保持输出的**期望与推理阶段一致**，框架采用了“反向 Dropout”机制，将未丢弃的激活除以保留概率$(1-p)$。
    """
    )
    return


@app.cell(hide_code=True)
def _(attn_weights_v3):
    torch.manual_seed(123)
    dropout = torch.nn.Dropout(0.5)  # dropout rate of 50%
    example = torch.ones(6, 6)  # create a matrix of ones

    print(dropout(example))
    print(dropout(attn_weights_v3))

    mo.show_code()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ### 2.5.3 实现因果自注意力

    > 我们将上述过程封装为一个因果注意力机制的实现类 `CausalAttention`。
    """
    )
    return


@app.cell(hide_code=True)
def _(inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    print(inputs.shape)
    print(
        batch.shape
    )  # 2 inputs with 6 tokens each, and each token has embedding dimension 3

    mo.show_code()
    return (batch,)


@app.cell(hide_code=True)
def _(batch, d_in, d_out):
    class CausalAttention(nn.Module):

        def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout)  # New
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1),
            )  # New

        def forward(self, x):
            b, num_tokens, d_in = x.shape  # New batch dimension b
            # For inputs where `num_tokens` exceeds `context_length`, this will result in errors
            # in the mask creation further below.
            # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
            # do not exceed `context_length` before reaching this forward method.
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.transpose(1, 2)  # Changed transpose
            attn_scores.masked_fill_(
                # New, _ ops are in-place
                self.mask.bool()[:num_tokens, :num_tokens],
                -torch.inf,
            )
            # `:num_tokens` to account for cases
            # where the number of tokens in the batch is smaller
            # than the supported context_size
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5, dim=-1
            )
            attn_weights = self.dropout(attn_weights)  # New

            context_vec = attn_weights @ values
            return context_vec


    torch.manual_seed(123)

    context_length_v2 = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length_v2, 0.0)

    context_vecs = ca(batch)

    print(context_vecs)
    print(context_vecs.shape)

    mo.show_code()
    return (CausalAttention,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :fire: 请注意，dropout 仅在训练期间应用，而不是在推理期间应用

    ![2-22](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-22.svg)
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    ## 2.6 将单头注意力扩展到多头注意力

    ### 2.6.1 堆叠多个单头注意力层

    > 以下是之前实现的自注意力机制的总结（为了简单起见，未显示因果和 `Dropout Mask` ），这也称为单头注意力机制：

    ![2-24](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-24.svg)

    :cloud: 我们只需堆叠多个单头注意力模块即可获得多头注意力模块：

    ![2-25](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-25.svg)

    :fire: 多头注意力机制的核心思想是使用不同的、已学习的线性投影多次（并行）运行注意力机制。这使得模型能够联合关注来自不同位置的不同表征子空间的信息。
    """
    )
    return


@app.cell(hide_code=True)
def _(CausalAttention, batch):
    class MultiHeadAttentionWrapper(nn.Module):

        def __init__(
            self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
        ):
            super().__init__()
            self.heads = nn.ModuleList(
                [
                    CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                    for _ in range(num_heads)
                ]
            )

        def forward(self, x):
            return torch.cat([head(x) for head in self.heads], dim=-1)


    torch.manual_seed(123)

    context_length_v3 = batch.shape[1]  # This is the number of tokens
    _tmp_d_in, _tmp_d_out = 3, 2
    mha = MultiHeadAttentionWrapper(
        _tmp_d_in, _tmp_d_out, context_length_v3, 0.0, num_heads=2
    )

    context_vecs_v2 = mha(batch)

    print("Context Vectors:", context_vecs_v2)
    print("Shape:", context_vecs_v2.shape)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :cloud: 在上面的实现中，嵌入维度为 $4$，因为我们将 $d_{out}=2$ 作为键、查询、值向量以及上下文向量的嵌入维度。由于我们有两个注意力头，因此输出嵌入维度为 $2 \times {2} = 4$

    ### 2.6.2 通过权重分割实现多头注意力

    > 虽然上面是多头注意力的直观且功能齐全的实现（包装了之前的单头注意力 `CausalAttention` 实现），但我们可以编写一个名为 `MultiHeadAttention` 的独立类来实现相同的功能。

    :rocket: 我们不会为这个独立的 `MultiHeadAttention` 类连接单个注意力头，而是创建单个 $W_{query}$、$W_{key}$ 和 $W_{value}$ 权重矩阵，然后将它们拆分为每个注意力头的单独矩阵：
    """
    )
    return


@app.cell(hide_code=True)
def _(batch):
    class MultiHeadAttention(nn.Module):
        def __init__(
            self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
        ):
            super().__init__()
            assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

            self.d_out = d_out
            self.num_heads = num_heads
            self.head_dim = (
                d_out // num_heads
            )  # Reduce the projection dim to match desired output dim

            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(
                d_out, d_out
            )  # Linear layer to combine head outputs
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1),
            )

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            # As in `CausalAttention`, for inputs where `num_tokens` exceeds `context_length`,
            # this will result in errors in the mask creation further below.
            # In practice, this is not a problem since the LLM (chapters 4-7) ensures that inputs
            # do not exceed `context_length` before reaching this forward method.

            keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
            queries = self.W_query(x)
            values = self.W_value(x)

            print(keys.shape, queries.shape, values.shape)
            # We implicitly split the matrix by adding a `num_heads` dimension
            # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            print(keys.shape, queries.shape, values.shape)

            # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            # Compute scaled dot-product attention (aka self-attention) with a causal mask
            attn_scores = queries @ keys.transpose(
                2, 3
            )  # Dot product for each head

            # Original mask truncated to the number of tokens and converted to boolean
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

            # Use the mask to fill attention scores
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5, dim=-1
            )
            attn_weights = self.dropout(attn_weights)

            # Shape: (b, num_tokens, num_heads, head_dim)
            context_vec = (attn_weights @ values).transpose(1, 2)

            # Combine heads, where self.d_out = self.num_heads * self.head_dim
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)  # optional projection

            return context_vec


    torch.manual_seed(123)
    batch_size, context_length_v4, d_in_v2 = batch.shape
    d_out_v2 = 2
    mha_v2 = MultiHeadAttention(
        d_in_v2, d_out_v2, context_length_v4, 0.0, num_heads=2
    )
    context_vecs_v3 = mha_v2(batch)

    print("Context Vectors:", context_vecs_v3)
    print("Shape:", context_vecs_v3.shape)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _():
    mo.md(
        r"""
    :fire: 请注意，上面本质上是 `MultiHeadAttentionWrapper` 的重写版本，效率更高。由于随机权重初始化不同，结果输出看起来有点不同，但两者都是功能齐全的实现，可以在我们将在接下来的章节中实现的 `GPT` 类中使用。

    /// admonition | 关于输出维度的说明

    + 在上面的 $MultiHeadAttention$ 中，我使用了 $d_{out}=2$ 来使用与之前的 $MultiHeadAttentionWrapper$ 类相同的设置

    + 由于连接，$MultiHeadAttentionWrapper$ 返回输出头部维度 $d_{out} \times num_{heads}$（即 $2 \times {2} = 4$）

    + 但是，$MultiHeadAttention$ 类（为了使其更加用户友好）允许我们直接通过 $d_{out}$ 控制输出头部维度；这意味着，如果我们设置 $d_{out} = 2$，则输出头部维度将为 $2$，无论头部数量是多少

    + 事后看来，正如读者指出的那样，使用 $d_{out} = 4$ 的 $MultiHeadAttention$ 可能更直观，这样它产生的输出维度与 $d_{out} = 2$ 的 $MultiHeadAttentionWrapper$ 相同

    ///

    /// admonition | 其他说明

    需要注意的是，我们在上面 `MultiHeadAttention` 类中添加了一个线性投影层 (`self.out_proj`)。这只是一个线性变换，不会改变维度。在 `LLM` 实现中使用这样的投影层是标准惯例，但并非绝对必要（最近的研究表明，可以将其移除而不会影响建模性能；请参阅本章末尾的延伸阅读部分）。

    ///

    ![2-23](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/2-23.svg)

    :rocket: 由于上述实现乍一看可能有点复杂，让我们看看执行 `attn_scores = queries @ keys.transpose(2, 3)` 时会发生什么：
    """
    )
    return


@app.cell(hide_code=True)
def _():
    # (b, num_heads, num_tokens, head_dim) = (1, 2, 3, 4)
    a = torch.tensor(
        [
            [
                [
                    [0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340],
                ],
                [
                    [0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786],
                ],
            ]
        ]
    )

    mo.md(
        f"""

    $a$ ({a.shape}): 

    ```python
    {a}
    ```

    > `a.transpose(2, 3)`

    $a^T$ ({a.transpose(2, 3).shape}): 

    ```python
    {a.transpose(2, 3)}
    ```

    $a \\times a^T$ ({(a @ a.transpose(2, 3)).shape}): 

    ```python
    {a @ a.transpose(2, 3)}
    ```
    """
    )
    return (a,)


@app.cell(hide_code=True)
def _():
    mo.md(
        r""":hammer: 在这种情况下，`PyTorch` 中的矩阵乘法实现将处理 `4` 维输入张量，以便在最后 `2` 个维度（`num_tokens`、`head_dim`）之间进行矩阵乘法，然后对各个头部重复执行。例如，以下成为一种更紧凑的方式来分别计算每个头部的矩阵乘法："""
    )
    return


@app.cell(hide_code=True)
def _(a):
    first_head = a[0, 0, :, :]
    first_res = first_head @ first_head.T

    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T

    mo.md(
        f"""
    第一个头:
    ```python
    {first_res}
    ```

    第二个头:
    ```python
    {second_res}
    ```
    """
    )
    return


if __name__ == "__main__":
    app.run()
