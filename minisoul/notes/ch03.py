import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium", app_title="LLMs-实现生成文本的GPT")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    return mo, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # 3.实现生成文本的GPT
    > 在本章中，我们将实现一个类似 GPT 的 LLM 架构；下一章将重点介绍如何训练这个 LLM。

    【图1】

    ## 3.1 编写LLM架构

    我们会看到，LLM 的架构中有很多元素是重复的。

    【图2】

    + 在之前的章节中，我们为了便于理解，使用了较小的嵌入维度。在本章中，我们考虑类似于小型 `GPT-2` 模型的嵌入和模型大小；
    + 我们将专门编写最小 `GPT-2` 模型（`1.24` 亿个参数）的架构，如 Radford 等人在《语言模型是无监督多任务学习者》中所述（请注意，初始报告将其列为 `1.17` 亿个参数，但后来在模型权重存储库中进行了更正）；
    + 第 5 章将展示如何将预训练权重加载到我们的实现中，这将与 3.45 亿、7.62 亿和 15.42 亿个参数的模型大小兼容。

    :rocket: 1.24 亿个参数的 `GPT-2` 模型的配置细节包括：
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-Key-Value bias
    }

    mo.show_code()
    return (GPT_CONFIG_124M,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | :fire: 我们使用简短的变量名来避免后面的长代码行。
    + `vocab_size`: 表示词汇量为 50,257 个单词，由第 1 章中讨论的 BPE 标记器支持
    + `context_length`: 表示模型的最大输入标记数，由第 1 章中介绍的位置嵌入启用“emb_dim”是标记输入的嵌入大小，将每个输入标记转换为 768 维向量
    + `n_heads`: 是第 2 章中实现的多头注意机制中的注意头的数量
    + `n_layers`: 是模型中 Transformer 块的数量，我们将在接下来的章节中实现
    + `drop_rate`: 是 `dropout` 机制的强度；0.1 表示在训练期间丢弃 10% 的隐藏单元以减轻过度拟合
    + `qkv_bias`: 决定多头注意机制中的线性层在计算查询（Q）、键（K）和值（V）张量时是否应包含偏差向量；我们将禁用此选项，这是现代 LLM 中的标准做法；但是，我们将在第 4 章中将 OpenAI 预训练的 GPT-2 权重加载到我们的重新实现中时再次讨论这一点
    ///

    【图3】
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, nn, torch):
    class DummyGPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            # Use a placeholder for TransformerBlock
            self.trf_blocks = nn.Sequential(
                *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )

            # Use a placeholder for LayerNorm
            self.final_norm = DummyLayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
            )

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits


    class DummyTransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            # A simple placeholder

        def forward(self, x):
            # This block does nothing and just returns its input.
            return x


    class DummyLayerNorm(nn.Module):
        def __init__(self, normalized_shape, eps=1e-5):
            super().__init__()
            # The parameters here are just to mimic the LayerNorm interface.

        def forward(self, x):
            # This layer does nothing and just returns its input.
            return x


    mo.show_code()
    return (DummyGPTModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""【图4】""")
    return


@app.cell(hide_code=True)
def _(mo, torch):
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    batch = []

    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"

    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)

    print(batch)
    mo.show_code()
    return batch, tokenizer


@app.cell(hide_code=True)
def _(DummyGPTModel, GPT_CONFIG_124M, batch, mo, torch):
    torch.manual_seed(123)
    model = DummyGPTModel(GPT_CONFIG_124M)

    logits = model(batch)
    print("Output shape:", logits.shape)
    print(logits)
    mo.show_code()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3.2 归一化激活

    > + 层归一化，也称为 LayerNorm (Ba et al. 2016)，将神经网络层的激活集中在平均值 0 附近，并将其方差归一化为 1
    > + 这可以稳定训练并能够更快地收敛到有效权重
    > + 层归一化应用于 Transformer 模块中的多头注意力模块之前和之后，我们将在稍后实现；它也应用于最终输出层之前

    【图5】

    :hammer: 让我们通过将一个小的输入样本传递到一个简单的神经网络层来看一下层规范化是如何工作的：
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, nn, torch):
    torch.manual_seed(123)

    # create 2 training examples with 5 dimensions (features) each
    batch_example = torch.randn(2, 5)

    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    out = layer(batch_example)
    print(out)
    mo.show_code()
    return batch_example, out


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":hammer: 让我们计算上述两个输入的平均值和方差：""")
    return


@app.cell(hide_code=True)
def _(mo, out):
    mean = out.mean(dim=-1, keepdim=True)
    var = out.var(dim=-1, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)
    mo.show_code()
    return mean, var


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    :fire: 规范化独立应用于两个输入（行）中的每一个；使用 `dim=-1` 将计算应用于最后一个维度（在本例中为特征维度），而不是行维度。

    【图6】

    :fire: 减去平均值并除以方差（标准差）的平方根，使输入在列（特征）维度上居中，平均值为 0，方差为 1：
    """
    )
    return


@app.cell(hide_code=True)
def _(mean, mo, out, torch, var):
    out_norm = (out - mean) / torch.sqrt(var)
    print("Normalized layer outputs:\n", out_norm)

    mean_v2 = out_norm.mean(dim=-1, keepdim=True)
    var_v2 = out_norm.var(dim=-1, keepdim=True)
    print("Mean:\n", mean_v2)
    print("Variance:\n", var_v2)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":hammer: 每个输入以 0 为中心，单位方差为 1；为了提高可读性，我们可以禁用 PyTorch 的科学计数法：""")
    return


@app.cell(hide_code=True)
def _(mean, mo, torch, var):
    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":hammer: 上面，我们对每个输入的特征进行了归一化。现在，使用相同的思路，我们可以实现一个 LayerNorm 类：""")
    return


@app.cell(hide_code=True)
def _(mo, nn, torch):
    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift


    mo.show_code()
    return (LayerNorm,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | **⚙️ scale（缩放）和 shift（平移）**

    > + **归一化（Normalization）中的可训练参数 `scale` 与 `shift` **是在 **Layer Normalization / Batch Normalization** 中常见的 $\gamma$（scale）和 $\beta$（shift）两个参数。
    > + 归一化去除了原始分布的偏移和尺度，但有时这些信息对模型有用，所以通过 trainable 的 scale 和 shift 让模型自己决定要不要恢复这些分布特征。

    :radio_button: 在深度学习中，归一化层通常会对输入做如下操作：

    $$
    \hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}
    $$

    其中：

    > * ( $\mu$ )：均值
    > * ( $\sigma^2$ )：方差
    > * ( $\epsilon$ )：一个很小的数，防止除零


    :radio_button: 为了让模型在归一化后仍能保持表达能力，加入了两个**可训练参数**：

    $$
    y = \text{scale} \times \hat{x} + \text{shift}
    $$

    * **scale（γ）**：缩放参数，初始值为 1。
    * **shift（β）**：平移参数，初始值为 0。

    :radio_button: 虽然它们最初不改变输出分布（乘 1 加 0），但在训练过程中会被自动学习，以便：

    * 调整不同特征的分布；
    * 让模型在归一化后仍能恢复适合任务的特征范围。


    `eps` 是一个很小的常数（例如 1e-5），加在方差里：

    $$
    \frac{1}{\sqrt{\sigma^2 + \epsilon}}
    $$

    它的作用是**避免除以 0** 导致的数值不稳定性。

    ///

    :rocket: 除了减均值除方差外，还加入`scale` 和 `shift`两个可训练参数已便模型在训练时自动学习最合适的缩放和平移，从而提升性能与稳定性。

    ---

    /// details | **有偏方差（Biased variance）**

    在方差计算时，当设置 `unbiased=False`，会用到公式 

    $$
    \boldsymbol{\frac{\sum_{i}(x_{i}-\bar{x})^2}{n}}
    $$

    来计算方差。这里的 $n$ 代表“样本量”，在大模型领域也可以理解成“特征数量”或者“列数”。  此外这个公式没有包含「贝塞尔校正（Bessel’s correction）」——要知道，如果用贝塞尔校正，方差公式的分母会是 $n - 1$。即：

    $$
    \frac{\sum_{i}(x_{i}-\bar{x})^2}{n - 1}
    $$

    :fire: `GPT-2` 在“归一化层（normalization layers）”里，训练时用的就是**有偏方差**。

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(LayerNorm, batch_example, mean, mo, var):
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)

    mean_v3 = out_ln.mean(dim=-1, keepdim=True)
    var_v3 = out_ln.var(dim=-1, unbiased=False, keepdim=True)

    print("Mean:\n", mean)
    print("Variance:\n", var)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    【图7】

    ## 3.3 GELU激活实现FNN

    > 本节内容来介绍一下GELU激活函数。

    + `ReLU`激活函数：在深度学习中因简单且在多种神经网络架构中有效而被广泛使用。
    + LLMs中的其他激活函数：除了传统的`ReLU` ，LLMs还使用多种其他类型的激活函数，如`GELU`（Gaussian Error Linear Unit，高斯误差线性单元）和`SwiGLU`（Swish - Gated Linear Unit，Sigmoid - 门控线性单元）。`GELU`和`SwiGLU`是更复杂、平滑的激活函数，分别结合了高斯和`Sigmoid` - 门控线性单元，相比ReLU简单的分段线性函数，能为深度学习模型提供更好的性能。

    `GELU`的实现：`GELU`有几种实现方式，其精确版本定义为：

    $$
    GELU(x)=x\cdot\varPhi(x)
    $$

    ，其中$\varPhi(x)$是标准高斯分布的累积分布函数。

    + 实际中常见的是一种计算成本较低的近似实现：

    $$
    GELU(x)\approx0.5\cdot x\cdot(1 + \tanh[\sqrt{\frac{2}{\pi}}\cdot(x + 0.044715\cdot x^{3})])
    $$
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, nn, torch):
    class GELU(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return (
                0.5
                * x
                * (
                    1
                    + torch.tanh(
                        torch.sqrt(torch.tensor(2.0 / torch.pi))
                        * (x + 0.044715 * torch.pow(x, 3))
                    )
                )
            )


    mo.show_code()
    return (GELU,)


@app.cell(hide_code=True)
def _(GELU, nn, torch):
    import matplotlib.pyplot as plt

    gelu, relu = GELU(), nn.ReLU()

    # Some sample data
    x = torch.linspace(-3, 3, 100)
    y_gelu, y_relu = gelu(x), relu(x)

    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)

    plt.tight_layout()
    plt.gca()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    + 可以看出，ReLU 是一个分段线性函数，如果输入为正，则直接输出；否则输出零。

    + `GELU` 是一个平滑的非线性函数，它近似于 `ReLU`，但对于负值具有非零梯度（除大约 `-0.75` 处）

    :hammer: 接下来，让我们实现小型神经网络模块 `FeedForward`，稍后我们将在 LLM 的转换器模块中使用它：
    """
    )
    return


@app.cell(hide_code=True)
def _(GELU, GPT_CONFIG_124M, mo, nn):
    class FeedForward(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
                GELU(),
                nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
            )

        def forward(self, x):
            return self.layers(x)


    print(GPT_CONFIG_124M["emb_dim"])
    mo.show_code()
    return (FeedForward,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""【图8】""")
    return


@app.cell(hide_code=True)
def _(FeedForward, GPT_CONFIG_124M, mo, torch):
    ffn = FeedForward(GPT_CONFIG_124M)

    # input shape: [batch_size, num_token, emb_size]
    x_v2 = torch.rand(2, 3, 768)
    out_v2 = ffn(x_v2)
    print(out_v2.shape)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    【图9】

    【图10】
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3.4 添加快捷连接

    > 接下来，我们来讨论一下快捷连接（也称为跳过连接或残差连接）背后的概念

    最初，在计算机视觉的深度网络（残差网络）中提出了快捷连接，以缓解梯度消失问题，而快捷连接为梯度流经网络创建了另一条更短的路径，这是通过将一层的输出添加到后一层的输出来实现的，通常会跳过中间的一个或多个层。

    让我们用一个小示例网络来说明这个想法：

    【图11】

    :hammer: 在代码中，它看起来像这样：
    """
    )
    return


@app.cell(hide_code=True)
def _(GELU, mo, nn, torch):
    class ExampleDeepNeuralNetwork(nn.Module):
        def __init__(self, layer_sizes, use_shortcut):
            super().__init__()
            self.use_shortcut = use_shortcut
            self.layers = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()
                    ),
                    nn.Sequential(
                        nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()
                    ),
                    nn.Sequential(
                        nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()
                    ),
                    nn.Sequential(
                        nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()
                    ),
                    nn.Sequential(
                        nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()
                    ),
                ]
            )

        def forward(self, x):
            for layer in self.layers:
                # Compute the output of the current layer
                layer_output = layer(x)
                # Check if shortcut can be applied
                if self.use_shortcut and x.shape == layer_output.shape:
                    x = x + layer_output
                else:
                    x = layer_output
            return x


    def print_gradients(model, x):
        # Forward pass
        output = model(x)
        target = torch.tensor([[0.0]])

        # Calculate loss based on how close the target
        # and output are
        loss = nn.MSELoss()
        loss = loss(output, target)

        # Backward pass to calculate the gradients
        loss.backward()

        for name, param in model.named_parameters():
            if "weight" in name:
                # Print the mean absolute gradient of the weights
                print(
                    f"{name} has gradient mean of {param.grad.abs().mean().item()}"
                )


    mo.show_code()
    return ExampleDeepNeuralNetwork, print_gradients


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":hammer: 让我们首先打印没有快捷连接的渐变值：""")
    return


@app.cell(hide_code=True)
def _(ExampleDeepNeuralNetwork, mo, print_gradients, torch):
    layer_sizes = [3, 3, 3, 3, 3, 1]

    sample_input = torch.tensor([[1.0, 0.0, -1.0]])

    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    print_gradients(model_without_shortcut, sample_input)
    mo.show_code()
    return layer_sizes, sample_input


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":hammer: 接下来，让我们使用快捷连接打印渐变值：""")
    return


@app.cell(hide_code=True)
def _(
    ExampleDeepNeuralNetwork,
    layer_sizes,
    mo,
    print_gradients,
    sample_input,
    torch,
):
    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)

    mo.show_code()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    :fire: 从上面的输出可以看出，快捷连接可以防止梯度在较早的层（朝向 layer.0）中消失。 接下来，我们将在实现 Transformer 模块时使用快捷连接的概念。

    ## 3.5 连接注意力层和线性层

    > 在本节中，我们将前面的概念组合成一个所谓的`Transform Block`


    + Transformer 模块将上一章中的因果多头注意力模块与线性层（我们在前面部分实现的前馈神经网络）相结合。

    + 此外，Transformer 模块还使用了 dropout 和快捷连接
    """
    )
    return


@app.cell(hide_code=True)
def _(FeedForward, LayerNorm, mo, nn):
    # If the `previous_chapters.py` file is not available locally,
    # you can import it from the `llms-from-scratch` PyPI package.
    # For details, see: https://github.com/rasbt/LLMs-from-scratch/tree/main/pkg
    # E.g.,
    # from llms_from_scratch.ch03 import MultiHeadAttention

    from blocks import MultiHeadAttention


    class TransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"],
            )
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

        def forward(self, x):
            # Shortcut connection for attention block
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            # Shortcut connection for feed forward block
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            return x


    mo.show_code()
    return (TransformerBlock,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    【图12】

    + 假设我们有 2 个输入样本，每个样本有 6 个标记，其中每个标记是一个 768 维嵌入向量；然后这个 Transformer 块应用自注意力机制，然后是线性层，以产生类似大小的输出
    + 你可以将输出视为我们在上一章讨论过的上下文向量的增强版本
    """
    )
    return


@app.cell(hide_code=True)
def _(GPT_CONFIG_124M, TransformerBlock, mo, torch):
    torch.manual_seed(123)

    x_v3 = torch.rand(2, 4, 768)  # Shape: [batch_size, num_tokens, emb_dim]
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x_v3)

    print("Input shape:", x_v3.shape)
    print("Output shape:", output.shape)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    【图13】

    ## 3.6 编写 GPT 模型

    :tada: 我们快完成了：现在让我们将 Transformer 模块插入到我们在本章开头编码的架构中，这样我们就可以获得一个可用的 GPT 架构

    :fire: 请注意，Transformer 块重复多次；对于最小的 124M GPT-2 模型，我们重复了 12 次：

    【图14】

    > 相应的代码实现，其中`cfg["n_layers"] = 12`：
    """
    )
    return


@app.cell(hide_code=True)
def _(LayerNorm, TransformerBlock, mo, nn, torch):
    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
            )

            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
            )

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits


    mo.show_code()
    return (GPTModel,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":rocket: 使用 124M 参数模型的配置，我们现在可以使用随机初始权重实例化此 GPT 模型，如下所示：""")
    return


@app.cell(hide_code=True)
def _(GPTModel, GPT_CONFIG_124M, batch, mo, model, torch):
    torch.manual_seed(123)
    model_v2 = GPTModel(GPT_CONFIG_124M)

    out_v3 = model(batch)
    print("Input batch:\n", batch)
    print("\nOutput shape:", out_v3.shape)
    print(out_v3)

    mo.show_code()
    return (model_v2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":hammer: 我们将在下一章训练这个模型。不过，需要简单说明一下它的大小：我们之前称它是一个 124M 参数的模型；我们可以用以下方法再次确认这个数字：""")
    return


@app.cell(hide_code=True)
def _(mo, model_v2):
    total_params = sum(p.numel() for p in model_v2.parameters())
    print(f"Total number of parameters: {total_params:,}")

    mo.show_code()
    return (total_params,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// admonition | 参数总数：163,009,536

    + 如上所示，该模型有 163M 个参数，而不是 124M 个；为什么？

    + 在原始 GPT-2 论文中，研究人员应用了权重绑定，这意味着他们重用了 token 嵌入层 (tok_emb) 作为输出层，也就是设置 `self.out_head.weight = self.tok_emb.weight`

    + token 嵌入层将 50,257 维的独热编码输入 token 投影到 768 维的嵌入表示

    + 输出层将 768 维嵌入投影回 50,257 维的表示，以便我们可以将它们转换回单词（下一节将详细介绍）

    + 因此，根据权重矩阵的形状，我们可以看到嵌入层和输出层具有相同数量的权重参数

    + 不过，关于它的大小，需要简单说明一下：我们之前称它为 124M 参数模型；我们可以按如下方式再次确认这个数字：
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, model):
    print("Token embedding layer shape:", model.tok_emb.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    + 在原始 GPT-2 论文中，研究人员将 token 嵌入矩阵重新用作输出矩阵。 
    + 相应地，如果我们减去输出层的参数数量，我们将得到一个 124M 参数的模型：
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, model, total_params):
    total_params_gpt2 = total_params - sum(
        p.numel() for p in model.out_head.parameters()
    )
    print(
        f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}"
    )

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    + 在实践中，我发现不使用权重绑定更容易训练模型，这就是我们在这里没有实现它的原因。

    + 不过，我们稍后会在第五章加载预训练权重时重新讨论并应用这个权重绑定的想法。

    + 最后，我们可以按如下方式计算模型的内存需求，这可以作为一个有用的参考点：
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, total_params):
    # Calculate the total size in bytes (assuming float32, 4 bytes per parameter)
    total_size_bytes = total_params * 4

    # Convert to megabytes
    total_size_mb = total_size_bytes / (1024 * 1024)

    print(f"Total size of the model: {total_size_mb:.2f} MB")

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    :hammer: 练习：你也可以尝试以下其他配置，这些配置在 GPT-2 论文中也有引用。

    + GPT2-small（我们已经实现的 124M 配置）：

        + “emb_dim” = 768
        + “n_layers” = 12
        + “n_heads” = 12


    + GPT2-medium：

        + “emb_dim” = 1024
        + “n_layers” = 24
        + “n_heads” = 16

    + GPT2-large：

        + “emb_dim” = 1280
        + “n_layers” = 36
        + “n_heads” = 20

    + GPT2-XL：

        + “emb_dim” = 1600
        + “n_layers” = 48
        + “n_heads” = 25

    ## 3.7 生成文本

    > 使用我们实现的`GPT`模型每次生成一个单词

    【图15】

    + 以下 `generate_text_simple` 函数实现了贪婪解码，这是一种简单快速的文本生成方法。

    + 在贪婪解码中，模型在每一步都会选择概率最高的单词（或 token）作为下一个输出（最高的 logit 对应最高的概率，因此从技术上讲，我们甚至不需要明确计算 softmax 函数）。

    + 在下一章中，我们将实现一个更高级的 `generate_text` 函数。

    + 下图描述了 GPT 模型如何在给定输入上下文的情况下生成下一个单词 token

    【图16】
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, torch):
    def generate_text_simple(model, idx, max_new_tokens, context_size):
        # idx is (batch, n_tokens) array of indices in the current context
        for _ in range(max_new_tokens):

            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]

            # Get the predictions
            with torch.no_grad():
                logits = model(idx_cond)

            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]

            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

        return idx


    mo.show_code()
    return (generate_text_simple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    :rocket: 上面的`generate_text_simple`实现了一个迭代过程，每次创建一个token

    【图17】

    :hammer: 让我们准备一个输入示例：
    """
    )
    return


@app.cell(hide_code=True)
def _(mo, tokenizer, torch):
    start_context = "Hello, I am"

    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)

    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)

    mo.show_code()
    return (encoded_tensor,)


@app.cell(hide_code=True)
def _(GPT_CONFIG_124M, encoded_tensor, generate_text_simple, mo, model):
    model.eval()  # disable dropout

    out_v4 = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Output:", out_v4)
    print("Output length:", len(out_v4[0]))
    mo.show_code()
    return (out_v4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":hammer: 删除批次维度并转换回文本：""")
    return


@app.cell
def _(mo, out_v4, tokenizer):
    decoded_text = tokenizer.decode(out_v4.squeeze(0).tolist())
    print(decoded_text)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r""":fire: 请注意，该模型尚未训练；因此上面的输出文本是随机的。 我们将在下一章训练该模型。""")
    return


if __name__ == "__main__":
    app.run()
