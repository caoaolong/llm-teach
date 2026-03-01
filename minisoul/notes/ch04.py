import marimo

__generated_with = "0.20.1"
app = marimo.App(
    width="medium",
    app_title="LLMs-无标签数据训练GPT",
    css_file="custom.css",
)


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import torch.nn

    print(torch.cuda.is_available())
    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4.无标签数据训练GPT

    > 在本章中，我们将实现用于预训练 LLM 的基本模型评估的训练循环和代码。在本章末尾，我们还将 OpenAI 公开提供的预训练权重加载到我们的模型中。

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-1.svg)

    本章涵盖的主题如下

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-02.svg)
    """)
    return


@app.cell(hide_code=True)
def _():
    import importlib.metadata as _im

    _pkgs = [
        "matplotlib",
        "numpy",
        "tiktoken",
        "torch",
        "tensorflow",  # For OpenAI's pretrained weights
    ]
    for _p in _pkgs:
        try:
            _v = _im.version(_p)
        except _im.PackageNotFoundError:
            _v = "not installed"
        except Exception as _e:
            _v = f"error: {_e.__class__.__name__}"
        print(f"{_p} version: {_v}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.1 评估生成文本模型

    + 本节首先简要回顾一下如何使用上一章的代码初始化 GPT 模型
    + 然后，我们会讨论`LLMs`的基本评估指标
    + 最后，在本节中，我们将这些评估指标应用于训练和验证数据集

    ### 4.1.1 使用GPT生成文本

    > 我们使用上一章的代码初始化 GPT 模型
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    from blocks import GPTModel

    GPT_CONFIG_124M = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 256,  # Shortened context length (orig: 1024)
        "emb_dim": 768,  # Embedding dimension
        "n_heads": 12,  # Number of attention heads
        "n_layers": 12,  # Number of layers
        "drop_rate": 0.1,  # Dropout rate
        "qkv_bias": False,  # Query-key-value bias
    }

    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.eval()
    mo.show_code()
    return GPTModel, GPT_CONFIG_124M, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 我们在上面使用了 `dropout=0.1`，但现在不使用 dropout 来训练 LLM 是比较常见的
    + 目前 LLM 也不在 `nn.Linear` 层中使用偏差向量来表示 $Query$、$Key$ 和 $Value$ 值矩阵（与早期的 GPT 模型不同），这是通过设置 `"qkv_bias": False` 来实现的
    + 我们将上下文长度（`context_length`）减少到仅 `256` 个 token，以减少训练模型所需的计算资源，而原始的 `1.24` 亿个参数 GPT-2 模型使用了 `1024` 个 token
        + 这样，更多的读者就能够在他们的笔记本电脑上跟踪和执行代码示例
        + 但是，请随意将 `context_length` 增加到 `1024` 个令牌（这不需要任何代码更改）
        + 我们稍后还将从预训练权重中加载一个 `context_length` 为 1024 的模型
    + 接下来，我们使用上一章的`generate_text_simple`函数来生成文本
    + 此外，我们定义了两个便捷函数，`text_to_token_ids` 和 `token_ids_to_text`，用于在本章中使用的标记和文本表示之间进行转换

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-3.svg)
    """)
    return


@app.cell(hide_code=True)
def _(GPT_CONFIG_124M, mo, model, torch):
    import tiktoken
    from blocks import generate_text_simple

    # Alternatively:
    # from llms_from_scratch.ch04 import generate_text_simple


    def text_to_token_ids(text, tokenizer):
        encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
        return encoded_tensor


    def token_ids_to_text(token_ids, tokenizer):
        flat = token_ids.squeeze(0)  # remove batch dimension
        return tokenizer.decode(flat.tolist())


    start_context = "Every effort moves you"
    tokenizer = tiktoken.get_encoding("gpt2")

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    mo.show_code()
    return (
        generate_text_simple,
        text_to_token_ids,
        token_ids,
        token_ids_to_text,
        tokenizer,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 正如我们上面所看到的，该模型没有产生好的文本，因为它还没有经过训练
    + 我们如何以数字形式衡量或捕捉“好文本”，以便在训练期间对其进行跟踪
    + 下一小节将介绍用于计算生成输出的损失指标的指标，我们可以使用该指标来衡量训练进度
    + 下一章关于微调 LLM 还将介绍衡量模型质量的其他方法

    ### 4.1.2 交叉熵和困惑度

    > 假设我们有一个输入张量，其中包含 2 个训练示例（行）的`Token ID`，与输入相对应，目标包含我们希望模型生成的所需`Token ID`。

    :fire: 请注意，目标是将输入移动 1 个位置，正如我们在第 2 章实现数据加载器时所解释的那样。
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    inputs = torch.tensor(
        [[16833, 3626, 6100], [40, 1107, 588]]  # ["every effort moves",
    )  #  "I really like"]

    targets = torch.tensor(
        [[3626, 6100, 345], [1107, 588, 11311]]  # [" effort moves you",
    )  #  " really like chocolate"]

    mo.show_code()
    return inputs, targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 将输入提供给模型，我们获得 2 个输入示例的 logits 向量，每个示例包含 3 个Token
    + 每个Token都是一个 50,257 维向量，与词汇表的大小相对应
    + 应用`softmax`函数，我们可以将`logits`张量转换为包含概率分数的相同维度的张量
    """)
    return


@app.cell(hide_code=True)
def _(inputs, mo, model, torch):
    with torch.no_grad():
        logits = model(inputs)

    probas = torch.softmax(
        logits, dim=-1
    )  # Probability of each token in vocabulary
    print(probas.shape)  # Shape: (batch_size, num_tokens, vocab_size)
    mo.show_code()
    return logits, probas


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```python
    probas.shape = [文本数, 位置数, 词表大小]
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    下图使用非常小的词汇量来说明如何将概率分数转换回文本，我们在上一章的末尾讨论过这一点

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-4.svg)

    + 正如上一章所讨论的，我们可以应用 `argmax` 函数将概率分数转换为预测的 token ID
    + 上面的 `softmax` 函数为每个 token 生成一个 50,257 维的向量；`argmax` 函数返回该向量中最高概率分数的位置，即给定 token 的预测 token ID
    + 由于我们有 2 个输入批次，每个批次有 3 个Token，因此我们获得 2 乘 3 的预测Token ID：
    """)
    return


@app.cell(hide_code=True)
def _(mo, probas, torch):
    token_ids_v2 = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids_v2)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 如果我们解码这些Token，我们会发现它们与我们希望模型预测的Token（即目标Token）有很大不同：
    """)
    return


@app.cell(hide_code=True)
def _(mo, targets, token_ids, token_ids_to_text, tokenizer):
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(
        f"Outputs batch 1: {token_ids_to_text(token_ids[0].flatten(), tokenizer)}"
    )

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// admonition | 这是因为模型尚未训练。

    为了训练模型，我们需要知道它与正确预测（目标）的差距有多大。

    ///

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-5.svg)

    :rocket: 目标索引对应的token概率如下：
    """)
    return


@app.cell(hide_code=True)
def _(mo, probas, targets):
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1:", target_probas_1)

    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2:", target_probas_2)

    mo.show_code()
    return target_probas_1, target_probas_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 我们希望最大化所有这些值，使它们的概率接近 1
    + 在数学优化中，最大化概率分数的对数比最大化概率分数本身更容易；这超出了本书的范围，但作者在这里录制了一个包含更多详细信息的讲座：[L8.2 逻辑回归损失函数](https://www.youtube.com/watch?v=GxJe0DZvydM)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 这里我简单来解释一下

    + 求概率分数的对数更简单是因为取对数不会改变最优解，因为对数函数是单调递增的，下面是一个 $y=ln(x)$ 的函数曲线。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def _():
        import numpy as np
        import matplotlib.pyplot as plt

        # 生成自变量（对数函数定义域：x > 0）
        x = np.linspace(0.1, 10, 400)

        # 计算函数值 y = ln(x)
        y = np.log(x)

        # -------- 数值验证单调性 --------
        # 计算相邻差值
        diff = np.diff(y)

        # 判断是否全部为正
        is_increasing = np.all(diff > 0)
        print("ln(x) 是否单调递增：", is_increasing)

        # -------- 绘图 --------
        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label="y = ln(x)")

        # 选取两个点进行可视化比较
        x1, x2 = 1, 3
        y1, y2 = np.log(x1), np.log(x2)

        plt.scatter([x1, x2], [y1, y2])
        plt.vlines(
            [x1, x2], ymin=min(y1, y2) - 0.5, ymax=[y1, y2], linestyles="dashed"
        )
        plt.hlines([y1, y2], xmin=0, xmax=[x1, x2], linestyles="dashed")

        plt.text(x1, y1, f"({x1}, {y1:.2f})")
        plt.text(x2, y2, f"({x2}, {y2:.2f})")

        plt.title("ln(x) Figure")
        plt.xlabel("x")
        plt.ylabel("ln(x)")
        plt.legend()
        plt.grid(True)
        return mo.mpl.interactive(plt.gcf())


    _()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    :cloud: 如果一个概率更大，它的对数也一定更大，所以：

    + 最大化概率

    + 最大化概率的对数

    得到的最优参数完全相同。因此，取对数只是换了一种更好算的形式，不会改变结果。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 接下来，我们计算平均对数概率：
    """)
    return


@app.cell(hide_code=True)
def _(mo, target_probas_1, target_probas_2, torch):
    # Compute logarithm of all token probabilities
    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
    print(log_probas)

    mo.show_code()
    return (log_probas,)


@app.cell(hide_code=True)
def _(log_probas, mo, torch):
    # Calculate the average probability for each token
    avg_log_probas = torch.mean(log_probas)
    print(avg_log_probas)

    mo.show_code()
    return (avg_log_probas,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 目标是通过优化模型权重，使平均对数概率尽可能大
    + 由于对数，最大可能值为 0，而我们目前距离 0 还很远
    + 在深度学习中，标准惯例不是最大化平均对数概率，而是最小化负平均对数概率值；在我们的例子中，我们不是最大化 `-10.7722` 以使其接近 0，而是最小化 `10.7722` 以使其接近 0
    + `-10.7722` 的负值，即 `10.7722`，在深度学习中也称为交叉熵损失
    """)
    return


@app.cell(hide_code=True)
def _(avg_log_probas, mo):
    neg_avg_log_probas = avg_log_probas * -1
    print(neg_avg_log_probas)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: PyTorch 已经实现了一个 cross_entropy 函数来执行前面的步骤

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-6.svg)

    在应用 `cross_entropy` 函数之前，让我们检查一下 `logits` 和目标的形状
    """)
    return


@app.cell(hide_code=True)
def _(logits, mo, targets):
    # Logits have shape (batch_size, num_tokens, vocab_size)
    print("Logits shape:", logits.shape)

    # Targets have shape (batch_size, num_tokens)
    print("Targets shape:", targets.shape)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 对于 PyTorch 中的 cross_entropy 函数，我们希望通过在批量维度上组合这些张量来展平它们：
    """)
    return


@app.cell(hide_code=True)
def _(logits, mo, targets):
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()

    print("Flattened logits:", logits_flat.shape)
    print("Flattened targets:", targets_flat.shape)

    mo.show_code()
    return logits_flat, targets_flat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    /// attention | 请注意

    目标是标记 ID，它也代表我们想要最大化的 `logits` 张量中的索引位置。PyTorch 中的 `cross_entropy` 函数将自动负责在内部对要最大化的 `logits` 中的标记索引应用 `softmax` 和对数概率计算
    """)
    return


@app.cell(hide_code=True)
def _(logits_flat, mo, targets_flat, torch):
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print(loss)

    mo.show_code()
    return (loss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 与交叉熵损失相关的一个概念是LLM的困惑度
    + 困惑度就是交叉熵损失的指数
    """)
    return


@app.cell(hide_code=True)
def _(loss, mo, torch):
    perplexity = torch.exp(loss)
    print(perplexity)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 困惑度通常被认为更容易解释，因为它可以理解为模型在每一步不确定的有效词汇量（在上面的例子中，就是 48,725 个单词或标记）。
    + 换句话说，困惑度衡量了模型预测的概率分布与数据集中单词的实际分布的匹配程度。
    + 与损失类似，困惑度越低，表示模型预测越接近实际分布。

    ### 4.1.3 计算训练和验证集损失

    > 我们使用相对较小的数据集来训练 LLM（实际上只有一个短篇故事，即`the_verdict.txt`）
    >
    > 原因如下：
    >
    > + 您可以在没有合适 GPU 的笔记本电脑上花几分钟运行代码示例
    > + 训练过程相对较快（几分钟而不是几周），这对于教学目的来说很有利
    > + 我们使用来自公共领域的文本，可以将其包含在此 GitHub 存储库中，而不会侵犯任何使用权或增加存储库大小
    >
    > 实际例如，`Llama-2:7B` 需要在 `A100 GPU` 上花费 `184,320` 个 `GPU` 小时来对 2 万亿个 token 进行训练
    >
    > + 在撰写本文时，AWS 上 8xA100 云服务器的每小时成本约为 30 美元
    > + 因此，通过粗略计算，训练模型的费用为 $184,320 \div 8 \times 30 \text{美元} = 690,000 \text{美元}$
    """)
    return


@app.cell(hide_code=True)
def _(mo, tokenizer):
    with open("./data/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    total_tokens = len(tokenizer.encode(raw_text))

    print(f"总字符数: {len(raw_text)}")
    print(f"总Token数: {total_tokens}")
    print(f"{raw_text[:50]}...")

    mo.show_code()
    return raw_text, total_tokens


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 文本包含 5,145 个Token，对于训练 LLM 来说非常短，但同样，它用于教育目的（我们稍后还将加载预训练权重）

    + 接下来，我们将数据集分为训练集和验证集，并使用第 1 章中的数据加载器准备 LLM 训练的批次
    + 为了便于可视化，下图假设 `max_length=6`，但对于训练加载器，我们将 `max_length` 设置为 LLM 支持的上下文长度
    + 为简单起见，下图仅显示输入标记。 由于我们训练 LLM 来预测文本中的下一个单词，因此目标看起来与这些输入相同，只是目标移动了一个位置。

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-7.svg)
    """)
    return


@app.cell(hide_code=True)
def _(GPT_CONFIG_124M, mo, raw_text, torch, total_tokens):
    from blocks import create_dataloader_v1

    # Train/validation ratio
    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]


    torch.manual_seed(123)

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # Sanity check

    if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
        print(
            "Not enough tokens for the training loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "increase the `training_ratio`"
        )

    if total_tokens * (1 - train_ratio) < GPT_CONFIG_124M["context_length"]:
        print(
            "Not enough tokens for the validation loader. "
            "Try to lower the `GPT_CONFIG_124M['context_length']` or "
            "decrease the `training_ratio`"
        )

    mo.show_code()
    return train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 我们使用相对较小的批处理大小来减少计算资源需求，因为数据集本身就很小。 例如，`Llama-2:7B` 的训练批处理大小为 1024。

    :hammer: 检查数据集是否被正确加载
    """)
    return


@app.cell(hide_code=True)
def _(mo, train_loader, val_loader):
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 检查Token大小是否在预期范围内
    """)
    return


@app.cell(hide_code=True)
def _(mo, train_loader, val_loader):
    train_tokens = 0
    for input_batch, target_batch in train_loader:
        train_tokens += input_batch.numel()

    val_tokens = 0
    for input_batch, target_batch in val_loader:
        val_tokens += input_batch.numel()

    print("Training tokens:", train_tokens)
    print("Validation tokens:", val_tokens)
    print("All tokens:", train_tokens + val_tokens)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 接下来，我们实现一个函数来计算给定批次的交叉熵损失，此外，我们实现了第二个函数来计算数据加载器中用户指定批次数量的损失
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.flatten(0, 1), target_batch.flatten()
        )
        return loss


    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches
            # in the data loader
            # if num_batches exceeds the number of batches in the data loader
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
            else:
                break
        return total_loss / num_batches


    mo.show_code()
    return calc_loss_batch, calc_loss_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 如果你的机器配备了支持 CUDA 的 GPU，那么 LLM 将在 GPU 上进行训练，而无需对代码进行任何更改（通过设备设置，我们确保数据加载到与 LLM 模型相同的设备上）
    """)
    return


@app.cell(hide_code=True)
def _(calc_loss_loader, mo, model, torch, train_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(
        device
    )  # no assignment model = model.to(device) necessary for nn.Module classes


    torch.manual_seed(
        123
    )  # For reproducibility due to the shuffling in the data loader

    with torch.no_grad():
        # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)

    mo.show_code()
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-8.svg)

    ## 4.2 训练LLM

    > 在本节中，我们将实现训练 LLM 的代码，专注于一个简单的训练函数

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-9.svg)
    """)
    return


@app.cell(hide_code=True)
def _(
    calc_loss_batch,
    calc_loss_loader,
    generate_text_simple,
    mo,
    text_to_token_ids,
    token_ids_to_text,
    torch,
):
    def train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
        start_context,
        tokenizer,
    ):
        # Initialize lists to track losses and tokens seen
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients
                tokens_seen += input_batch.numel()
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    track_tokens_seen.append(tokens_seen)
                    print(
                        f"Ep {epoch + 1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

            # Print a sample text after each epoch
            generate_and_print_sample(model, tokenizer, device, start_context)

        return train_losses, val_losses, track_tokens_seen


    def evaluate_model(model, train_loader, val_loader, device, eval_iter):
        model.eval()
        with torch.no_grad():
            train_loss = calc_loss_loader(
                train_loader, model, device, num_batches=eval_iter
            )
            val_loss = calc_loss_loader(
                val_loader, model, device, num_batches=eval_iter
            )
        model.train()
        return train_loss, val_loss


    def generate_and_print_sample(model, tokenizer, device, start_context):
        model.eval()
        context_size = model.pos_emb.weight.shape[0]
        encoded = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model,
                idx=encoded,
                max_new_tokens=50,
                context_size=context_size,
            )
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
        model.train()


    mo.show_code()
    return (train_model_simple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 现在，让我们使用上面定义的训练函数来训练 LLM：
    """)
    return


@app.cell
def _(
    device,
    mo,
    model,
    tokenizer,
    torch,
    train_loader,
    train_model_simple,
    val_loader,
):
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context="Every effort moves you",
        tokenizer=tokenizer,
    )

    mo.show_code()
    return num_epochs, optimizer, tokens_seen, train_losses, val_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 请注意，您的计算机上可能会出现略微不同的损失值，如果它们大致相似（训练损失低于 1，验证损失低于 7），则无需担心。

    /// admonition | 提示

    + 细微的差异通常是由于不同的 GPU 硬件和 CUDA 版本，或者较新的 PyTorch 版本中的细微变化造成的；

    + 即使您在 CPU 上运行示例，也可能会观察到细微的差异；造成差异的一个可能原因是 `nn.Dropout` 在各个操作系统上的行为不同，具体取决于 PyTorch 的编译方式。
    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo, num_epochs, tokens_seen, torch, train_losses, val_losses):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator


    def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_losses, label="Training loss")
        ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.legend(loc="upper right")
        ax1.xaxis.set_major_locator(
            MaxNLocator(integer=True)
        )  # only show integer labels on x-axis

        # Create a second x-axis for tokens seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(
            tokens_seen, train_losses, alpha=0
        )  # Invisible plot for aligning ticks
        ax2.set_xlabel("Tokens seen")

        fig.tight_layout()  # Adjust layout to make room
        plt.savefig("loss-plot.pdf")
        return plt.gcf()


    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    mo.mpl.interactive(
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    )
    return (plt,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 从上面的结果可以看出，该模型一开始会生成难以理解的字符串，但到最后，它能够生成语法上或多或少正确的句子
    + 然而，根据训练和验证集的损失，我们可以看到模型开始过度拟合
    + 如果我们检查它在最后写的几段话，我们会发现它们逐字逐句地包含在训练集中——它只是记住了训练数据
    + 稍后，我们将介绍可以在一定程度上减轻这种记忆的解码策略

    ///admonition | 提示

    + 这里出现过度拟合是因为我们的训练集非常小，而且我们对其进行了多次迭代；
    + 这里的 LLM 培训主要用于教育目的；
    + 我们主要想看看该模型能否学会生成连贯的文本；
    + 我们不需要花费数周或数月的时间在大量昂贵的硬件上训练这个模型，而是稍后加载预训练的权重。

    ///

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-10.svg)

    ## 4.3 控制随机性的解码策略
    """)
    return


@app.cell
def _(
    GPT_CONFIG_124M,
    device,
    generate_text_simple,
    mo,
    model,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer,
):
    model.eval()

    token_ids_v3 = generate_text_simple(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=GPT_CONFIG_124M["context_length"],
    )

    print("Output text:\n", token_ids_to_text(token_ids_v3, tokenizer))

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 即使我们多次执行上述 `generate_text_simple` 函数，LLM 也始终会生成相同的输出。

    > 现在我们引入两种概念，即所谓的解码策略，来改进 `generate_text_simple` 函数 `temperature scaling` 和 `top-k` 个采样。这将使模型能够控制生成文本的随机性和多样性。

    ### 4.3.1 温度缩放

    > 之前，我们总是使用 `torch.argmax` 将概率最高的 `token` 采样为下一个 `token`。
    >
    > 为了增加多样性，我们可以使用 `torch.multinomial(probs, num_samples=1)` 对下一个标记进行采样，该采样从概率分布中抽取样本。
    >
    > 这里，每个索引被选中的概率对应于它在输入张量中的概率。

    :cloud: 以下简要回顾一下生成下一个词元的过程，为了便于说明，假设词汇量非常小：
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    vocab = {
        "closer": 0,
        "every": 1,
        "effort": 2,
        "forward": 3,
        "inches": 4,
        "moves": 5,
        "pizza": 6,
        "toward": 7,
        "you": 8,
    }

    inverse_vocab = {v: k for k, v in vocab.items()}

    # Suppose input is "every effort moves you", and the LLM
    # returns the following logits for the next token:
    next_token_logits = torch.tensor(
        [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]
    )

    probas_v2 = torch.softmax(next_token_logits, dim=0)
    next_token_id = torch.argmax(probas_v2).item()

    # The next generated token is then as follows:
    print(inverse_vocab[next_token_id])

    mo.show_code()
    return inverse_vocab, next_token_logits, probas_v2, vocab


@app.cell(hide_code=True)
def _(inverse_vocab, mo, probas_v2, torch):
    torch.manual_seed(123)
    next_token_id_v2 = torch.multinomial(probas_v2, num_samples=1).item()
    print(inverse_vocab[next_token_id_v2])

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 我们不通过 `torch.argmax` 来确定最可能的标记，而是使用 `torch.multinomial(probas, num_samples=1)` 通过从 `softmax` 分布中采样来确定最可能的标记。

    > 为了便于说明，让我们看看使用原始 `softmax` 概率对下一个标记进行 `1000` 次采样会发生什么：
    """)
    return


@app.cell(hide_code=True)
def _(inverse_vocab, mo, probas_v2, torch):
    def print_sampled_tokens(probas):
        torch.manual_seed(123)  # Manual seed for reproducibility
        sample = [
            torch.multinomial(probas, num_samples=1).item() for i in range(1_000)
        ]
        sampled_ids = torch.bincount(torch.tensor(sample), minlength=len(probas))
        for i, freq in enumerate(sampled_ids):
            print(f"{freq} x {inverse_vocab[i]}")


    print_sampled_tokens(probas_v2)

    mo.show_code()
    return (print_sampled_tokens,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 我们可以通过一种称为温度缩放的概念来控制分布和选择过程。
    + `Temperature scaling` 只不过是把 `logits` 除以一个大于 `0` 的数的另一种说法而已。
    + 温度大于 `1` 时，应用 `softmax` 函数后，令牌概率的分布会更加均匀。
    + 温度值小于 `1` 时，应用 `softmax` 函数后会得到更可靠的（更尖锐或更突出的）分布。

    /// attention | 注意
    根据操作系统不同，最终的 `dropout` 输出可能有所不同；您可以在 `PyTorch` 问题跟踪器上阅读更多关于此不一致性的信息。
    ///
    """)
    return


@app.cell
def _(next_token_logits, torch):
    def softmax_with_temperature(logits, temperature):
        scaled_logits = logits / temperature
        return torch.softmax(scaled_logits, dim=0)


    # Temperature values
    temperatures = [1, 0.1, 5]  # Original, higher confidence, and lower confidence

    # Calculate scaled probabilities
    scaled_probas = [
        softmax_with_temperature(next_token_logits, T) for T in temperatures
    ]
    return scaled_probas, temperatures


@app.cell
def _(mo, plt, scaled_probas, temperatures, torch, vocab):
    # Plotting
    x_v2 = torch.arange(len(vocab))
    bar_width = 0.15

    fig, ax = plt.subplots(figsize=(5, 3))
    for i, T in enumerate(temperatures):
        rects = ax.bar(
            x_v2 + i * bar_width,
            scaled_probas[i],
            bar_width,
            label=f"Temperature = {T}",
        )

    ax.set_ylabel("Probability")
    ax.set_xticks(x_v2)
    ax.set_xticklabels(vocab.keys(), rotation=90)
    ax.legend()

    plt.tight_layout()
    plt.savefig("temperature-plot.pdf")
    mo.mpl.interactive(plt.gcf())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 我们可以看到，通过温度 `0.1` 进行重新缩放，可以得到更尖锐的分布，接近 `torch.argmax`，从而几乎总是选择最有可能的单词：
    """)
    return


@app.cell(hide_code=True)
def _(mo, print_sampled_tokens, scaled_probas):
    print_sampled_tokens(scaled_probas[1])
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 通过温度 `5` 重新调整后的概率分布更加均匀：
    """)
    return


@app.cell(hide_code=True)
def _(mo, print_sampled_tokens, scaled_probas):
    print_sampled_tokens(scaled_probas[2])

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    假设 `LLM` 输入 `every effort moves you`，使用上述方法有时会产生无意义的文本，例如 `every effort moves you pizza`，这种情况发生的概率为 `3.2%`（`1000` 次中有 `32` 次）。

    ### 4.3.2 Top-K 采样

    > 为了能够利用更高的温度来增加输出多样性并降低无意义句子的概率，我们可以将采样标记限制为最有可能的前 `k` 个标记：

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-11.svg)

    /// attention | 注意
    为了减少视觉混乱，图中数字仅保留小数点后两位。`Softmax` 行中的值之和应为 `1.0`。
    ///

    在代码中，我们可以按如下方式实现：
    """)
    return


@app.cell(hide_code=True)
def _(mo, next_token_logits, torch):
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)

    print("Top logits:", top_logits)
    print("Top positions:", top_pos)

    mo.show_code()
    return (top_logits,)


@app.cell(hide_code=True)
def _(mo, next_token_logits, top_logits, torch):
    new_logits = torch.where(
        condition=next_token_logits < top_logits[-1],
        input=torch.tensor(float("-inf")),
        other=next_token_logits,
    )

    print(new_logits)

    mo.show_code()
    return (new_logits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ///attention | NOTE:

    以下是前述代码单元的另一种实现方式，效率略高一些：

    ```python
    new_logits = torch.full_like( # create tensor containing -inf values
       next_token_logits, -torch.inf
    )
    new_logits[top_pos] = next_token_logits[top_pos] # copy top k values into the -inf tensor
    ```

    ///
    """)
    return


@app.cell(hide_code=True)
def _(mo, new_logits, torch):
    topk_probas = torch.softmax(new_logits, dim=0)
    print(topk_probas)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4.3.3 修改文本生成功能

    > 前两节介绍了温度 `temperature sampling` 和 `top-k sampling`。让我们运用这两个概念来修改之前的 `generate_text_simple` 函数，创建一个新的 `generate` 函数：
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    def generate(
        model,
        idx,
        max_new_tokens,
        context_size,
        temperature=0.0,
        top_k=None,
        eos_id=None,
    ):
        # For-loop is the same as before: Get logits, and only focus on last time step
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            # New: Filter logits with top_k sampling
            if top_k is not None:
                # Keep only top_k values
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float("-inf")).to(logits.device),
                    logits,
                )

            # New: Apply temperature scaling
            if temperature > 0.0:
                logits = logits / temperature

                # New (not in book): numerical stability tip to get equivalent results on mps device
                # subtract rowwise max before softmax
                logits = logits - logits.max(dim=-1, keepdim=True).values

                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

                # Sample from the distribution
                idx_next = torch.multinomial(
                    probs, num_samples=1
                )  # (batch_size, 1)

            # Otherwise same as before: get idx of the vocab entry with the highest logits value
            else:
                idx_next = torch.argmax(
                    logits, dim=-1, keepdim=True
                )  # (batch_size, 1)

            if (
                idx_next == eos_id
            ):  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
                break

            # Same as before: append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

        return idx


    mo.show_code()
    return (generate,)


@app.cell
def _(
    GPT_CONFIG_124M,
    device,
    generate,
    mo,
    model,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer,
    torch,
):
    torch.manual_seed(123)

    token_ids_v4 = generate(
        model=model,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=15,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=25,
        temperature=1.4,
    )

    print("Output text:\n", token_ids_to_text(token_ids_v4, tokenizer))

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.4 在 PyTorch 中加载和保存模型权重

    > 训练 LLM 的计算成本很高，因此能够保存和加载 LLM 权重至关重要。

    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-12.svg)

    :rocket: 在 `PyTorch` 中，推荐的保存模型权重（即所谓的 `state_dict`）的方法是，通过将 `torch.save` 函数应用于 `.state_dict()` 方法：
    """)
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), "model.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 然后我们可以按如下方式将模型权重加载到新的 GPTModel 模型实例中：
    """)
    return


@app.cell(hide_code=True)
def _(GPTModel, GPT_CONFIG_124M, device, mo, torch):
    model_v2 = GPTModel(GPT_CONFIG_124M)

    model_v2.load_state_dict(
        torch.load("model.pth", map_location=device, weights_only=True)
    )
    model_v2.eval()

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 通常使用自适应优化器（例如 `Adam` 或 `AdamW`）而不是常规的随机梯度下降法 (`SGD`) 来训练 `LLM`。

    + 这些自适应优化器会为每个模型权重存储额外的参数，因此，如果我们计划稍后继续进行预训练，将它们也保存下来是很有意义的：
    """)
    return


@app.cell(hide_code=True)
def _(mo, model, optimizer, torch):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth",
    )

    mo.show_code()
    return


@app.cell
def _(GPTModel, GPT_CONFIG_124M, mo, model, torch):
    checkpoint = torch.load("model_and_optimizer.pth", weights_only=True)

    model_v3 = GPTModel(GPT_CONFIG_124M)
    model_v3.load_state_dict(checkpoint["model_state_dict"])

    optimizer_v2 = torch.optim.AdamW(
        model.parameters(), lr=0.0005, weight_decay=0.1
    )
    optimizer_v2.load_state_dict(checkpoint["optimizer_state_dict"])
    model_v3.train()

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4.5 从 OpenAI 加载预训练权重

    > 之前，我们仅出于教育目的，使用一本篇幅很短的短篇小说集训练了一个小型 `GPT-2` 模型。
    >
    > 幸运的是，我们不必花费数万甚至数十万美元在大型预训练语料库上预训练模型，而是可以加载 `OpenAI` 提供的预训练权重。

    /// admonition | 致谢！

    再次感谢`LLMs-from-scratch`作者`Sebastian Raschka`的慷慨付出

    + [Github](https://github.com/rasbt)

    + [Huggingface](https://huggingface.co/rasbt)

    ![Avatar](https://avatars.githubusercontent.com/u/5618407?v=4)

    ///
    """)
    return


@app.cell
def _():
    from gpt2 import download_and_load_gpt2

    settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
    return params, settings


@app.cell(hide_code=True)
def _(mo, settings):
    print("Settings:", settings)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo, params):
    print("Parameter dictionary keys:", params.keys())
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo, params):
    print(params["wte"])
    print("Token embedding weight tensor dimensions:", params["wte"].shape)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/4-13.svg)
    """)
    return


@app.cell(hide_code=True)
def _(GPTModel, GPT_CONFIG_124M, device, mo):
    # Define model configurations in a dictionary for compactness
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # Copy the base configuration and update with specific model settings
    model_name = "gpt2-small (124M)"  # Example model name
    NEW_CONFIG = GPT_CONFIG_124M.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

    gpt = GPTModel(NEW_CONFIG)
    gpt.to(device)
    gpt.eval()
    mo.show_code()
    return NEW_CONFIG, gpt


@app.cell(hide_code=True)
def _(mo, torch):
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(
                f"Shape mismatch. Left: {left.shape}, Right: {right.shape}"
            )
        return torch.nn.Parameter(torch.tensor(right))


    mo.show_code()
    return (assign,)


@app.cell(hide_code=True)
def _(assign, device, gpt, mo, params):
    import numpy as np


    def load_weights_into_gpt(gpt, params):
        gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params["wpe"])
        gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params["wte"])

        for b in range(len(params["blocks"])):
            q_w, k_w, v_w = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1
            )
            gpt.trf_blocks[b].att.W_query.weight = assign(
                gpt.trf_blocks[b].att.W_query.weight, q_w.T
            )
            gpt.trf_blocks[b].att.W_key.weight = assign(
                gpt.trf_blocks[b].att.W_key.weight, k_w.T
            )
            gpt.trf_blocks[b].att.W_value.weight = assign(
                gpt.trf_blocks[b].att.W_value.weight, v_w.T
            )

            q_b, k_b, v_b = np.split(
                (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1
            )
            gpt.trf_blocks[b].att.W_query.bias = assign(
                gpt.trf_blocks[b].att.W_query.bias, q_b
            )
            gpt.trf_blocks[b].att.W_key.bias = assign(
                gpt.trf_blocks[b].att.W_key.bias, k_b
            )
            gpt.trf_blocks[b].att.W_value.bias = assign(
                gpt.trf_blocks[b].att.W_value.bias, v_b
            )

            gpt.trf_blocks[b].att.out_proj.weight = assign(
                gpt.trf_blocks[b].att.out_proj.weight,
                params["blocks"][b]["attn"]["c_proj"]["w"].T,
            )
            gpt.trf_blocks[b].att.out_proj.bias = assign(
                gpt.trf_blocks[b].att.out_proj.bias,
                params["blocks"][b]["attn"]["c_proj"]["b"],
            )

            gpt.trf_blocks[b].ff.layers[0].weight = assign(
                gpt.trf_blocks[b].ff.layers[0].weight,
                params["blocks"][b]["mlp"]["c_fc"]["w"].T,
            )
            gpt.trf_blocks[b].ff.layers[0].bias = assign(
                gpt.trf_blocks[b].ff.layers[0].bias,
                params["blocks"][b]["mlp"]["c_fc"]["b"],
            )
            gpt.trf_blocks[b].ff.layers[2].weight = assign(
                gpt.trf_blocks[b].ff.layers[2].weight,
                params["blocks"][b]["mlp"]["c_proj"]["w"].T,
            )
            gpt.trf_blocks[b].ff.layers[2].bias = assign(
                gpt.trf_blocks[b].ff.layers[2].bias,
                params["blocks"][b]["mlp"]["c_proj"]["b"],
            )

            gpt.trf_blocks[b].norm1.scale = assign(
                gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"]
            )
            gpt.trf_blocks[b].norm1.shift = assign(
                gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"]
            )
            gpt.trf_blocks[b].norm2.scale = assign(
                gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"]
            )
            gpt.trf_blocks[b].norm2.shift = assign(
                gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"]
            )

        gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
        gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
        gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


    load_weights_into_gpt(gpt, params)
    gpt.to(device)

    mo.show_code()
    return


@app.cell
def _(
    NEW_CONFIG,
    device,
    generate,
    gpt,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer,
    torch,
):
    torch.manual_seed(123)

    token_ids_v5 = generate(
        model=gpt,
        idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
        max_new_tokens=25,
        context_size=NEW_CONFIG["context_length"],
        top_k=50,
        temperature=1.5,
    )

    print("Output text:\n", token_ids_to_text(token_ids_v5, tokenizer))
    return


if __name__ == "__main__":
    app.run()
