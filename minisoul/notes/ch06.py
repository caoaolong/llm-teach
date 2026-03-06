import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium", app_title="LLMs-指令微调", css_file="custom.css")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from importlib.metadata import version

    pkgs = [
        "numpy",  # PyTorch & TensorFlow dependency
        "matplotlib",  # Plotting library
        "tiktoken",  # Tokenizer
        "torch",  # Deep learning library
        "tqdm",  # Progress bar
        "tensorflow",  # For OpenAI's pretrained weights
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. 指令微调模型

    【图1】

    ## 6.1 指令微调简介

    + 在第4章中，我们看到预训练语言学习模型（LLM）的过程是让它一次学习生成一个词。
    + 因此，预训练的LLM擅长文本补全，但不擅长遵循指令。
    + 本章我们将训练LLM更好地遵循指令。

    【图2】

    + 本章涵盖的主题总结如下图所示。

    【图3】

    ## 6.2 准备用于指令微调的数据集
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import json
    import os
    import requests


    def download_and_load_file(file_path, url):
        if not os.path.exists(file_path):
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            text_data = response.text
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_data)

        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data


    file_path = "instruction-data.json"
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )

    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    mo.show_code()
    return data, json, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 从上述 JSON 文件加载的数据列表中的每个项目都是一个字典，形式如下
    """)
    return


@app.cell(hide_code=True)
def _(data):
    data[50]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 请注意，`input` 字段可以为空：
    """)
    return


@app.cell(hide_code=True)
def _(data):
    data[999]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 指令微调通常被称为“监督式指令微调”，因为它涉及在数据集上训练模型，而数据集的输入输出对是明确提供的。
    + LLM 的输入格式（即提示此模板）有多种。
        + [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html)
        + [Phi-3](https://arxiv.org/abs/2404.14219)

    【图4】

    + 本章采用 `Alpaca` 风格的提示符格式，这是最初用于指令微调的提示符模板。
    + 下面，我们将格式化传递给 LLM 的输入。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    def format_input(entry):
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text


    mo.show_code()
    return (format_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 带有输入字段的格式化响应如下所示
    """)
    return


@app.cell(hide_code=True)
def _(data, format_input, mo):
    model_input = format_input(data[50])
    desired_response = f"\n\n### Response:\n{data[50]['output']}"

    print(model_input + desired_response)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 以下是格式化后的响应，不包含输入字段。
    """)
    return


@app.cell(hide_code=True)
def _(data, format_input, mo):
    model_input_v2 = format_input(data[999])
    desired_response_v2 = f"\n\n### Response:\n{data[999]['output']}"

    print(model_input_v2 + desired_response_v2)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 最后，在下一节准备 PyTorch 数据加载器之前，我们将数据集划分为训练集、验证集和测试集。
    """)
    return


@app.cell(hide_code=True)
def _(data, mo):
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = (
        len(data) - train_portion - test_portion
    )  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion : train_portion + test_portion]
    val_data = data[train_portion + test_portion :]
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    mo.show_code()
    return test_data, train_data, val_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.3 将数据整理成训练批次

    【图5】

    + 我们分几个步骤处理这个数据集批处理问题，如下图所示。

    【图6】

    + 首先，我们实现一个 `InstructionDataset` 类，该类预先对数据集中的所有输入进行标记化，类似于上一章中的 `SpamDataset`。

    【图7】
    """)
    return


@app.cell(hide_code=True)
def _(format_input, mo):
    import torch
    from torch.utils.data import Dataset


    class InstructionDataset(Dataset):
        def __init__(self, data, tokenizer):
            self.data = data

            # Pre-tokenize texts
            self.encoded_texts = []
            for entry in data:
                instruction_plus_input = format_input(entry)
                response_text = f"\n\n### Response:\n{entry['output']}"
                full_text = instruction_plus_input + response_text
                self.encoded_texts.append(tokenizer.encode(full_text))

        def __getitem__(self, index):
            return self.encoded_texts[index]

        def __len__(self):
            return len(self.data)


    mo.show_code()
    return InstructionDataset, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 与上一章类似，我们希望批量收集多个训练样本以加速训练；这需要将所有输入填充到相似的长度。
    + 同样与上一章类似，我们使用 `<|endoftext|>` 标记作为填充标记。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")

    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    mo.show_code()
    return (tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 在第5章中，我们将数据集中的所有样本填充到相同长度。
    + 在这里，我们采用更复杂的方法，开发一个自定义的“整理”函数，并将其传递给数据加载器。
    + 这个自定义的整理函数会将每个批次中的训练样本填充到相同长度（但不同批次的样本长度可以不同）。

    【图8】
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    def custom_collate_draft_1(batch, pad_token_id=50256, device="cpu"):
        # Find the longest sequence in the batch
        # and increase the max length by +1, which will add one extra
        # padding token below
        batch_max_length = max(len(item) + 1 for item in batch)

        # Pad and prepare inputs
        inputs_lst = []

        for item in batch:
            new_item = item.copy()
            # Add an <|endoftext|> token
            new_item += [pad_token_id]
            # Pad sequences to batch_max_length
            padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
            # Via padded[:-1], we remove the extra padded token
            # that has been added via the +1 setting in batch_max_length
            # (the extra padding token will be relevant in later codes)
            inputs = torch.tensor(padded[:-1])
            inputs_lst.append(inputs)

        # Convert list of inputs to tensor and transfer to target device
        inputs_tensor = torch.stack(inputs_lst).to(device)
        return inputs_tensor


    mo.show_code()
    return (custom_collate_draft_1,)


@app.cell(hide_code=True)
def _(custom_collate_draft_1, mo):
    inputs_1 = [0, 1, 2, 3, 4]
    inputs_2 = [5, 6]
    inputs_3 = [7, 8, 9]

    batch = (inputs_1, inputs_2, inputs_3)

    print(custom_collate_draft_1(batch))
    mo.show_code()
    return (batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    【图9】

    + 上面我们只返回了 LLM 的输入；然而，LLM 训练还需要目标值。
    + 与 LLM 预训练类似，目标值是将输入值向右平移一位，这样 LLM 就能学习预测下一个词元。

    【图10】
    """)
    return


@app.cell(hide_code=True)
def _(batch, mo, torch):
    def custom_collate_draft_2(batch, pad_token_id=50256, device="cpu"):
        # Find the longest sequence in the batch
        batch_max_length = max(len(item) + 1 for item in batch)

        # Pad and prepare inputs
        inputs_lst, targets_lst = [], []

        for item in batch:
            new_item = item.copy()
            # Add an <|endoftext|> token
            new_item += [pad_token_id]
            # Pad sequences to max_length
            padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
            inputs = torch.tensor(
                padded[:-1]
            )  # Truncate the last token for inputs
            targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets
            inputs_lst.append(inputs)
            targets_lst.append(targets)

        # Convert list of inputs to tensor and transfer to target device
        inputs_tensor = torch.stack(inputs_lst).to(device)
        targets_tensor = torch.stack(targets_lst).to(device)
        return inputs_tensor, targets_tensor


    inputs, targets = custom_collate_draft_2(batch)
    print(inputs)
    print(targets)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 接下来，我们引入 `ignore_index` 值，将所有填充标记 ID 替换为新值；此 `ignore_index` 的目的是让我们可以在损失函数中忽略填充值（稍后会详细介绍）。

    【图11】

    + 具体来说，这意味着我们将对应于 50256 的令牌 ID 替换为 -100，如下所示。

    【图12】

    > 此外，我们还引入了 `allowed_max_length` 参数，以便在需要限制样本长度时使用；如果您计划使用长度超过 GPT-2 模型支持的 1024 个 token 上下文大小的数据集，这将非常有用。
    """)
    return


@app.cell(hide_code=True)
def _(batch, mo, torch):
    def custom_collate_fn(
        batch,
        pad_token_id=50256,
        ignore_index=-100,
        allowed_max_length=None,
        device="cpu",
    ):
        # Find the longest sequence in the batch
        batch_max_length = max(len(item) + 1 for item in batch)

        # Pad and prepare inputs and targets
        inputs_lst, targets_lst = [], []

        for item in batch:
            new_item = item.copy()
            # Add an <|endoftext|> token
            new_item += [pad_token_id]
            # Pad sequences to max_length
            padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
            inputs = torch.tensor(
                padded[:-1]
            )  # Truncate the last token for inputs
            targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

            # New: Replace all but the first padding tokens in targets by ignore_index
            mask = targets == pad_token_id
            indices = torch.nonzero(mask).squeeze()
            if indices.numel() > 1:
                targets[indices[1:]] = ignore_index

            # New: Optionally truncate to maximum sequence length
            if allowed_max_length is not None:
                inputs = inputs[:allowed_max_length]
                targets = targets[:allowed_max_length]

            inputs_lst.append(inputs)
            targets_lst.append(targets)

        # Convert list of inputs and targets to tensors and transfer to target device
        inputs_tensor = torch.stack(inputs_lst).to(device)
        targets_tensor = torch.stack(targets_lst).to(device)

        return inputs_tensor, targets_tensor


    inputs_v3, targets_v3 = custom_collate_fn(batch)
    print(inputs_v3)
    print(targets_v3)
    mo.show_code()
    return (custom_collate_fn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 让我们看看用 -100 替换后会发生什么。
    + 为了便于说明，假设我们有一个包含两个类别标签（0 和 1）的小型分类任务，类似于第5章的内容。
    + 如果我们有以下 logits 值（模型最后一层的输出），我们可以计算以下损失。
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    logits_1 = torch.tensor(
        [
            [-1.0, 1.0],  # 1st training example
            [-0.5, 1.5],
        ]  # 2nd training example
    )
    targets_1 = torch.tensor([0, 1])


    loss_1 = torch.nn.functional.cross_entropy(logits_1, targets_1)
    print(loss_1)
    mo.show_code()
    return (loss_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 现在，正如预期的那样，增加一个训练样本会对损失函数产生影响。
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    logits_2 = torch.tensor(
        [[-1.0, 1.0], [-0.5, 1.5], [-0.5, 1.5]]  # New 3rd training example
    )
    targets_2 = torch.tensor([0, 1, 1])

    loss_2 = torch.nn.functional.cross_entropy(logits_2, targets_2)
    print(loss_2)
    mo.show_code()
    return (logits_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 让我们看看如果将其中一个示例的类标签替换为 -100 会发生什么。
    """)
    return


@app.cell(hide_code=True)
def _(logits_2, loss_1, mo, torch):
    targets_3 = torch.tensor([0, 1, -100])

    loss_3 = torch.nn.functional.cross_entropy(logits_2, targets_3)
    print(loss_3)
    print("loss_1 == loss_3:", loss_1 == loss_3)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 正如我们所见，这 3 个训练样本的损失与我们从 2 个训练样本计算出的损失相同，这意味着交叉熵损失函数忽略了标签为 -100 的训练样本。
    + 默认情况下，PyTorch 的 `cross_entropy(..., ignore_index=-100)` 设置会忽略标签为 -100 的样本。
    + 使用 `ignore_index -100`，我们可以忽略批次中用于填充训练样本长度的额外文本结束符（填充标记）。
    + 但是，我们不希望忽略第一个文本结束符（填充标记）（50256），因为它可以帮助 LLM 判断响应何时完成。
    + 在实践中，通常也会屏蔽与指令对应的目标标记 ID，如下图所示（建议读者在完成本章后进行练习）。

    【图13】

    ## 6.4 创建指令数据加载器

    > 在本节中，我们使用 `InstructionDataset` 类和 `custom_collat_e_fn` 函数来实例化训练、验证和测试数据加载器。

    【图14】

    + 之前自定义的 `custom_collat_e_fn` 函数的另一个改进之处在于，我们现在直接将数据移动到目标设备（例如 GPU），而不是在主训练循环中进行操作。
    + 这提高了效率，因为当我们将 `custom_collat_e_fn` 作为数据加载器的一部分时，它可以作为后台进程执行。
    + 我们使用 Python 标准库 `functools` 中的 `partial` 函数，创建一个新函数，并将原始函数的设备参数预先填充。
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Use PyTorch 2.9 or newer for stable mps results
        major, minor = map(int, torch.__version__.split(".")[:2])
        if (major, minor) >= (2, 9):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    print("Device:", device)
    mo.show_code()
    return (device,)


@app.cell(hide_code=True)
def _(custom_collate_fn, device, mo):
    from functools import partial

    customized_collate_fn = partial(
        custom_collate_fn, device=device, allowed_max_length=1024
    )
    mo.show_code()
    return (customized_collate_fn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 接下来，我们实例化数据加载器，方法与前几章类似，不同之处在于，我们现在为批处理过程提供了我们自己的整理函数。
    """)
    return


@app.cell(hide_code=True)
def _(
    InstructionDataset,
    customized_collate_fn,
    mo,
    tokenizer,
    torch,
    train_data,
):
    from torch.utils.data import DataLoader

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )
    mo.show_code()
    return DataLoader, batch_size, num_workers, train_loader


@app.cell(hide_code=True)
def _(
    DataLoader,
    InstructionDataset,
    batch_size,
    customized_collate_fn,
    mo,
    num_workers,
    test_data,
    tokenizer,
    val_data,
):
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
    mo.show_code()
    return (val_loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 让我们来看看最终的输入批次和目标批次的维度是什么样的。
    """)
    return


@app.cell(hide_code=True)
def _(mo, train_loader):
    print("Train loader:")
    for __inputs, __targets in train_loader:
        print(__inputs.shape, __targets.shape)
    mo.show_code()
    return __inputs, __targets


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 我们还要通过打印输入批次中第一个训练样本的内容，再次确认输入是否包含对应于标记 ID 50256 的 `<|endoftext|>` 填充标记。
    """)
    return


@app.cell
def _(__inputs):
    __inputs[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 同样，我们通过目视检查再次确认目标是否包含 -100 占位符标记。
    """)
    return


@app.cell
def _(__targets):
    __targets[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.5 加载预训练模型

    【图15】

    > 然而，我们没有加载参数量最小的`1.24`亿参数模型，而是加载了参数量为`3.55`亿参数的中等版本，因为`1.24`亿参数的模型太小，无法通过指令微调获得定性上合理的结果。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    from gpt2 import download_and_load_gpt2
    from blocks import GPTModel, load_weights_into_gpt

    BASE_CONFIG = {
        "vocab_size": 50257,  # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,  # Dropout rate
        "qkv_bias": True,  # Query-key-value bias
    }

    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    CHOOSE_MODEL = "gpt2-medium (355M)"

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    mo.show_code()
    return BASE_CONFIG, CHOOSE_MODEL, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 在下一节开始微调模型之前，让我们先看看它在其中一个验证任务上的表现。
    """)
    return


@app.cell(hide_code=True)
def _(format_input, mo, torch, val_data):
    torch.manual_seed(123)

    input_text = format_input(val_data[0])
    print(input_text)
    mo.show_code()
    return (input_text,)


@app.cell(hide_code=True)
def _(BASE_CONFIG, input_text, mo, model, tokenizer):
    from blocks import generate, text_to_token_ids, token_ids_to_text

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer),
        max_new_tokens=35,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256,
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    mo.show_code()
    return generate, generated_text, text_to_token_ids, token_ids_to_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 请注意，我们在前几章中使用的 `generate` 函数返回的是输入和输出文本的组合，这在上一节中对于生成易读文本非常方便。为了提取响应，我们可以从 `generated_text` 的开头减去指令的长度。
    """)
    return


@app.cell(hide_code=True)
def _(generated_text, input_text, mo):
    response_text = (
        generated_text[len(input_text) :].replace("### Response:", "").strip()
    )
    print(response_text)
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 正如我们所见，该模型目前还无法遵循指令；它创建了一个“响应”部分，但只是简单地重复了原始输入句子和指令。

    ## 6.6 使用指令数据微调模型

    【图16】

    :cloud: 我们可以重用之前章节中用到的所有损失计算和训练函数。
    """)
    return


@app.cell
def _(device, model, torch, train_loader, val_loader):
    from blocks import calc_loss_loader, train_model_simple

    model.to(device)

    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss:", train_loss)
    print("Validation loss:", val_loss)
    return (train_model_simple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 由于我们使用了更大的模型（3.55亿个参数而不是1.24亿个参数），因此训练成本比前几章要高一些。下面列出了各种设备的运行时间供参考（在兼容的GPU设备上运行此笔记本无需更改代码）。

    | Model              | Device                | Runtime for 2 Epochs |
    |--------------------|-----------------------|----------------------|
    | gpt2-medium (355M) | CPU (M3 MacBook Air)  | 15.78 minutes        |
    | gpt2-medium (355M) | GPU (M3 MacBook Air)  | 10.77 minutes        |
    | gpt2-medium (355M) | GPU (L4)              | 1.83 minutes         |
    | gpt2-medium (355M) | GPU (A100)            | 0.86 minutes         |
    | gpt2-small (124M)  | CPU (M3 MacBook Air)  | 5.74 minutes         |
    | gpt2-small (124M)  | GPU (M3 MacBook Air)  | 3.73 minutes         |
    | gpt2-small (124M)  | GPU (L4)              | 0.69 minutes         |
    | gpt2-small (124M)  | GPU (A100)            | 0.39 minutes         |
    """)
    return


@app.cell
def _(
    device,
    format_input,
    mo,
    model,
    tokenizer,
    torch,
    train_loader,
    train_model_simple,
    val_data,
    val_loader,
):
    import time

    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    mo.show_code()
    return num_epochs, tokens_seen, train_losses, val_losses


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 从上面的输出可以看出，模型训练效果良好，训练损失和验证损失值都在下降。
    + 此外，从每个 epoch 后打印的响应文本可以看出，模型正确地将输入句子“The chef cooks the meal every day.”转换为被动语态“The meal is cook every day by the chef.”
    + 最后，让我们看一下训练损失和验证损失曲线。
    """)
    return


@app.cell(hide_code=True)
def _(mo, num_epochs, tokens_seen, torch, train_losses, val_losses):
    from blocks import plot_losses

    # Alternatively:
    # from llms_from_scratch.ch05 import plot_losses

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    mo.ui.matplotlib(
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 损失在第一个训练周期开始时急剧下降，这意味着模型开始快速学习。
    + 大约在第一个训练周期左右，模型开始出现轻微的过拟合。

    ## 6.7 提取和保存响应

    【图17】
    """)
    return


@app.cell(hide_code=True)
def _(
    BASE_CONFIG,
    device,
    format_input,
    generate,
    input_text,
    mo,
    model,
    test_data,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer,
    torch,
):
    torch.manual_seed(123)


    for _entry in test_data[:3]:

        _input_text = format_input(_entry)

        _token_ids = generate(
            model=model,
            idx=text_to_token_ids(_input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        _generated_text = token_ids_to_text(_token_ids, tokenizer)
        _response_text = (
            _generated_text[len(_input_text) :]
            .replace("### Response:", "")
            .strip()
        )

        print(input_text)
        print(f"\nCorrect response:\n>> {_entry['output']}")
        print(f"\nModel response:\n>> {_response_text.strip()}")
        print("-------------------------------------")

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 根据测试集指令、给定响应和模型响应，我们可以看出模型表现相对较好。

    + 第一道和最后一道指令的答案显然是正确的。

    + 第二个答案比较接近；模型给出的答案是“积云”而不是“积雨云”（但是，请注意，积云可以发展成积雨云，而积雨云能够产生雷暴）。

    + 最重要的是，我们可以看到模型评估不像上一章那样直接，上一章我们只需要计算正确垃圾邮件/非垃圾邮件类别标签的百分比即可获得分类准确率。

    + 实际上，像聊天机器人这样的指令微调语言学习模型（LLM）是通过多种方法进行评估的。

    + 简答题和多项选择题基准测试，例如 MMLU（“大规模多任务语言理解测量”，https://arxiv.org/abs/2009.03300://arxiv.org/abs/2009.03300），用于测试模型的知识。

    + 将模型的人类偏好与其他语言学习模型进行比较，例如 LMSYS 聊天机器人竞技场（https://arena.Imsys.org）。

    + 自动化对话基准测试，其中使用另一个 LLM（例如 GPT-4）来评估响应，例如 AlpacaEval (https://tatsu-lab.github.io/alpaca)

    + 在下一节中，我们将使用类似于 `AlpacaEval` 的方法，并使用另一个 LLM 来评估我们模型的响应；但是，我们将使用我们自己的测试集，而不是使用公开可用的基准数据集

    + 为此，我们将模型响应添加到 `test_data` 字典中，并将其保存为“instruction-data-with-response.json”文件以进行记录，以便我们可以在需要时在单独的 Python 会话中加载和分析它
    """)
    return


@app.cell
def _(
    BASE_CONFIG,
    device,
    format_input,
    generate,
    json,
    model,
    test_data,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer,
):
    from tqdm import tqdm

    for _i, _entry in tqdm(enumerate(test_data), total=len(test_data)):

        _input_text = format_input(_entry)

        _token_ids = generate(
            model=model,
            idx=text_to_token_ids(_input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256,
        )
        _generated_text = token_ids_to_text(_token_ids, tokenizer)
        _response_text = (
            _generated_text[len(_input_text) :]
            .replace("### Response:", "")
            .strip()
        )
        test_data[_i]["model_response"] = _response_text


    with open("instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)  # "indent" for pretty-printing
    return (tqdm,)


@app.cell
def _(test_data):
    test_data[0]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 最后，我们还会保存模型，以备将来需要时使用。
    """)
    return


@app.cell
def _(CHOOSE_MODEL, model, torch):
    import re

    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6.8 评估微调后的模型

    【图18】

    + 在本节中，我们使用另一个更大的LLM来自动评估微调后的LLM的响应。

    + 具体来说，我们使用`Meta AI`开发的指令微调的80亿参数Llama 3模型，该模型可以通过ollama（https://ollama.com）在本地运行。

    > [Ollama](https://ollama.com/)是一个高效运行LLM的应用程序。
    >
    > 它是llama.cpp（https://github.com/ggerganov/llama.cpp）的封装，llama.cpp使用纯C/C++实现LLM以最大限度地提高效率。

    :fire: 请注意，它是一个用于使用LLM生成文本（推理）的工具，而不是用于训练或微调LLM的工具
    """)
    return


@app.cell
def _(mo):
    import psutil


    def check_if_running(process_name):
        running = False
        for proc in psutil.process_iter(["name"]):
            if process_name in proc.info["name"]:
                running = True
                break
        return running


    ollama_running = check_if_running("ollama")

    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))
    mo.show_code()
    return


@app.cell
def _(json, mo):
    _file_path = "instruction-data-with-response.json"

    with open(_file_path, "r") as _file:
        test_data_v2 = json.load(_file)


    def _format_input(entry):
        _instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        _input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return _instruction_text + _input_text


    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 现在，除了之前使用的 ollama run 命令之外，我们还可以通过 Python 中的 REST API 使用以下函数与模型交互。在运行此笔记本中的后续单元格之前，请确保 ollama 仍在运行（之前的代码单元格应打印“Ollama running: True”）。接下来，运行以下代码单元格来查询模型。
    """)
    return


@app.cell
def _(json, mo, requests):
    def query_model(
        prompt,
        _model="llama3",
        # If you used OLLAMA_HOST=127.0.0.1:11435 ollama serve
        # update the address from 11434 to 11435
        url="http://localhost:11434/api/chat",
    ):
        # Create the data payload as a dictionary
        data = {
            "model": _model,
            "messages": [{"role": "user", "content": prompt}],
            "options": {  # Settings below are required for deterministic responses
                "seed": 123,
                "temperature": 0,
                "num_ctx": 2048,
            },
        }

        """
        # Convert the dictionary to a JSON formatted string and encode it to bytes
        payload = json.dumps(data).encode("utf-8")

        # Create a request object, setting the method to POST and adding necessary headers
        request = urllib.request.Request(
            url,
            data=payload,
            method="POST"
        )
        request.add_header("Content-Type", "application/json")

        # Send the request and capture the response
        response_data = ""
        with urllib.request.urlopen(request) as response:
            # Read and decode the response
            while True:
                line = response.readline().decode("utf-8")
                if not line:
                    break
                response_json = json.loads(line)
                response_data += response_json["message"]["content"]

        return response_data
        """

        # The book originally used the commented-out above, which is based
        # on urllib. It works generally fine, but some readers reported
        # issues with using urlib when using a (company) VPN.
        # The code below uses the requests library, which doesn't seem
        # to have these issues.

        # Send the POST request
        with requests.post(url, json=data, stream=True, timeout=30) as r:
            r.raise_for_status()
            response_data = ""
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                response_json = json.loads(line)
                if "message" in response_json:
                    response_data += response_json["message"]["content"]

        return response_data


    _model = "llama3"
    result = query_model("What do Llamas eat?", _model)
    print(result)
    mo.show_code()
    return (query_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 现在，使用我们上面定义的 `query_model` 函数，我们可以评估微调后的模型的响应；让我们用之前章节中提到的前 3 个测试集响应来测试一下。
    """)
    return


@app.cell(hide_code=True)
def _(format_input, mo, query_model, test_data):
    for _entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(_entry)}` "
            f"and correct output `{_entry['output']}`, "
            f"score the model response `{_entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )
        print("\nDataset response:")
        print(">>", _entry["output"])
        print("\nModel response:")
        print(">>", _entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt))
        print("\n-------------------------")
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 正如我们所见，Llama 3 模型提供了合理的评估，即使模型并非完全正确，也会给予部分分数，正如我们从“积云”的答案中看到的那样。请注意，之前的提示会返回非常详细的评估结果；我们可以调整提示，使其生成 0 到 100 之间的整数响应（100 为最佳），以便计算模型的平均得分。
    """)
    return


@app.cell
def _(format_input, mo, query_model, test_data, tqdm):
    def generate_model_scores(json_data, json_key, model="llama3"):
        scores = []
        for entry in tqdm(json_data, desc="Scoring entries"):
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            score = query_model(prompt, model)
            try:
                scores.append(int(score))
            except ValueError:
                print(f"Could not convert score: {score}")
                continue

        return scores


    scores = generate_model_scores(test_data, "model_response")
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")
    mo.show_code()
    return


if __name__ == "__main__":
    app.run()
