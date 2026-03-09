import marimo

__generated_with = "0.20.1"
app = marimo.App(
    width="medium",
    app_title="LLMs-文本分类的微调",
    css_file="custom.css",
    html_head_file="",
)


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _():
    from importlib.metadata import version

    pkgs = [
        "matplotlib",  # Plotting library
        "numpy",  # PyTorch & TensorFlow dependency
        "tiktoken",  # Tokenizer
        "torch",  # Deep learning library
        "tensorflow",  # For OpenAI's pretrained weights
        "pandas",  # Dataset loading
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5.文本分类的微调

    ![【图1】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-1.svg)

    ## 5.1 不同类型的微调

    > 语言模型微调最常见的方法是`指令微调(instruction-finetuning)`和`分类微调(classification finetuning)`，本章主要介绍分类微调。

    ![【图2】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-2.svg)

    + 本章的主题是分类微调，如果您有机器学习背景，可能已经熟悉这个过程——例如，它类似于训练卷积神经网络来对手写数字进行分类。
    + 在分类微调中，我们为模型设定了特定数量的类别标签（例如，“垃圾邮件”和“非垃圾邮件”）。
    + 经过分类微调的模型只能预测它在训练过程中遇到的类别（例如，“垃圾邮件”或“非垃圾邮件”），而经过指令微调的模型通常可以执行多种任务。我们可以将分类微调的模型视为一个非常专业的模型。
    + 实际上，创建一个专业模型比创建一个可以在许多不同任务上表现良好的通用模型要容易得多。

    ![【图3】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-3.svg)

    ## 5.2 准备数据集

    > 我们使用包含垃圾短信和非垃圾短信的数据集来微调LLM模型，以对其进行分类。

    ![【图4】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-4.svg)

    [:link: 下载连接](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    import zipfile
    from pathlib import Path
    import os

    extracted_path = "data/sms_spam_collection"
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


    def prepare_dataset():
        sms_spam_collection_file = "data/sms+spam+collection.zip"
        if os.path.exists(sms_spam_collection_file):
            print("Dataset already exists at:", data_file_path)
            return
        # Unzipping the file
        with zipfile.ZipFile(sms_spam_collection_file, "r") as f:
            f.extractall(extracted_path)

        # Add .tsv file extension
        original_file_path = Path(extracted_path) / "SMSSpamCollection"
        os.rename(original_file_path, data_file_path)
        print(f"File downloaded and saved as {data_file_path}")


    prepare_dataset()

    mo.show_code()
    return (data_file_path,)


@app.cell(hide_code=True)
def _(data_file_path, mo):
    import pandas as pd

    df = pd.read_csv(
        data_file_path, sep="\t", header=None, names=["Label", "Text"]
    )

    mo.ui.data_editor(df)
    return df, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 这里为了缩短训练时间，我们使用较小的数据量进行微调，为了保证平衡性，将两类数据的条数保持一致。
    """)
    return


@app.cell(hide_code=True)
def _(df, mo, pd):
    def create_balanced_dataset(df):
        # Count the instances of "spam"
        num_spam = df[df["Label"] == "spam"].shape[0]

        # Randomly sample "ham" instances to match the number of "spam" instances
        ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

        # Combine ham "subset" with "spam"
        balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

        return balanced_df


    balanced_df = create_balanced_dataset(df)
    print(balanced_df["Label"].value_counts())

    mo.show_code()
    return (balanced_df,)


@app.cell
def _(balanced_df, mo):
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    mo.ui.data_editor(balanced_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 现在我们来定义一个函数，该函数将数据集随机划分为训练集、验证集和测试集。
    """)
    return


@app.cell(hide_code=True)
def _(balanced_df, mo):
    def random_split(df, train_frac, validation_frac):
        # Shuffle the entire DataFrame
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)

        # Calculate split indices
        train_end = int(len(df) * train_frac)
        validation_end = train_end + int(len(df) * validation_frac)

        # Split the DataFrame
        train_df = df[:train_end]
        validation_df = df[train_end:validation_end]
        test_df = df[validation_end:]

        return train_df, validation_df, test_df


    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    # Test size is implied to be 0.2 as the remainder

    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.3 创建数据加载器

    > 请注意，文本消息的长度各不相同；如果我们想将多个训练样本合并到一个批次中，我们有以下两种选择
    >
    > 1. 将所有消息截断为数据集或批次中最短消息的长度
    >
    > 2. 将所有消息填充为数据集或批次中最长消息的长度（需要使用` <lendoftext|> `作为填充标记）
    >
    > 这里为了保证数据的完整性我们选择选第二项。

    ![【图5】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-5.svg)
    """)
    return


@app.cell
def _(mo):
    import tiktoken

    tokenizer = tiktoken.get_encoding("gpt2")
    print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))

    mo.show_code()
    return (tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 下面的 `SpamDataset` 类会识别训练数据集中的最长序列，并向其他序列添加填充标记，使其长度与该最长序列相匹配。
    """)
    return


@app.cell(hide_code=True)
def _(mo, pd):
    import torch
    from torch.utils.data import Dataset


    class SpamDataset(Dataset):
        def __init__(
            self, csv_file, tokenizer, max_length=None, pad_token_id=50256
        ):
            self.data = pd.read_csv(csv_file)

            # Pre-tokenize texts
            self.encoded_texts = [
                tokenizer.encode(text) for text in self.data["Text"]
            ]

            if max_length is None:
                self.max_length = self._longest_encoded_length()
            else:
                self.max_length = max_length
                # Truncate sequences if they are longer than max_length
                self.encoded_texts = [
                    encoded_text[: self.max_length]
                    for encoded_text in self.encoded_texts
                ]

            # Pad sequences to the longest sequence
            self.encoded_texts = [
                encoded_text
                + [pad_token_id] * (self.max_length - len(encoded_text))
                for encoded_text in self.encoded_texts
            ]

        def __getitem__(self, index):
            encoded = self.encoded_texts[index]
            label = self.data.iloc[index]["Label"]
            return (
                torch.tensor(encoded, dtype=torch.long),
                torch.tensor(label, dtype=torch.long),
            )

        def __len__(self):
            return len(self.data)

        def _longest_encoded_length(self):
            max_length = 0
            for encoded_text in self.encoded_texts:
                encoded_length = len(encoded_text)
                if encoded_length > max_length:
                    max_length = encoded_length
            return max_length
            # Note: A more pythonic version to implement this method
            # is the following, which is also used in the next chapter:
            # return max(len(encoded_text) for encoded_text in self.encoded_texts)


    mo.show_code()
    return SpamDataset, torch


@app.cell
def _(SpamDataset, mo, tokenizer):
    train_dataset = SpamDataset(
        csv_file="train.csv", max_length=None, tokenizer=tokenizer
    )

    print(train_dataset.max_length)

    mo.show_code()
    return (train_dataset,)


@app.cell
def _(SpamDataset, mo, tokenizer, train_dataset):
    val_dataset = SpamDataset(
        csv_file="validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    test_dataset = SpamDataset(
        csv_file="test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer,
    )
    mo.show_code()
    return test_dataset, val_dataset


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![【图6】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-6.svg)

    :fire: 接下来，我们使用数据集来实例化数据加载器，这与前面章节中创建数据加载器的过程类似。
    """)
    return


@app.cell
def _(mo, test_dataset, torch, train_dataset, val_dataset):
    from torch.utils.data import DataLoader

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    mo.show_code()
    return test_loader, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 作为验证步骤，我们遍历数据加载器，确保每个批次包含 8 个训练样本，其中每个训练样本包含 120 个标记。
    """)
    return


@app.cell
def _(mo, train_loader):
    print("Train loader:")
    for input_batch, target_batch in train_loader:
        pass

    print("Input batch dimensions:", input_batch.shape)
    print("Label batch dimensions", target_batch.shape)

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 最后，我们打印每个数据集中的批次总数。
    """)
    return


@app.cell(hide_code=True)
def _(mo, test_loader, train_loader, val_loader):
    print(f"{len(train_loader)} training batches")
    print(f"{len(val_loader)} validation batches")
    print(f"{len(test_loader)} test batches")

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.4 使用预训练权重初始化模型

    > 在本节中，我们将初始化上一章中使用的预训练模型。

    ![【图3】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-7.svg)
    """)
    return


@app.cell
def _(mo, train_dataset):
    CHOOSE_MODEL = "gpt2-small (124M)"
    INPUT_PROMPT = "Every effort moves"

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

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )

    mo.show_code()
    return BASE_CONFIG, CHOOSE_MODEL


@app.cell
def _(BASE_CONFIG, CHOOSE_MODEL, mo):
    from gpt2 import download_and_load_gpt2
    from blocks import load_weights_into_gpt, GPTModel

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size, models_dir="gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()

    mo.show_code()
    return (model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 为确保模型已正确加载，我们再检查一下它是否生成了连贯的文本。
    """)
    return


@app.cell
def _(BASE_CONFIG, mo, model, tokenizer):
    from blocks import generate_text_simple, text_to_token_ids, token_ids_to_text

    text_1 = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"],
    )

    print(token_ids_to_text(token_ids, tokenizer))

    mo.show_code()
    return generate_text_simple, text_to_token_ids, token_ids_to_text


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 在对模型进行微调以使其具备分类器功能之前，让我们先看看该模型是否能够通过提示对垃圾邮件进行分类。
    """)
    return


@app.cell
def _(
    BASE_CONFIG,
    generate_text_simple,
    mo,
    model,
    text_to_token_ids,
    token_ids_to_text,
    tokenizer,
):
    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    token_ids_v2 = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"],
    )

    print(token_ids_to_text(token_ids_v2, tokenizer))

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 正如我们所见，该模型在执行指令方面表现不佳。这是意料之中的，因为它仅经过预训练，而没有进行指令微调。

    ## 5.5 添加分类头

    > 在本节中，我们将修改预训练的LLM模型，使其能够进行分类微调。

    ![【图8】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-8.svg)

    :rocket: 首先，让我们看一下模型架构。
    """)
    return


@app.cell
def _(model):
    print(model)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 上图清晰地展示了我们在第四章中实现的架构。
    + 目标是替换并微调输出层。
    + 为了实现这一目标，我们首先冻结模型，即将所有层设置为不可训练。
    """)
    return


@app.cell(hide_code=True)
def _(mo, model):
    for param in model.parameters():
        param.requires_grad = False

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 然后，我们替换输出层（model.out_head），它原本将层输入映射到 50,257 维（词汇表的大小）。
    + 由于我们微调模型是为了进行二元分类（预测两个类别：“垃圾邮件”和“非垃圾邮件”），我们可以像下面这样替换输出层，默认情况下它是可训练的。
    + 请注意，我们使用 BASE_CONFIG["emb_dim"]（在“gpt2-small (124M)”模型中等于 768）来使下面的代码更具通用性。
    """)
    return


@app.cell(hide_code=True)
def _(BASE_CONFIG, mo, model, torch):
    torch.manual_seed(123)

    num_classes = 2
    model.out_head = torch.nn.Linear(
        in_features=BASE_CONFIG["emb_dim"], out_features=num_classes
    )

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 理论上，只需训练输出层就足够了。
    + 但是大量的实验证明，微调额外的层可以显著提升性能。因此，我们也使最后一个 Transformer 模块以及连接最后一个 Transformer 模块和输出层的最终 LayerNorm 模块可训练。

    ![【图9】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-9.svg)
    """)
    return


@app.cell(hide_code=True)
def _(mo, model):
    def config_transformer_block():
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True

        for param in model.final_norm.parameters():
            param.requires_grad = True


    config_transformer_block()
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 我们仍然可以像前几章那样使用这个模型。 例如，让我们给它输入一些文本。
    """)
    return


@app.cell
def _(tokenizer, torch):
    inputs = tokenizer.encode("Do you have time")
    inputs = torch.tensor(inputs).unsqueeze(0)
    print("Inputs:", inputs)
    print("Inputs dimensions:", inputs.shape)  # shape: (batch_size, num_tokens)
    return (inputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 与前几章不同的是，现在输出维度从 50,257 个减少到两个。
    """)
    return


@app.cell(hide_code=True)
def _(inputs, mo, model, torch):
    with torch.no_grad():
        outputs = model(inputs)

    print("Outputs:\n", outputs)
    print(
        "Outputs dimensions:", outputs.shape
    )  # shape: (batch_size, num_tokens, num_classes)
    mo.show_code()
    return (outputs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 如前几章所述，每个输入词元对应一个输出向量。
    + 由于我们向模型输入了一个包含 4 个输入词元的文本样本，因此输出由上述 4 个二维输出向量组成。

    ![【图10】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-10.svg)

    + 在之前的章节中我们讨论了注意力机制，它将每个输入词元与其他所有输入词元关联起来。
    + 还介绍了GPT类模型中使用的因果注意力掩码；该因果掩码使得当前词元只关注其当前位置和前一个位置的词元。
    + 基于这种因果注意力机制，第四个（最后一个）词元包含的信息量最大，因为它是唯一一个包含了所有其他词元信息的词元。
    + 因此，我们对最后一个词元特别感兴趣，并将对其进行微调以用于垃圾邮件分类任务。
    """)
    return


@app.cell(hide_code=True)
def _(mo, outputs):
    print("Last output token:", outputs[:, -1, :])
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![【图11】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-11.svg)

    ## 5.6 计算分类损失和准确率

    ![【图12】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-12.svg)

    :fire: 在解释损失计算之前，我们先简要了解一下模型输出是如何转换为类别标签的。

    ![【图13】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-13.svg)
    """)
    return


@app.cell(hide_code=True)
def _(mo, outputs):
    print("Last output token:", outputs[:, -1, :])
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 与上一章类似，我们首先通过 `softmax` 函数将输出（`logits`）转换为概率分数，然后通过 `argmax` 函数获取最大概率值的索引位置。
    """)
    return


@app.cell(hide_code=True)
def _(mo, outputs, torch):
    probas = torch.softmax(outputs[:, -1, :], dim=-1)
    label = torch.argmax(probas)
    print("Class label:", label.item())
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 其实，`softmax` 函数在此处是可选的，因为最大的输出对应于最大的概率得分。
    """)
    return


@app.cell(hide_code=True)
def _(mo, outputs, torch):
    logits = outputs[:, -1, :]
    label_v2 = torch.argmax(logits)
    print("Class label:", label_v2.item())
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 我们可以将这个概念应用于计算所谓的分类准确率，即给定数据集中正确预测的百分比。
    + 为了计算分类准确率，我们可以将前面基于 `argmax` 的预测代码应用于数据集中的所有样本。

    并按如下方式计算正确预测的比例：
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    def calc_accuracy_loader(data_loader, model, device, num_batches=None):
        model.eval()
        correct_predictions, num_examples = 0, 0

        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                input_batch, target_batch = (
                    input_batch.to(device),
                    target_batch.to(device),
                )

                with torch.no_grad():
                    logits = model(input_batch)[
                        :, -1, :
                    ]  # Logits of last output token
                predicted_labels = torch.argmax(logits, dim=-1)

                num_examples += predicted_labels.shape[0]
                correct_predictions += (
                    (predicted_labels == target_batch).sum().item()
                )
            else:
                break
        return correct_predictions / num_examples


    mo.show_code()
    return (calc_accuracy_loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 让我们应用该函数来计算不同数据集的分类准确率：
    """)
    return


@app.cell(hide_code=True)
def _(
    calc_accuracy_loader,
    mo,
    model,
    test_loader,
    torch,
    train_loader,
    val_loader,
):
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

    model.to(
        device
    )  # no assignment model = model.to(device) necessary for nn.Module classes

    torch.manual_seed(
        123
    )  # For reproducibility due to the shuffling in the training data loader

    train_accuracy = calc_accuracy_loader(
        train_loader, model, device, num_batches=10
    )
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device, num_batches=10
    )

    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")

    mo.show_code()
    return (device,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 正如我们所见，预测准确率并不高，因为我们还没有对模型进行微调。
    + 在训练之前，我们首先需要定义训练过程中要优化的损失函数。
    + 目标是最大化模型的垃圾邮件分类准确率；然而，分类准确率并非可微函数。
    + 因此，我们改为最小化交叉熵损失，以此作为最大化分类准确率的替代指标。
    + 这里的 `calc_loss_batch` 函数与之前相同，只是我们只对优化最后一个标记模型`(input_batch)[:, -1, :]` 感兴趣，而不是对所有标记模型(`input_batch`) 感兴趣。
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    def calc_loss_batch(input_batch, target_batch, model, device):
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)[:, -1, :]  # Logits of last output token
        loss = torch.nn.functional.cross_entropy(logits, target_batch)
        return loss


    mo.show_code()
    return (calc_loss_batch,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: `calc_loss_loader `函数与上一章相同
    """)
    return


@app.cell(hide_code=True)
def _(calc_loss_batch, mo):
    def calc_loss_loader(data_loader, model, device, num_batches=None):
        total_loss = 0.0
        if len(data_loader) == 0:
            return float("nan")
        elif num_batches is None:
            num_batches = len(data_loader)
        else:
            # Reduce the number of batches to match the total number of batches in the data loader
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
    return (calc_loss_loader,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 使用 `calc_closs_loader`，我们在开始训练之前计算初始训练集、验证集和测试集的损失。
    """)
    return


@app.cell(hide_code=True)
def _(
    calc_loss_loader,
    device,
    mo,
    model,
    test_loader,
    torch,
    train_loader,
    val_loader,
):
    with (
        torch.no_grad()
    ):  # Disable gradient tracking for efficiency because we are not training, yet
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=5)

    print(f"Training loss: {train_loss:.3f}")
    print(f"Validation loss: {val_loss:.3f}")
    print(f"Test loss: {test_loss:.3f}")
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 现在我们已经完成了微调之前的所有准备工作，下一节中，我们将训练模型以改善损失值，从而提高分类准确率。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5.7 监督微调模型

    + 在本节中，我们定义并使用训练函数来提高模型的分类准确率。
    + 下面的 `train_classifier_simple` 函数与我们在上一章中用于模型预训练的 `train_model_simple` 函数基本相同。
    + 唯一的两个区别是：
        + 我们现在跟踪的是训练样本的数量（`examples_seen`），而不是看到的词元数量；
        + 并且在每个 `epoch` 后计算准确率，而不是在每个 `epoch` 后打印示例文本。

    ![【图14】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-14.svg)
    """)
    return


@app.cell(hide_code=True)
def _(calc_accuracy_loader, calc_loss_batch, calc_loss_loader, mo, torch):
    def train_classifier_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs,
        eval_freq,
        eval_iter,
    ):
        # Initialize lists to track losses and examples seen
        train_losses, val_losses, train_accs, val_accs = [], [], [], []
        examples_seen, global_step = 0, -1

        # Main training loop
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode

            for input_batch, target_batch in train_loader:
                optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                loss.backward()  # Calculate loss gradients
                optimizer.step()  # Update model weights using loss gradients
                examples_seen += input_batch.shape[
                    0
                ]  # New: track examples instead of tokens
                global_step += 1

                # Optional evaluation step
                if global_step % eval_freq == 0:
                    train_loss, val_loss = evaluate_model(
                        model, train_loader, val_loader, device, eval_iter
                    )
                    train_losses.append(train_loss)
                    val_losses.append(val_loss)
                    print(
                        f"Ep {epoch + 1} (Step {global_step:06d}): "
                        f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                    )

            # Calculate accuracy after each epoch
            train_accuracy = calc_accuracy_loader(
                train_loader, model, device, num_batches=eval_iter
            )
            val_accuracy = calc_accuracy_loader(
                val_loader, model, device, num_batches=eval_iter
            )
            print(f"Training accuracy: {train_accuracy * 100:.2f}% | ", end="")
            print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
            train_accs.append(train_accuracy)
            val_accs.append(val_accuracy)

        return train_losses, val_losses, train_accs, val_accs, examples_seen


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


    mo.show_code()
    return (train_classifier_simple,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 现在我们已经准备好了训练函数和验证函数，接下来开始训练并进行验证：
    """)
    return


@app.cell(hide_code=True)
def _(
    device,
    mo,
    model,
    torch,
    train_classifier_simple,
    train_loader,
    val_loader,
):
    import time

    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    num_epochs = 5
    train_losses, val_losses, train_accs, val_accs, examples_seen = (
        train_classifier_simple(
            model,
            train_loader,
            val_loader,
            optimizer,
            device,
            num_epochs=num_epochs,
            eval_freq=50,
            eval_iter=5,
        )
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    mo.show_code()
    return (
        examples_seen,
        num_epochs,
        train_accs,
        train_losses,
        val_accs,
        val_losses,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 我们使用`matplotlib`来绘制一下损失函数的图像
    """)
    return


@app.cell(hide_code=True)
def _(examples_seen, mo, num_epochs, torch, train_losses, val_losses):
    import matplotlib.pyplot as plt


    def plot_values(
        epochs_seen, examples_seen, train_values, val_values, label="loss"
    ):
        fig, ax1 = plt.subplots(figsize=(5, 3))

        # Plot training and validation loss against epochs
        ax1.plot(epochs_seen, train_values, label=f"Training {label}")
        ax1.plot(
            epochs_seen, val_values, linestyle="-.", label=f"Validation {label}"
        )
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel(label.capitalize())
        ax1.legend()

        # Create a second x-axis for examples seen
        ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
        ax2.plot(
            examples_seen, train_values, alpha=0
        )  # Invisible plot for aligning ticks
        ax2.set_xlabel("Examples seen")

        fig.tight_layout()  # Adjust layout to make room
        plt.savefig(f"{label}-plot.pdf")
        return plt.gca()


    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    mo.ui.matplotlib(
        plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
    )
    return (plot_values,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :cloud: 从上图的下降斜率可以看出，该模型学习效果良好。此外，训练损失和验证损失非常接近，表明该模型没有过拟合训练数据的倾向。类似地，我们可以绘制如下的准确率曲线。
    """)
    return


@app.cell
def _(examples_seen, mo, num_epochs, plot_values, torch, train_accs, val_accs):
    epochs_tensor_v2 = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor_v2 = torch.linspace(0, examples_seen, len(train_accs))

    mo.ui.matplotlib(
        plot_values(
            epochs_tensor_v2,
            examples_seen_tensor_v2,
            train_accs,
            val_accs,
            label="accuracy",
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 根据上面的准确率图，我们可以看到模型在第 4 和第 5 个 epoch 后达到了相对较高的训练和验证准确率。
    + 但是，我们需要记住，我们在之前的训练函数中指定了 `eval_iter=5`，这意味着我们只评估了训练集和验证集的性能。
    + 我们可以按如下方式计算模型在完整数据集上的训练集、验证集和测试集性能。
    """)
    return


@app.cell(hide_code=True)
def _(
    calc_accuracy_loader,
    device,
    mo,
    model,
    test_loader,
    train_loader,
    val_loader,
):
    train_accuracy_v2 = calc_accuracy_loader(train_loader, model, device)
    val_accuracy_v2 = calc_accuracy_loader(val_loader, model, device)
    test_accuracy_v2 = calc_accuracy_loader(test_loader, model, device)

    print(f"Training accuracy: {train_accuracy_v2 * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy_v2 * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy_v2 * 100:.2f}%")

    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    + 我们可以看到，训练集和验证集的性能几乎完全相同。
    + 然而，基于略低的测试集性能，我们可以看出模型对训练数据存在轻微的过拟合，对用于调整某些超参数（例如学习率）的验证数据也存在轻微的过拟合。
    + 但这属于正常现象，可以通过增加模型的 `dropout` 或优化器设置中的 `weight_decay` 值来进一步缩小这种差距。

    ## 6.8 使用LLM作为垃圾邮件分类器

    ![【图15】](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/5-15.svg)

    + 最后，让我们实际应用一下微调后的 GPT 模型。
    + 下面的 `classify_review` 函数实现了与我们之前实现的 `SpamDataset` 类似的预处理步骤。
    + 然后，该函数返回模型预测的整数类别标签，并返回相应的类别名称。
    """)
    return


@app.cell(hide_code=True)
def _(mo, torch):
    def classify_review(
        text, model, tokenizer, device, max_length=None, pad_token_id=50256
    ):
        model.eval()

        # Prepare inputs to the model
        input_ids = tokenizer.encode(text)
        supported_context_length = model.pos_emb.weight.shape[0]
        # Note: In the book, this was originally written as pos_emb.weight.shape[1] by mistake
        # It didn't break the code but would have caused unnecessary truncation (to 768 instead of 1024)

        # Truncate sequences if they too long
        input_ids = input_ids[: min(max_length, supported_context_length)]
        assert max_length is not None, (
            "max_length must be specified. If you want to use the full model context, "
            "pass max_length=model.pos_emb.weight.shape[0]."
        )
        assert max_length <= supported_context_length, (
            f"max_length ({max_length}) exceeds model's supported context length ({supported_context_length})."
        )
        # Alternatively, a more robust version is the following one, which handles the max_length=None case better
        # max_len = min(max_length,supported_context_length) if max_length else supported_context_length
        # input_ids = input_ids[:max_len]

        # Pad sequences to the longest sequence
        input_ids += [pad_token_id] * (max_length - len(input_ids))
        input_tensor = torch.tensor(input_ids, device=device).unsqueeze(
            0
        )  # add batch dimension

        # Model inference
        with torch.no_grad():
            logits = model(input_tensor)[
                :, -1, :
            ]  # Logits of the last output token
        predicted_label = torch.argmax(logits, dim=-1).item()

        # Return the classified result
        return "spam" if predicted_label == 1 else "not spam"


    mo.show_code()
    return (classify_review,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 让我们通过以下几个例子来尝试一下。
    """)
    return


@app.cell(hide_code=True)
def _(classify_review, device, mo, model, tokenizer, train_dataset):
    text_t1 = (
        "You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award."
    )

    print(
        classify_review(
            text_t1, model, tokenizer, device, max_length=train_dataset.max_length
        )
    )
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(classify_review, device, mo, model, tokenizer, train_dataset):
    text_t2 = (
        "Hey, just wanted to check if we're still on"
        " for dinner tonight? Let me know!"
    )

    print(
        classify_review(
            text_t2, model, tokenizer, device, max_length=train_dataset.max_length
        )
    )
    mo.show_code()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 最后，我们保存模型，以便以后无需重新训练即可再次使用该模型。
    """)
    return


@app.cell
def _(model, torch):
    torch.save(model.state_dict(), "review_classifier.pth")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :hammer: 然后，在新会话中，我们可以按如下方式加载模型
    """)
    return


@app.cell(hide_code=True)
def _(device, mo, model, torch):
    model_state_dict = torch.load(
        "review_classifier.pth", map_location=device, weights_only=True
    )
    model.load_state_dict(model_state_dict)
    mo.show_code()
    return


if __name__ == "__main__":
    app.run()
