import marimo

__generated_with = "0.19.9"
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

    【图1】

    ## 5.1 不同类型的微调

    > 语言模型微调最常见的方法是`指令微调(instruction-finetuning)`和`分类微调(classification finetuning)`，本章主要介绍分类微调。

    【图2】

    **分类微调**类似于训练卷积神经网络来对手写数字进行分类。在分类微调中，我们为模型可以输出的特定数量的类别标签（例如，“垃圾邮件”和“非垃圾邮件”）。

    【图3】

    ## 5.2 准备数据集

    > 我们使用包含垃圾短信和非垃圾短信的数据集来微调LLM模型，以对其进行分类。

    【图4】

    [:link: 下载连接](https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip)
    """)
    return


@app.cell
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


@app.cell
def _(data_file_path, mo):
    import pandas as pd

    df = pd.read_csv(
        data_file_path, sep="\t", header=None, names=["Label", "Text"]
    )

    mo.vstack([mo.show_code(), mo.ui.data_editor(df)])
    return df, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 这里为了缩短训练时间，我们使用较小的数据量进行微调，为了保证平衡性，将两类数据的条数保持一致。
    """)
    return


@app.cell
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


@app.cell
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

    【图1】
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
    【图2】

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


@app.cell
def _(mo):
    mo.md(r"""
    :hammer: 最后，我们打印每个数据集中的批次总数。
    """)
    return


@app.cell
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

    【图3】
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

    【图4】

    :rocket: 首先，让我们看一下模型架构。
    """)
    return


@app.cell
def _(model):
    print(model)
    return


@app.cell
def _(mo):
    mo.md(r"""
    + 上图清晰地展示了我们在第四章中实现的架构。
    + 目标是替换并微调输出层。
    + 为了实现这一目标，我们首先冻结模型，即将所有层设置为不可训练。
    """)
    return


@app.cell
def _(mo, model):
    for param in model.parameters():
        param.requires_grad = False

    mo.show_code()
    return


if __name__ == "__main__":
    app.run()
