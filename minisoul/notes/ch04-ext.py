import marimo

__generated_with = "0.17.2"
app = marimo.App(css_file="custom.css")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. 预训练GPT

    ## 1.1. Gutenberg Dataset
    > Project Gutenberg 是全球最早、最大的免费电子书库之一，包含数万本已进入公共领域的文学作品（小说、戏剧、散文、诗歌等）。
    >
    > Gutenberg Dataset 就是从这些电子书中提取的原始文本或清洗后的版本，用于研究与训练。

    原著中的获取数据的方法如下：

    1. Clone the `guteberg`

    ```shell
    git clone https://github.com/pgcorpus/gutenberg.git
    ```

    2. Install the required packages defined in requirements.txt from the gutenberg repository's folder:

    ```shell
    cd gutenberg && pip install -r requirements.txt
    ```

    3. Download the data:

    ```shell
    python get_data.py
    ```

    > 由于该项目需要用到`rsync`命令，但是`Windows`中未提供，因此可选择以下方式。

    ///attention | 注意

    1. 使用脚本或者官网链接 [:link: 官网](https://zenodo.org/records/2422561) 下载速度非常慢
    2. 数据集中包含了多种语言，我们只需要英文类的，因此可以选择清洗过的数据集 [:link: huggingface.co](https://huggingface.co/datasets/incredible45/Gutenberg-BookCorpus-Cleaned-Data-English)

    **注：这些数据集不可直接使用，需要进行转换！**
    ///

    :rocket: 以下是数据集转换代码

    ```python
    from datasets import load_dataset
    import os
    from tqdm import tqdm


    def combine_dataset_to_text(
        dataset,
        target_dir,
        text_field="context",
        max_size_mb=500,
        separator="<|endoftext|>",
    ):
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        current_content = []
        current_size = 0
        file_counter = 1

        num_rows = len(dataset)

        for i in tqdm(range(num_rows)):
            content = dataset[i][text_field]

            if not isinstance(content, str):
                continue

            estimated_size = len(content.encode("utf-8"))

            if current_size + estimated_size > max_size_mb * 1024 * 1024:
                path = os.path.join(target_dir, f"combined_{file_counter}.txt")
                with open(path, "w", encoding="utf-8") as f:
                    f.write(separator.join(current_content))

                file_counter += 1
                current_content = [content]
                current_size = estimated_size
            else:
                current_content.append(content)
                current_size += estimated_size

        if current_content:
            path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(separator.join(current_content))

        return file_counter


    # 执行prepare_dataset()即可完成数据集转换
    def prepare_dataset():
        dataset = load_dataset(
            "parquet",
            # 下载的数据集文件
            data_files={
                "train": "/root/.cache/huggingface/hub/datasets--incredible45--Gutenberg-BookCorpus-Cleaned-Data-English/snapshots/b316d9e4b38eba427f8016477ac4078c4a46dbca/data/*.parquet"
            },
        )
        num_files = combine_dataset_to_text(
            dataset=dataset["train"],
            target_dir="gutenberg_preprocessed_from_hf",
            text_field="context",  # 你数据里的文本字段
            max_size_mb=500,
        )

        print(f"输出 {num_files} 个 .txt 文件")

    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 如果你觉得麻烦，可以直接用我转换好的数据集

    + ModelScope: [:link: caoaolong/minisoul-gutenberg-en_US](https://modelscope.cn/datasets/caoaolong/minisoul-gutenberg-en_US)
    + Huggingface: [:link: caoaolong/minisoul-gutenberg-en_US](https://huggingface.co/datasets/caoaolong/minisoul-gutenberg-en_US)

    ## 1.2. 训练模型

    > 数据集准备好后就可以进行训练了。我在`NVIDIA GeForce RTX 3090(24G)`设备上训练全部数据用时大概`120h`左右。

    :rocket: 以下是训练代码

    ```python
    # Copyright (c) Sebastian Raschka under Apache License 2.0.
    # Modified by calong

    "\"\"
    Script for pretraining a small GPT-2 124M parameter model
    on books from Project Gutenberg.

    Modified:
    1. All print() replaced with logging
    2. Fixed checkpoint saving logic
    "\"\"

    import argparse
    import os
    import logging
    from pathlib import Path
    import time
    import tiktoken
    import torch

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # project modules
    from blocks import create_dataloader_v1
    from blocks import GPTModel
    from loader import (
        calc_loss_batch,
        evaluate_model,
        plot_losses,
        generate_and_print_sample,
    )


    def read_text_file(file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        return text_data


    def create_dataloaders(
        text_data, train_ratio, batch_size, max_length, stride, num_workers=0
    ):
        split_idx = int(train_ratio * len(text_data))
        train_loader = create_dataloader_v1(
            text_data[:split_idx],
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            drop_last=True,
            shuffle=True,
            num_workers=num_workers,
        )
        val_loader = create_dataloader_v1(
            text_data[split_idx:],
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            drop_last=False,
            shuffle=False,
            num_workers=num_workers,
        )
        return train_loader, val_loader


    def convert_time(seconds):
        hours, rem = divmod(seconds, 3600)
        minutes, seconds = divmod(rem, 60)
        return int(hours), int(minutes), int(seconds)


    def print_eta(start_time, book_start_time, index, total_files):
        book_end_time = time.time()
        elapsed_time = book_end_time - book_start_time
        total_elapsed_time = book_end_time - start_time
        books_remaining = total_files - index
        average_time_per_book = total_elapsed_time / index
        eta = average_time_per_book * books_remaining

        book_h, book_m, book_s = convert_time(elapsed_time)
        total_h, total_m, total_s = convert_time(total_elapsed_time)
        eta_h, eta_m, eta_s = convert_time(eta)

        return (
            f"Book processed {book_h}h {book_m}m {book_s}s\n"
            f"Total elapsed {total_h}h {total_m}m {total_s}s\n"
            f"ETA: {eta_h}h {eta_m}m {eta_s}s"
        )


    def train_model_simple(
        model,
        optimizer,
        device,
        n_epochs,
        eval_freq,
        eval_iter,
        print_sample_iter,
        start_context,
        output_dir,
        save_ckpt_freq,
        tokenizer,
        batch_size=1024,
        train_ratio=0.90,
    ):
        train_losses, val_losses, track_tokens_seen = [], [], []
        tokens_seen = 0
        global_step = -1
        start_time = time.time()

        try:
            for epoch in range(n_epochs):
                for index, file_path in enumerate(all_files, 1):
                    book_start_time = time.time()

                    logger.info(
                        f"Tokenizing file {index}/{total_files}: {file_path}"
                    )
                    text_data = read_text_file(file_path) + " <|endoftext|> "

                    train_loader, val_loader = create_dataloaders(
                        text_data,
                        train_ratio=train_ratio,
                        batch_size=batch_size,
                        max_length=GPT_CONFIG_124M["context_length"],
                        stride=GPT_CONFIG_124M["context_length"],
                        num_workers=0,
                    )

                    logger.info("Training ...")
                    model.train()

                    for input_batch, target_batch in train_loader:
                        optimizer.zero_grad()
                        loss = calc_loss_batch(
                            input_batch, target_batch, model, device
                        )
                        loss.backward()
                        optimizer.step()

                        tokens_seen += input_batch.numel()
                        global_step += 1

                        # evaluation
                        if global_step % eval_freq == 0:
                            train_loss, val_loss = evaluate_model(
                                model, train_loader, val_loader, device, eval_iter
                            )
                            train_losses.append(train_loss)
                            val_losses.append(val_loss)
                            track_tokens_seen.append(tokens_seen)
                            logger.info(
                                f"Epoch {epoch + 1}, Step {global_step}, "
                                f"Train {train_loss:.3f}, Val {val_loss:.3f}"
                            )

                        # sample generation
                        if global_step % print_sample_iter == 0:
                            generate_and_print_sample(
                                model, tokenizer, device, start_context
                            )

                        # ===== 修复后的 checkpoint 保存逻辑 =====
                        if (global_step + 1) % save_ckpt_freq == 0:
                            ckpt_path = (
                                output_dir / f"model_pg_step_{global_step + 1}.pth"
                            )
                            torch.save(model.state_dict(), ckpt_path)
                            logger.info(f"Checkpoint saved: {ckpt_path}")

                    # per-book ETA output
                    logger.info(
                        print_eta(start_time, book_start_time, index, total_files)
                    )

        except KeyboardInterrupt:
            ckpt_path = output_dir / f"model_pg_{global_step}_interrupted.pth"
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Interrupted! Emergency saved: {ckpt_path}")

        return train_losses, val_losses, track_tokens_seen


    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description="GPT Model Training Configuration"
        )

        parser.add_argument(
            "--data_dir",
            type=str,
            default="gutenberg/data",
            help="Directory containing the training data",
        )
        parser.add_argument(
            "--output_dir",
            type=str,
            default="model_checkpoints",
            help="Directory where the model checkpoints will be saved",
        )
        parser.add_argument("--n_epochs", type=int, default=1)
        parser.add_argument("--print_sample_iter", type=int, default=1000)
        parser.add_argument("--eval_freq", type=int, default=100)
        parser.add_argument("--save_ckpt_freq", type=int, default=100_000)
        parser.add_argument("--lr", type=float, default=5e-4)
        parser.add_argument("--batch_size", type=int, default=4)
        parser.add_argument("--debug", type=bool, default=False)

        args = parser.parse_args()

        # model config
        if args.debug:
            GPT_CONFIG_124M = {
                "vocab_size": 50257,
                "context_length": 10,
                "emb_dim": 12,
                "n_heads": 2,
                "n_layers": 2,
                "drop_rate": 0.0,
                "qkv_bias": False,
            }
        else:
            GPT_CONFIG_124M = {
                "vocab_size": 50257,
                "context_length": 1024,
                "emb_dim": 768,
                "n_heads": 12,
                "n_layers": 12,
                "drop_rate": 0.1,
                "qkv_bias": False,
            }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(123)

        model = GPTModel(GPT_CONFIG_124M).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.1
        )
        tokenizer = tiktoken.get_encoding("gpt2")

        # load training files
        data_dir = args.data_dir
        all_files = [
            os.path.join(path, name)
            for path, subdirs, files in os.walk(data_dir)
            for name in files
            if name.endswith(".txt")
        ]
        total_files = len(all_files)

        if total_files == 0:
            logger.error("No training text files found!")
            quit()

        logger.info(f"Total files: {total_files}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        train_losses, val_losses, tokens_seen = train_model_simple(
            model,
            optimizer,
            device,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            eval_freq=args.eval_freq,
            eval_iter=1,
            print_sample_iter=args.print_sample_iter,
            output_dir=output_dir,
            save_ckpt_freq=args.save_ckpt_freq,
            start_context="Every effort moves you",
            tokenizer=tokenizer,
        )

        epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
        plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

        final_path = output_dir / "model_pg_final.pth"
        torch.save(model.state_dict(), final_path)
        logger.info(f"Training complete. Final saved: {final_path}")
        logger.info(
            f"Max GPU memory used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
        )
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :rocket: 以下是训练模型的执行命令

    ```shell
    python pretraining_simple.py \
        --data_dir "gutenberg_preprocessed_from_hf"\
        --n_epochs 1\
        --batch_size 6\
        --output_dir model_checkpoints
    ```

    **注: `batch_size=6` 大约需要占用 `22G` 左右的显存。**
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    :fire: 如果你暂时不便去进行训练，或者只想了解以下最终的效果可以使用我已经训练好的模型。

    + ModelScope: [:link: caoaolong/minisoul](https://modelscope.cn/datasets/caoaolong/minisoul)
    + Huggingface: [:link: caoaolong/minisoul](https://huggingface.co/caoaolong/minisoul)
    """)
    return


if __name__ == "__main__":
    app.run()
