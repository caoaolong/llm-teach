import marimo

__generated_with = "0.19.9"
app = marimo.App(width="medium", app_title="LLMs-处理文本数据")

with app.setup(hide_code=True):
    # Initialization code that runs before all other cells
    import marimo as mo


@app.cell(hide_code=True)
def section01():
    mo.md(r"""
    参考项目: [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

    # 1.处理问本数据

    > 本章介绍数据准备和采样，以便为 LLM 准备输入数据

    ---

    ![1-1](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-001.svg)

    ---
    ## 1.1 理解词嵌入

    > 嵌入有很多种形式，这里我们主要介绍文本嵌入

    ![1-2](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-2.svg)

    /// details | :fire: 什么是*Embedding*
        type: info
    大模型中使用的数据嵌入维度很高，以至于无法用直观的方式展现（三维已经是人类能够理解的极限）。

    但是我们可以用低维度的数据来帮助我们理解什么是嵌入（*Embedding*）。
    ///

    ![1-3](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-03.svg)

    ## 1.2 文本标记化

    > 也可以理解为文本分词，主要工作就是将长文本（文章、小说）拆分为更小的单位，比如单词和标点符号。

    ![1-4](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-04.svg)

    /// details | :hammer: 获取文本数据源
        type: warn
    要想实现这一功能，首先需要一段比较长的文本，这里我们选择一篇英文小说 ***[The Verdict](https://en.wikisource.org/wiki/The_Verdict)***。

    将其下载下来保存为一个文本文件`the_verdict.txt`存放在你的工程目录下。
    ///
    """)
    return


@app.cell
def _():
    with open("./data/the_verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    with mo.redirect_stdout():
        print(f"总字符数: {len(raw_text)}")
        print(f"{raw_text[:50]}...")
    return (raw_text,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// details | :hammer: 如何实现*Embedding*

    我们的任务是将这段长文本进行标记化和嵌入化操作

    首先来看什么是标记化，以下是使用正则表达式拆分一句话的过程
    ///
    """)
    return


@app.cell
def _():
    import re

    re_text = "Hello, world. This, is a test."
    re_result1 = re.split(r"(\s)", re_text)
    re_result2 = re.split(r"([,.]|\s)", re_text)

    mo.md(
        f"""
    :hammer: 简单拆分: 

    ```python
    {re_result1}
    ```

    /// attention | 问题出现了
    可以看到数组中出现了很多空格，而且符号和单词粘连在一起了！
    ///

    :hammer: 复杂拆分: 

    ```python
    {re_result2}
    ```
    """
    )
    return re, re_result2


@app.cell
def _(re_result2):
    split_result = [item for item in re_result2 if item.strip()]
    mo.md(
        f"""
    :hammer: 去除空格: 
    ```python
    {split_result}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// details | :cyclone: 已经完美了吗？

    现在看起来已经很完美了，但实际我们得到的原始文本内容会更复杂，比如下面这个例子：

    ```text
    Hello, world. Is this-- a test?
    ```
    因此我们需要更加复杂的分割方式。
    ///
    """)
    return


@app.cell
def _(re):
    text = "Hello, world. Is this-- a test?"

    split_result2 = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    list_result = [item.strip() for item in split_result2 if item.strip()]
    mo.md(
        f"""
    :hammer: 分割结果: 
    ```python
    {list_result}
    ```
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// admonition | :white_check_mark: 文本分割完成

    现在使用这个正则表达式就可以很好得对文本进行分割了。
    ///

    ![1-5](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-05.svg)

    :hammer: 接下来按照这个思路对 ***the_verdict.txt*** 进行分割
    """)
    return


@app.cell
def _(raw_text, re):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    mo.md(
        f"""
    分割后长度为 `{len(preprocessed)}`, Tokens如下：
    ```python
    {preprocessed[:10]} ...
    ```

    """
    )
    return (preprocessed,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.3 将 Tokens 转换为 Token IDs

    > 我们已经将长文本拆分成了 `Token` 序列，但是计算机仍然无法读懂这些内容，因此还需要将它们量化。

    ![1-6](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-06.svg)

    从这些 `Tokens` 中可以整理出一个词表 `vocabulary`，此表中 <u>包含了所有且不重复</u> 的 `Token`。
    """)
    return


@app.cell
def _(preprocessed):
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    vocab = {token: integer for integer, token in enumerate(all_words)}

    mo.ui.table(
        data=[{"token": token, "id": id} for token, id in vocab.items()],
        label=f"Tokens, Length={vocab_size}",
    )
    return (vocab,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    下图展示了文本样本编码为 `Token IDs` 的过程：

    ![1-7](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-07.svg)

    /// admonition | :white_check_mark: 词表构建完成

    词表已将创建完成了，接下来我们使用词表中的一小部分样本来说明文本标记化 `Tokenization`。

    ///
    """)
    return


@app.cell(hide_code=True)
def _(re, vocab):
    class SimpleTokenizerV1:
        def __init__(self, vocab):
            """Token -> TokenID：对应编码过程"""
            self.str_to_int = vocab

            """TokenID -> Token：对应解码过程"""
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            # Replace spaces before the specified punctuations
            text = re.sub(r'\s+([,.?!"()\'])', r"\1", text)
            return text


    tokenizer_v1 = SimpleTokenizerV1(vocab)
    mo.show_code()
    return (tokenizer_v1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    我们用之前的方式构建了一个名为 `SimpleTokenizerV1` 的简易版文本标记器。

    + `str_to_int` 字段保存了从 `Token` 到 `TokenID` 的映射，对应 <u>编码(`encode`)</u> 的过程
    + `int_to_str` 字段保存了从 `TokenID` 到 `Token` 的映射，对应 <u>解码(`decode`)</u> 的过程

    ![1-8](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-08.svg)

    /// details | :hammer: 我们可以用一段简单的文本对这个文本标记器进行测试。
        type: warn

    **(1/3)** 使用一段文本作为输入进行编码(`encode`)操作

    **(2/3)** 将编码后的文本进行解码(`decode`)操作

    **(3/3)** 检查解码后的文本与原文本是否一致
    ///
    """)
    return


@app.cell(hide_code=True)
def _(tokenizer_v1):
    simple_text = """
    "It's the last he painted, you know," Mrs. Gisburn said with pardonable pride.
    """
    encoded_value = tokenizer_v1.encode(simple_text)
    decoded_value = tokenizer_v1.decode(encoded_value)

    mo.md(
        f"""
    原文本：<br/>
    {simple_text}

    编码后的内容：
    ```python
    {encoded_value}
    ```

    解码后的内容：<br/>
    {decoded_value}

    /// admonition | :white_check_mark: 编-解码器(v1)创建完成

    现在我们已经完成了一个简易版 原始文本和标记文本ID转换 的工具
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    /// attention | 问题出现了
    如果输入的字符中有些字符并没有包含在词表中，就无法将其映射到 `TokenID`
    ///
    """)
    return


@app.cell
def _(tokenizer_v1):
    text_new_token = "Hello, do you like tea. Is this-- a test?"
    tokenizer_v1.encode(text_new_token)
    return (text_new_token,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.4 添加特殊的上下文标记

    > 在词表中定义一些特殊的标记文本来弥补输入的单词无法映射到 `TokenID` 的缺陷。

    ![1-9](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-09.svg)

    事实上，这些特殊标记的作用不止如此，很多模型使用特殊标记来为模型提供额外的上下文。

    /// details | :cyclone: 特殊标记的作用

    :radio_button: `[BOS]` (beginning of sequence) 作为文本开始的标记<br/>
    :radio_button: `[EOS]` (end of sequence) 作为文本结束的标记<br/>
    :radio_button: `[PAD]` (padding) 用于在训练时对样本进行长度补齐<br/>
    :radio_button: `[UNK]` (unknown) 用于表示词表中不包含的单词<br/>
    ///

    🔥 请注意，GPT-2没有使用上述任何一种，而是使用`<|endoftext|>`，既用来作为文本的结束标记，有用作训练时的长度补齐。

    我们将参考GPT-2，两个独立的文本源之间使用 `<|endoftext|>` 标记。

    ![1-10](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-10.svg)
    """)
    return


@app.cell
def _(preprocessed):
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab_v2 = {token: integer for integer, token in enumerate(all_tokens)}

    mo.ui.table(
        data=[{"token": token, "id": id} for token, id in vocab_v2.items()],
        label=f"Tokens, Length={len(vocab_v2)}",
    )
    return (vocab_v2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :hammer: 接下来让我们完善一下 `SimpleTokenizerV1`，让它可以处理词表中不存在的单词。
    """)
    return


@app.cell(hide_code=True)
def _(re, vocab_v2):
    class SimpleTokenizerV2:
        def __init__(self, vocab):
            self.str_to_int = vocab
            self.int_to_str = {i: s for s, i in vocab.items()}

        def encode(self, text):
            preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
            preprocessed = [item.strip() for item in preprocessed if item.strip()]
            preprocessed = [
                item if item in self.str_to_int else "<|unk|>"
                for item in preprocessed
            ]

            ids = [self.str_to_int[s] for s in preprocessed]
            return ids

        def decode(self, ids):
            text = " ".join([self.int_to_str[i] for i in ids])
            # Replace spaces before the specified punctuations
            text = re.sub(r'\s+([,.:;?!"()\'])', r"\1", text)
            return text


    tokenizer_v2 = SimpleTokenizerV2(vocab_v2)

    mo.show_code()
    return (tokenizer_v2,)


@app.cell
def _():
    mo.md(r"""
    :hammer: 接下来让我们测试一下新的 `SimpleTokenizerV2`，看是否可以处理此表中不存在的单词。
    """)
    return


@app.cell
def _(text_new_token, tokenizer_v2):
    encoded_values_v2 = tokenizer_v2.encode(text_new_token)
    decoded_values_v2 = tokenizer_v2.decode(encoded_values_v2)

    mo.md(
        f"""
    原文本：<br/>
    {text_new_token}

    编码后的内容：
    ```python
    {encoded_values_v2}
    ```

    解码后的内容：<br/>
    {decoded_values_v2}

    /// admonition | :white_check_mark: 编-解码器(v2)创建完成

    现在我们已经完成了一个较为完善的编解码器
    ///
    """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.5 BytePair encoding

    > `GPT-2`使用了BytePair encoding(BPE)算法进行文本标记化处理。

    :rocket: BPE算法的优势在于它能够将不存在于词表中的单词拆分为更小的单元（比如单个字符），从而使其能够处理词表之外的单词。

    + 例如，如果 `GPT-2` 的词汇表中没有 `"unfamiliarword"` 这个词，它可能会将其标记为 `["unfam"、"iliar"、"word"]` 或其他子词分解，具体取决于其训练过的 BPE 合并过程。

    + :link: [`GPT-2`的编码器源码](https://github.com/openai/gpt-2/blob/master/src/encoder.py)可以从这里获取！

    接下来我们将使用 `OpenAI` 的开源库 `tiktoken` 中的 `BPE` 分词器，该库使用 `Rust` 实现，效率非常高！可以使用以下命令安装。

    ```shell
    pip install tiktoken
    ```
    """)
    return


@app.cell
def _():
    import importlib
    import tiktoken

    with mo.redirect_stdout():
        print("tiktoken version:", importlib.metadata.version("tiktoken"))
    return (tiktoken,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :hammer: 我们来看一下BPE的执行效果

    :rocket: BPE算法会将未知的单词拆分为更小的单位（若干个子单词或者单个字符）

    ![1-11](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-11.svg)
    """)
    return


@app.cell
def _(tiktoken):
    text_gpt_test = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
        "of someunknownPlace."
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    # 这里配置允许使用的特殊标记
    gpt_token_ids = tokenizer.encode(
        text_gpt_test, allowed_special={"<|endoftext|>"}
    )


    gpt_decoded_text = tokenizer.decode(gpt_token_ids)

    mo.md(
        f"""
    :tada: `GPT-2` 的编码器

    原始文本：
    ```text
    {text_gpt_test}
    ```

    编码结果：
    ```python
    {gpt_token_ids}
    ```

    解码结果：
    ```text
    {gpt_decoded_text}
    ```

    /// admonition | :white_check_mark: 探索 `GPT-2` 的编码器

    现在我们已经初步了解了 `GPT-2` 的编码器的基本工作原理
    ///
    """
    )
    return (tokenizer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.6 BPE的训练过程

    > 字节对编码（Byte Pair Encoding, BPE）是一种用于文本处理的压缩算法，广泛应用于自然语言处理（NLP）中的词汇构建。

    ///details | :rocket: 基本过程如下：
        type: warn
    :radio_button: **(1/4)** 把文本切分为单词，并在单词末尾加特殊标记 `</w>`。<br/>
    :radio_button: **(2/4)** 统计所有bigram出现频率。<br/>
    :radio_button: **(3/4)** 执行合并。<br/>
    :radio_button: **(4/4)** 重复步骤2、3直到达到所需的词汇大小或无法再找到频率高的字符对为止。<br/>
    ///

    :page_with_curl: 假如现有如下文本样本：

    ```shell
    # 10个
    hug hug hug hug hug hug hug hug hug hug
    # 5个
    pug pug pug pug pug
    # 12个
    pun pun pun pun pun pun pun pun pun pun pun pun
    # 4个
    bun bun bun bun
    # 5个
    hugs hugs hugs hugs hugs
    ```

    :hammer: 首先需要将他们拆分为单个字符
    """)
    return


@app.cell
def _(re):
    from collections import Counter

    bpe_text_sample = """
    hug hug hug hug hug hug hug hug hug hug
    pug pug pug pug pug
    pun pun pun pun pun pun pun pun pun pun pun pun
    bun bun bun bun
    hugs hugs hugs hugs hugs
    """

    bpe_words = re.findall(r"\w+|[^\w\s]", bpe_text_sample)
    bpe_corpus = [" ".join(list(w)) + " </w>" for w in bpe_words]

    bpe_word_freqs = Counter(bpe_corpus)

    mo.md(
        f"""
    :tada: 字符列表：

    + `bpe_word_freqs`
    ```python
    {bpe_word_freqs} ...
    ```
    """
    )
    return Counter, bpe_word_freqs


@app.cell
def _(bpe_word_freqs, re):
    from collections import defaultdict


    def get_stats(word_freqs):
        """统计所有bigram出现频率"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs


    def merge_vocab(pair, v_in):
        """执行一次合并"""
        v_out = {}
        bigram = re.escape(" ".join(pair))
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in v_in:
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out


    bpe_vocab = bpe_word_freqs.copy()

    bpe_first_pairs = get_stats(bpe_vocab)
    bpe_first_best = max(bpe_first_pairs, key=bpe_first_pairs.get)

    mo.md(
        f"""
    :hammer: 计相邻两个字符出现的频率

    ---

    合并前：
    ```python
    {dict(bpe_vocab)}
    ```


    本次合并项为：`{bpe_first_best}`, 出现频率为：`{bpe_first_pairs[bpe_first_best]}`

    ---

    :tada: 字符频率统计结果：
    ```python
    {merge_vocab(bpe_first_best, bpe_vocab)}
    ```
    """
    )
    return get_stats, merge_vocab


@app.cell(hide_code=True)
def _(Counter, bpe_word_freqs, get_stats, merge_vocab):
    get_merge_table, set_merge_table = mo.state([])
    get_merge_entry, set_merge_entry = mo.state("")


    def bpe_merge(count: any, vocab: Counter):
        if type(count) != int:
            return []
        for t in range(count):
            pairs = get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)  # 选择出现次数最多的bigram
            set_merge_entry(
                f"""本次合并项为：`{best}`, 出现频率为：`{pairs[best]}`"""
            )
            vocab = merge_vocab(best, vocab)
        return vocab


    bpe_merge_slider = mo.ui.slider(
        start=0,
        stop=20,
        step=1,
        show_value=True,
        label="合并次数",
        on_change=lambda value: set_merge_table(
            bpe_merge(value, bpe_word_freqs.copy())
        ),
    )
    return bpe_merge_slider, get_merge_entry, get_merge_table


@app.cell(hide_code=True)
def _(bpe_merge_slider, get_merge_entry, get_merge_table):
    mo.vstack(
        [
            bpe_merge_slider,
            mo.md(get_merge_entry()),
            mo.ui.table(data=get_merge_table()),
            mo.md(
                f"""
    /// admonition | :white_check_mark: BPE算法

    现在，你已经完全掌握了BPE算法的训练过程！
            """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.7 滑动窗口实现数据采样

    > 我们训练 LLM 一次生成一个单词，因此我们希望相应地准备训练数据，将序列中的下一个单词作为预测的目标。

    ![1-12](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-12.svg)
    """)
    return


@app.cell
def _(raw_text, tokenizer):
    encoded_text = tokenizer.encode(raw_text)

    mo.md(
        f"""
    编码后的词表大小：`{len(encoded_text)}`

    词表：
    ```python
    {encoded_text[:10]} ...
    ```
    """
    )
    return (encoded_text,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ///details | :fire: 如何处理输入文本？

    :radio_button: 首先，每次输入给模型的文本块都需要有两部分（可以参考的字符序列，需要预测的目标单词）<br/>
    :radio_button: 因为我们希望模型预测下一个单词，所以目标是向右移动一个位置的输入
    ///

    :hammer: 我们假设窗口大小为4，已输入文本的前10个单词举例：
    """)
    return


@app.cell(hide_code=True)
def _(encoded_text):
    # 用于演示的文本序列
    encoded_sample = encoded_text[:10]
    # 用于演示的模型窗口大小
    context_size = 4

    mo.show_code()
    return (encoded_sample,)


@app.cell(hide_code=True)
def _(encoded_sample):
    get_refer_sequence, set_refer_sequence = mo.state([])
    get_predict_words, set_predict_words = mo.state([])


    def set_llm_window(value: any):
        if type(value) != int:
            return
        if value > 3:
            set_refer_sequence(encoded_sample[value - 4 : value])
        else:
            set_refer_sequence(encoded_sample[:value])
        set_predict_words(encoded_sample[value : value + 1])


    window_slider = mo.ui.slider(
        start=1,
        stop=9,
        step=1,
        show_value=True,
        label="窗口位置",
        on_change=set_llm_window,
    )
    return get_predict_words, get_refer_sequence, window_slider


@app.cell(hide_code=True)
def _(encoded_sample, get_predict_words, get_refer_sequence, window_slider):
    mo.vstack(
        [
            window_slider,
            mo.md(
                f"""
    用于训练的样本文本块
    ```python
    {encoded_sample}
    ```
    参考字符序列：
    ```python
    {get_refer_sequence()}
    ```
    需要预测的字符：
    ```python
    {get_predict_words()}
    ```

    /// admonition | :white_check_mark: 模型的上下文

    现在，你已经完初步了解了大模型上下文(`Context`)的基本概念
    """
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.8 构建训练数据集

    > 参考滑动窗口的工作原理，创建数据集和数据加载器，从输入文本数据集中提取块

    ![1-13](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-13.svg)

    /// attention | 小提示
    从这里开始，我们需要用到 `pytorch`，可以使用以下命令安装

    ```shell
    pip install pytorch
    ```
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    import torch
    from torch.utils.data import Dataset, DataLoader


    class GPTDatasetV1(Dataset):
        """
        txt: 文本数据
        tokenizer: 分词器
        max_length: 最大长度
        stride: 前进步幅
        """

        def __init__(self, txt, tokenizer, max_length, stride):
            self.input_ids = []
            self.target_ids = []

            # Tokenize the entire text
            token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
            assert len(token_ids) > max_length, (
                "标记化输入的数量必须至少等于 max_length+1"
            )

            # Use a sliding window to chunk the book into overlapping sequences of max_length
            for i in range(0, len(token_ids) - max_length, stride):
                input_chunk = token_ids[i : i + max_length]
                target_chunk = token_ids[i + 1 : i + max_length + 1]
                self.input_ids.append(torch.tensor(input_chunk))
                self.target_ids.append(torch.tensor(target_chunk))

        def __len__(self):
            return len(self.input_ids)

        def __getitem__(self, idx):
            return self.input_ids[idx], self.target_ids[idx]


    mo.show_code()
    return DataLoader, GPTDatasetV1, torch


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :hammer: 然后再创建一个数据集加载器
    """)
    return


@app.cell(hide_code=True)
def _(DataLoader, GPTDatasetV1, tiktoken):
    def create_dataloader_v1(
        txt,
        batch_size=4,
        max_length=256,
        stride=128,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    ):
        # Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")

        # Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
        )

        return dataloader


    mo.show_code()
    return (create_dataloader_v1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :hammer: 接下来让我们测试以下新建的数据集加载器的运行效果
    """)
    return


@app.cell
def _(create_dataloader_v1, raw_text):
    dataloader_v1 = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    dataset_v1_table = []
    for batch in dataloader_v1:
        dataset_v1_table.append(
            {
                "Input TokenIDs": batch[0].numpy(),
                "Input Shape": batch[0].shape,
                "Target TokenIDs": batch[1].numpy(),
                "Target Shape": batch[1].shape,
            }
        )

    mo.ui.table(data=dataset_v1_table)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    使用步幅等于上下文长度（此处为 4）的示例如下所示：

    ![1-14](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-14.svg)

    我们还可以创建批量输出

    /// attention | 小提示
    请注意，我们在这里增加了步幅，这样批次之间就不会出现重叠，因为更多的重叠可能会导致过度拟合
    ///

    /// admonition | :white_check_mark: 训练数据集

    现在，你已经掌握了如何使用 `pytorch` 构建大语言模型的训练数据集
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.9 创建词嵌入

    > 词嵌入就是 `Token Embeddings`。数据几乎已经为 LLM 做好了准备，但最后让我们使用嵌入层将标记嵌入到连续向量表示中，通常，这些嵌入层是 LLM 本身的一部分，并在模型训练期间进行更新（训练）

    ![1-15](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-15.svg)

    :rocket: 为了简单起见，假设我们的假设我们有以下四个输入示例，输入 ID 分别为 2、3、5 和 1（标记化后），词汇量只有 `6` 个单词，并且我们想要创建大小为 `3` 的嵌入：
    """)
    return


@app.cell
def _(torch):
    import torch.nn as nn

    emedding_input_ids = torch.tensor([2, 3, 5, 1])
    example_vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    example_embedding_layer = nn.Embedding(example_vocab_size, output_dim)

    mo.md(
        f"""
    Embedding的权重如下：
    ```python
    {example_embedding_layer.weight}
    ```
    """
    )
    return emedding_input_ids, example_embedding_layer, nn


@app.cell(hide_code=True)
def _(example_embedding_layer, torch):
    mo.md(f"""
    :hammer: 要将 ID 为 3 的标记转换为三维向量，我们执行以下操作：

    ```python
    example_embedding_layer(torch.tensor([3]))
    example_embedding_layer(torch.tensor([5]))
    ```

    输出结果如下：

    ```python
    3 = {example_embedding_layer(torch.tensor([3]))}
    5 = {example_embedding_layer(torch.tensor([5]))}
    ```
    """)
    return


@app.cell
def _(emedding_input_ids, example_embedding_layer):
    mo.md(f"""
    :hammer: 参考以上过程，把 `emedding_input_ids` 全部转换为张量：

    ```python
    example_embedding_layer(emedding_input_ids))
    ```

    输出结果如下：

    ```python
    {example_embedding_layer(emedding_input_ids)}
    ```
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :fire: 嵌入层本质其实是一种查询操作

    ![1-16](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-16.svg)

    :rocket: 嵌入的过程就是根据 `Token ID` 从权重矩阵查找出相应索引对应的 `权重`。

    /// admonition | :white_check_mark: 词嵌入

    现在，你已经完全掌握了词嵌入的基本原理。
    ///
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.10 词的位置编码

    > 嵌入层将 `Token ID` 转换为相同的向量表示，这一操作将会忽视文本序列的输入顺序。

    ![1-17](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-17.svg)

    :rocket: 位置嵌入与标记嵌入向量相结合，形成大型语言模型的输入嵌入：

    ![1-18](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-18.svg)

    假设 `BPE` 的词表大小为 `50257`，输入的张量维度为 `256` 维（即用 `256` 个特征量表示一个 `Token`）
    """)
    return


@app.cell(hide_code=True)
def _(torch):
    token_vocab_size = 50257
    token_output_dim = 256

    token_embedding_layer = torch.nn.Embedding(token_vocab_size, token_output_dim)

    mo.show_code()
    return token_embedding_layer, token_output_dim


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :fire: 如果我们从数据加载器中采样数据，我们会将每个批次中的标记嵌入到一个 256 维向量中。 假设批次大小为 8，每个批次包含 4 个标记，则会产生一个 $8 \times {4} \times {256}$ 的张量：
    """)
    return


@app.cell
def _(create_dataloader_v1, raw_text):
    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text,
        batch_size=8,
        max_length=max_length,
        stride=max_length,
        shuffle=False,
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)

    mo.md(
        f"""
    Token IDs:
    ```python
    {inputs.numpy()}
    ```

    Inputs shape:
    ```python
    {inputs.shape}
    ```
    """
    )
    return inputs, max_length


@app.cell
def _(inputs, token_embedding_layer):
    token_embeddings = token_embedding_layer(inputs)
    with mo.redirect_stdout():
        print(token_embeddings.shape)
    return (token_embeddings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :rocket: `GPT-2` 使用绝对位置嵌入，因此我们只需创建另一个嵌入层：
    """)
    return


@app.cell
def _(max_length, nn, token_output_dim, torch):
    context_length = max_length
    pos_embedding_layer = nn.Embedding(context_length, token_output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    with mo.redirect_stdout():
        print(pos_embeddings.shape)
    return (pos_embeddings,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :rocket: 要创建 LLM 中使用的输入嵌入，我们只需添加标记和位置嵌入：
    """)
    return


@app.cell
def _(pos_embeddings, token_embeddings):
    input_embeddings = token_embeddings + pos_embeddings
    with mo.redirect_stdout():
        print(input_embeddings.shape)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    :rocket: 在输入处理工作流程的初始阶段，输入文本会被分割成单独的标记。 分割完成后，这些标记会根据预定义的词汇表转换为标记 ID：

    ![1-19](https://codingsoul-images.tos-cn-beijing.volces.com/LLM/1-19.svg)

    /// admonition | :white_check_mark: 模型输入层（`Token Embedding` 和 `Positional Emcoding`）

    现在，你已经完全掌握了 `Transformer` 模型输入层的基本原理
    ///
    """)
    return


if __name__ == "__main__":
    app.run()
