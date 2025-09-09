#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
维基百科数据集创建脚本
将WikiExtractor提取的维基百科文章转换为JSONL格式并上传到ModelScope

下载链接
https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2

# 安装依赖
pip install modelscope tqdm

# 运行数据转换脚本
python scripts/create_dataset.py \
    --repo_name "<repo>/zhwiki-dataset" \
    --output_file "zhwiki_dataset.jsonl"
"""

import os
import json
import re
import argparse
from pathlib import Path
from typing import Iterator, Dict, Any
import logging
from tqdm import tqdm
from modelscope.hub.api import HubApi

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_wiki_file(file_path: str) -> Iterator[Dict[str, Any]]:
    """
    解析单个维基百科文件，提取文档信息

    Args:
        file_path: 文件路径

    Yields:
        包含文档信息的字典
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 使用正则表达式匹配所有doc标签
        doc_pattern = r'<doc id="([^"]+)" url="([^"]+)" title="([^"]+)">\n(.*?)\n</doc>'
        matches = re.findall(doc_pattern, content, re.DOTALL)

        # 如果没有找到匹配，尝试更宽松的模式
        if not matches:
            doc_pattern = r'<doc id="([^"]+)" url="([^"]+)" title="([^"]+)">(.*?)</doc>'
            matches = re.findall(doc_pattern, content, re.DOTALL)

        for doc_id, url, title, text in matches:
            # 清理文本内容
            text = text.strip()
            if not text:
                continue

            yield {
                "id": doc_id,
                "url": url,
                "title": title,
                "content": text,
                "source": "zhwiki"
            }

    except Exception as e:
        logger.error(f"解析文件 {file_path} 时出错: {e}")


def process_wiki_directory(wiki_dir: str) -> Iterator[Dict[str, Any]]:
    """
    处理整个维基百科目录，遍历所有文件

    Args:
        wiki_dir: 维基百科数据目录路径

    Yields:
        包含文档信息的字典
    """
    wiki_path = Path(wiki_dir)

    if not wiki_path.exists():
        raise FileNotFoundError(f"维基百科目录不存在: {wiki_dir}")

    # 获取所有wiki文件
    wiki_files = []

    # 检查当前目录是否包含wiki文件
    current_wiki_files = list(wiki_path.glob("wiki_*"))
    if current_wiki_files:
        wiki_files.extend(current_wiki_files)
    else:
        # 如果当前目录没有wiki文件，查找子目录
        for subdir in wiki_path.iterdir():
            if subdir.is_dir():
                wiki_files.extend(subdir.glob("wiki_*"))

    logger.info(f"找到 {len(wiki_files)} 个维基百科文件")

    # 处理每个文件
    for file_path in tqdm(wiki_files, desc="处理维基百科文件"):
        for doc in parse_wiki_file(str(file_path)):
            yield doc


def save_to_jsonl(data_iterator: Iterator[Dict[str, Any]], output_file: str):
    """
    将数据保存为JSONL格式

    Args:
        data_iterator: 数据迭代器
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(data_iterator, desc="保存JSONL文件"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def validate_jsonl_file(file_path: str) -> bool:
    """
    验证JSONL文件的完整性

    Args:
        file_path: JSONL文件路径

    Returns:
        文件是否有效
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:  # 只检查前10行
                    break
                line = line.strip()
                if line:
                    json.loads(line)
        return True
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.error(f"JSONL文件验证失败: {e}")
        return False


def upload_to_modelscope(jsonl_file: str, repo_name: str, token: str = None):
    """
    上传JSONL文件到ModelScope

    Args:
        jsonl_file: JSONL文件路径
        repo_name: ModelScope仓库名称
        token: ModelScope访问令牌
    """
    # 获取绝对路径
    abs_path = os.path.abspath(jsonl_file)

    api = HubApi()
    api.login(access_token=token)

    api.upload_file(
        repo_id=repo_name,
        path_or_fileobj=abs_path,
        path_in_repo=os.path.basename(jsonl_file),  # 只使用文件名作为仓库中的路径
        repo_type='dataset',
        commit_message='upload dataset file to repo'
    )
    logger.info(f"上传成功: {jsonl_file} 到 {repo_name}")


def main():
    parser = argparse.ArgumentParser(description="将维基百科文章转换为JSONL格式并上传到ModelScope")
    parser.add_argument(
        "--wiki_dir",
        type=str,
        default="../data/zhwiki",
        help="维基百科数据目录路径"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="zhwiki_dataset.jsonl",
        help="输出JSONL文件路径"
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        required=True,
        help="ModelScope仓库名称 (格式: username/dataset-name)"
    )
    parser.add_argument(
        "--token",
        type=str,
        help="ModelScope访问令牌"
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="是否上传到ModelScope"
    )
    parser.add_argument(
        "--force_recreate",
        action="store_true",
        help="强制重新创建JSONL文件，即使文件已存在"
    )

    args = parser.parse_args()

    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    wiki_dir = script_dir / args.wiki_dir
    output_file = script_dir / args.output_file

    logger.info(f"维基百科目录: {wiki_dir}")
    logger.info(f"输出文件: {output_file}")

    try:
        # 检查JSONL文件是否已存在
        if output_file.exists() and not args.force_recreate:
            logger.info(f"JSONL文件已存在: {output_file}")
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"文件大小: {file_size:.2f} MB")

            # 验证文件完整性
            if validate_jsonl_file(str(output_file)):
                # 统计文件行数
                with open(output_file, 'r', encoding='utf-8') as f:
                    line_count = sum(1 for _ in f)
                logger.info(f"文件包含 {line_count:,} 个文档")
                logger.info("文件验证通过，跳过创建步骤")
            else:
                logger.warning("文件验证失败，将重新创建")
                args.force_recreate = True

        if not output_file.exists() or args.force_recreate:
            # 处理维基百科数据
            if output_file.exists():
                logger.info(f"强制重新创建JSONL文件: {output_file}")
            else:
                logger.info("开始处理维基百科数据...")

            data_iterator = process_wiki_directory(str(wiki_dir))

            # 保存为JSONL格式
            logger.info("保存为JSONL格式...")
            save_to_jsonl(data_iterator, str(output_file))

            # 统计文件大小
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"JSONL文件已保存: {output_file} ({file_size:.2f} MB)")

        # 上传到ModelScope
        if args.upload:
            upload_to_modelscope(str(output_file), args.repo_name, args.token)

        logger.info("处理完成!")

    except Exception as e:
        logger.error(f"处理过程中出错: {e}")
        raise


if __name__ == "__main__":
    main()