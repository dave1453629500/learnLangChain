import re
import requests
from typing import List, Tuple
import argparse
import time

class MarkdownTranslator:
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.model = "llama3:latest"  # 使用 llama2 模型
        self.chunk_size = 1000  # 每个翻译块的大小
        self.max_retries = 3    # 最大重试次数
        self.retry_delay = 2    # 重试延迟（秒）

    def _extract_markdown_blocks(self, content: str) -> List[Tuple[str, str, int]]:
        """
        提取Markdown内容中的代码块和普通文本
        返回: List of (type, content, position)
        type可以是 'code', 'text', 'header', 'list', 'link', 'image' 等
        """
        # 首先提取代码块，因为它们需要特殊处理
        code_blocks = []
        code_pattern = r'```[^\n]*\n(.*?)```'
        for match in re.finditer(code_pattern, content, re.DOTALL):
            start, end = match.span()
            code_blocks.append(('code', match.group(), start))
        
        # 创建一个掩码，标记哪些部分是代码块
        mask = [False] * len(content)
        for _, block_content, position in code_blocks:
            for i in range(position, position + len(block_content)):
                if i < len(mask):
                    mask[i] = True
        
        # 现在处理非代码块部分
        blocks = list(code_blocks)  # 复制代码块列表
        
        # 按行分割内容
        lines = content.split('\n')
        current_pos = 0
        current_text = ""
        text_start_pos = 0
        
        for line in lines:
            line_with_newline = line + '\n'
            line_length = len(line_with_newline)
            
            # 检查这一行是否在代码块内
            if any(mask[current_pos:current_pos + line_length]):
                # 如果之前有文本，添加它
                if current_text:
                    blocks.append(('text', current_text, text_start_pos))
                    current_text = ""
            else:
                # 检查是否是标题行
                if re.match(r'^#{1,6}\s+', line):
                    # 如果之前有文本，添加它
                    if current_text:
                        blocks.append(('text', current_text, text_start_pos))
                        current_text = ""
                    blocks.append(('header', line_with_newline, current_pos))
                # 检查是否是列表项
                elif re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                    # 如果之前有文本，添加它
                    if current_text:
                        blocks.append(('text', current_text, text_start_pos))
                        current_text = ""
                    blocks.append(('list', line_with_newline, current_pos))
                else:
                    # 普通文本行
                    if not current_text:
                        text_start_pos = current_pos
                    current_text += line_with_newline
            
            current_pos += line_length
        
        # 添加最后的文本块（如果有）
        if current_text:
            blocks.append(('text', current_text, text_start_pos))
        
        # 按位置排序
        return sorted(blocks, key=lambda x: x[2])

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """
        将文本分割成较小的块，尽量在句子边界分割
        """
        # 使用句号、问号、感叹号作为分割点
        sentences = re.split(r'([.!?。！？])', text)
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
            if len(current_chunk) + len(sentence) > self.chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk += sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _translate_text_with_retry(self, text: str) -> str:
        """
        使用Ollama API翻译文本，带重试机制
        """
        prompt = f"""请将以下英文文本翻译成中文。要求：
1. 保持专业术语的准确性
2. 保持原文的语气和风格
3. 保持Markdown格式符号不变（如 #, *, -, 等）
4. 保持链接和图片的格式不变
5. 保持代码块和行内代码不变
6. 保持列表的层级结构不变
7. 确保翻译后的中文通顺、自然

英文原文：
{text}"""

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=30  # 设置超时时间
                )
                
                if response.status_code == 200:
                    return response.json()["response"].strip()
                else:
                    print(f"Translation failed with status code {response.status_code}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    raise Exception(f"Translation failed: {response.text}")
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                    continue
                raise

    def _translate_text(self, text: str) -> str:
        """
        翻译文本，支持分块处理
        """
        chunks = self._split_text_into_chunks(text)
        translated_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Translating chunk {i}/{len(chunks)}...")
            translated_chunk = self._translate_text_with_retry(chunk)
            translated_chunks.append(translated_chunk)
            time.sleep(1)  # 添加短暂延迟，避免请求过快
        
        return " ".join(translated_chunks)

    def translate_markdown(self, content: str) -> str:
        """
        翻译整个Markdown文档，保持格式不变
        """
        blocks = self._extract_markdown_blocks(content)
        translated_content = content
        
        print(f"Found {len(blocks)} blocks in the document")
        for i, (block_type, block_content, position) in enumerate(blocks):
            print(f"Block {i+1}: Type={block_type}, Position={position}, Length={len(block_content)}")
            if i < 3:  # 只打印前3个块的内容预览
                preview = block_content[:100] + "..." if len(block_content) > 100 else block_content
                print(f"Content preview: {preview}")
        
        # 从后向前替换，这样不会影响前面内容的位置
        translated_blocks_count = 0
        for block_type, block_content, position in reversed(blocks):
            # 翻译文本块和标题块
            if block_type in ['text', 'header', 'list']:
                print(f"\nTranslating {block_type} block at position {position}...")
                translated_text = self._translate_text(block_content)
                print(f"Original: {block_content[:50]}...")
                print(f"Translated: {translated_text[:50]}...")
                translated_content = (
                    translated_content[:position] +
                    translated_text +
                    translated_content[position + len(block_content):]
                )
                translated_blocks_count += 1
        
        print(f"Translated {translated_blocks_count} blocks out of {len(blocks)} total blocks")
        return translated_content

def main():
    # 直接设置输入输出文件路径
    input_file = "/home/dave/PycharmProjects/learnLangChain/output/VAE.md"
    output_file = "/home/dave/PycharmProjects/learnLangChain/abcdefg.md"
    
    # 创建翻译器实例
    translator = MarkdownTranslator()
    translator.model = "llama3:latest"  # 使用 llama3:latest 模型
    
    try:
        print(f"Reading input file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("Starting translation...")
        translated_content = translator.translate_markdown(content)
        
        print(f"Writing output to: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
            
        print("Translation completed successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 