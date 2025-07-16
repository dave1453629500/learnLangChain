import re
import requests
import argparse
import time
import logging
import concurrent.futures
from typing import List, Tuple, Dict, Optional
import os

class OllamaMarkdownTranslator:
    """
    使用Ollama API翻译Markdown文档的类
    支持保留格式、代码块和其他Markdown元素
    """
    def __init__(
        self, 
        ollama_url: str = "http://localhost:11434", 
        model: str = "llama3:latest",
        source_lang: str = "英文", 
        target_lang: str = "中文",
        chunk_size: int = 1500,
        max_workers: int = 3,
        max_retries: int = 3,
        retry_delay: int = 2
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.chunk_size = chunk_size
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("OllamaMarkdownTranslator")
        
        # 检查Ollama服务器是否可用
        self._check_ollama_server()

    def _check_ollama_server(self):
        """检查Ollama服务器是否可用"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code != 200:
                self.logger.warning(f"Ollama服务器状态异常: {response.status_code}")
            else:
                models = response.json().get("models", [])
                model_names = [m.get("name") for m in models]
                self.logger.info(f"Ollama服务器可用，已加载模型: {model_names}")
                
                # 检查所需模型是否已加载
                model_base = self.model.split(":")[0] if ":" in self.model else self.model
                if not any(model_base in m for m in model_names):
                    self.logger.warning(f"所需模型 '{self.model}' 可能未加载，请确保已通过 'ollama pull {self.model}' 下载")
                
        except Exception as e:
            self.logger.error(f"无法连接到Ollama服务器: {str(e)}")
            self.logger.error(f"请确保Ollama服务器正在运行，并可通过 {self.ollama_url} 访问")

    def extract_markdown_elements(self, content: str) -> List[Tuple[str, str, int]]:
        """
        提取Markdown内容中的不同元素
        返回: List of (type, content, position)
        """
        elements = []
        
        # 查找代码块 (```code```)
        code_pattern = r'```[^\n]*\n(.*?)```'
        for match in re.finditer(code_pattern, content, re.DOTALL):
            start, end = match.span()
            elements.append(('code', match.group(), start))
        
        # 查找HTML标签和内联代码
        html_inline_code_pattern = r'(<[^>]+>|`[^`]+`)'
        for match in re.finditer(html_inline_code_pattern, content):
            start, end = match.span()
            elements.append(('html_or_inline_code', match.group(), start))
        
        # 查找链接和图片
        link_pattern = r'(!?\[.*?\]\(.*?\))'
        for match in re.finditer(link_pattern, content):
            start, end = match.span()
            elements.append(('link', match.group(), start))
        
        # 创建一个掩码，标记哪些部分已经处理过
        mask = [False] * len(content)
        for _, element_content, position in elements:
            for i in range(position, position + len(element_content)):
                if i < len(mask):
                    mask[i] = True
        
        # 处理剩余的文本部分
        text_blocks = []
        current_text = ""
        text_start = 0
        
        for i, char in enumerate(content):
            if mask[i]:
                if current_text:
                    text_blocks.append((text_start, current_text))
                    current_text = ""
            else:
                if not current_text:
                    text_start = i
                current_text += char
        
        # 添加最后一个文本块
        if current_text:
            text_blocks.append((text_start, current_text))
        
        # 将文本块添加到元素列表
        for start, text in text_blocks:
            elements.append(('text', text, start))
        
        # 按位置排序所有元素
        elements.sort(key=lambda x: x[2])
        
        return elements

    def split_text_into_chunks(self, text: str) -> List[str]:
        """将文本分割成较小的块，尽量在句子边界分割"""
        # 使用句号、问号、感叹号和换行符作为分割点
        sentences = re.split(r'([.!?。！？\n])', text)
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

    def translate_chunk_with_retry(self, chunk: str) -> str:
        """使用Ollama API翻译文本块，带重试机制"""
        prompt = f"""请将以下{self.source_lang}文本翻译成{self.target_lang}。要求：
1. 保持专业术语的准确性
2. 保持原文的语气和风格
3. 保持Markdown格式符号不变（如 #, *, -, 等）
4. 保持链接和图片的格式不变
5. 保持代码块和行内代码不变
6. 保持列表的层级结构不变
7. 确保翻译后的{self.target_lang}通顺、自然

{self.source_lang}原文：
{chunk}

{self.target_lang}翻译："""

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # 低温度以保持翻译准确
                            "top_p": 0.9
                        }
                    },
                    timeout=60  # 设置更长的超时时间
                )
                
                if response.status_code == 200:
                    translated_text = response.json().get("response", "").strip()
                    return translated_text
                else:
                    self.logger.warning(f"翻译失败，状态码 {response.status_code}，尝试 {attempt+1}/{self.max_retries}")
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (attempt + 1))  # 增加延迟时间
                    else:
                        raise Exception(f"翻译失败: {response.text}")
                    
            except Exception as e:
                self.logger.warning(f"尝试 {attempt+1} 失败: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    self.logger.error(f"翻译块失败，将返回原文")
                    return chunk

    def translate_text(self, text: str) -> str:
        """并发翻译多个文本块"""
        if not text.strip():
            return text
            
        chunks = self.split_text_into_chunks(text)
        if len(chunks) == 0:
            return text
            
        translated_chunks = [""] * len(chunks)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有翻译任务
            future_to_index = {
                executor.submit(self.translate_chunk_with_retry, chunk): i 
                for i, chunk in enumerate(chunks)
            }
            
            # 收集结果
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    translated_chunks[index] = future.result()
                except Exception as e:
                    self.logger.error(f"翻译第 {index+1} 块时出错: {str(e)}")
                    translated_chunks[index] = chunks[index]  # 失败时保留原文
                
                # 输出进度
                completed = sum(1 for tc in translated_chunks if tc)
                self.logger.info(f"翻译进度: {completed}/{len(chunks)} 块")
        
        return " ".join(translated_chunks)

    def translate_markdown(self, content: str) -> str:
        """翻译整个Markdown文档，保持格式不变"""
        elements = self.extract_markdown_elements(content)
        translated_content = content
        
        self.logger.info(f"从文档中提取了 {len(elements)} 个元素")
        
        # 记录需要翻译的文本元素数量
        text_elements = [e for e in elements if e[0] == 'text']
        self.logger.info(f"需要翻译的文本元素: {len(text_elements)} 个")
        
        # 从后向前替换，这样不会影响前面内容的位置
        translated_count = 0
        for element_type, element_content, position in reversed(elements):
            if element_type == 'text':
                self.logger.info(f"翻译第 {translated_count+1}/{len(text_elements)} 个文本元素")
                
                # 只翻译非空文本
                if element_content.strip():
                    translated_text = self.translate_text(element_content)
                    
                    # 替换原文
                    translated_content = (
                        translated_content[:position] +
                        translated_text +
                        translated_content[position + len(element_content):]
                    )
                
                translated_count += 1
                
        self.logger.info(f"翻译完成！共翻译了 {translated_count} 个文本元素")
        return translated_content

def main():
    """主函数：解析命令行参数并执行翻译"""
    parser = argparse.ArgumentParser(description='使用Ollama API翻译Markdown文档')
    parser.add_argument('--input', '-i', type=str, required=True, help='输入Markdown文件路径')
    parser.add_argument('--output', '-o', type=str, help='输出Markdown文件路径（默认为input文件名+_translated.md）')
    parser.add_argument('--model', '-m', type=str, default='llama3:latest', help='Ollama模型名称（默认为llama3:latest）')
    parser.add_argument('--url', '-u', type=str, default='http://localhost:11434', help='Ollama服务器URL（默认为http://localhost:11434）')
    parser.add_argument('--source', '-s', type=str, default='英文', help='源语言（默认为英文）')
    parser.add_argument('--target', '-t', type=str, default='中文', help='目标语言（默认为中文）')
    parser.add_argument('--chunk-size', '-c', type=int, default=1500, help='文本块大小（默认为1500字符）')
    parser.add_argument('--workers', '-w', type=int, default=3, help='并发工作线程数（默认为3）')
    parser.add_argument('--retries', '-r', type=int, default=3, help='最大重试次数（默认为3）')
    
    args = parser.parse_args()
    
    input_file = args.input
    
    # 如果未指定输出文件，创建默认输出文件名
    if args.output:
        output_file = args.output
    else:
        input_base = os.path.splitext(input_file)[0]
        output_file = f"{input_base}_translated.md"
    
    # 创建翻译器实例
    translator = OllamaMarkdownTranslator(
        ollama_url=args.url,
        model=args.model,
        source_lang=args.source,
        target_lang=args.target,
        chunk_size=args.chunk_size,
        max_workers=args.workers,
        max_retries=args.retries
    )
    
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 翻译内容
        translated_content = translator.translate_markdown(content)
        
        # 写入输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
            
        logging.info(f"翻译完成！结果已保存到 {output_file}")
        
    except Exception as e:
        logging.error(f"翻译过程中发生错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 