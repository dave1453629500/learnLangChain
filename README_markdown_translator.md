# Ollama Markdown 翻译器

这是一个使用Ollama API翻译Markdown文档的工具，它能够保留Markdown的格式，并提供多种配置选项。

## 特性

- 使用Ollama API进行文本翻译
- 保留Markdown格式（标题、列表、链接、图片、代码块等）
- 支持并发翻译，加快处理速度
- 内置重试机制，提高稳定性
- 可自定义源语言和目标语言
- 详细的日志输出，方便调试和监控

## 前提条件

1. 安装并运行[Ollama](https://ollama.ai/)服务
2. 拉取所需的模型，例如：`ollama pull llama3`

## 安装

```bash
# 克隆仓库
git clone <repository-url>
cd <repository-directory>

# 安装依赖
pip install -r requirements_translator.txt
```

## 使用方法

### 基本使用

```bash
python ollama_markdown_translator.py --input your_file.md
```

这将使用默认设置翻译文档，并将结果保存到`your_file_translated.md`。

### 完整参数

```bash
python ollama_markdown_translator.py --input your_file.md --output translated.md --model llama3:latest --url http://localhost:11434 --source 英文 --target 中文 --chunk-size 1500 --workers 3 --retries 3
```

### 参数说明

- `--input`, `-i`: 输入Markdown文件路径（必需）
- `--output`, `-o`: 输出Markdown文件路径（默认为input文件名+_translated.md）
- `--model`, `-m`: Ollama模型名称（默认为llama3:latest）
- `--url`, `-u`: Ollama服务器URL（默认为http://localhost:11434）
- `--source`, `-s`: 源语言（默认为英文）
- `--target`, `-t`: 目标语言（默认为中文）
- `--chunk-size`, `-c`: 文本块大小（默认为1500字符）
- `--workers`, `-w`: 并发工作线程数（默认为3）
- `--retries`, `-r`: 最大重试次数（默认为3）

## 测试工具

包含了一个测试脚本，可以帮助你快速测试翻译功能：

```bash
python test_ollama_translator.py
```

这将创建一个测试Markdown文件并进行翻译。你也可以指定自己的文件：

```bash
python test_ollama_translator.py --file your_test_file.md
```

## 在代码中使用

```python
from ollama_markdown_translator import OllamaMarkdownTranslator

# 创建翻译器实例
translator = OllamaMarkdownTranslator(
    ollama_url="http://localhost:11434",
    model="llama3:latest",
    source_lang="英文",
    target_lang="中文",
    chunk_size=1500,
    max_workers=3,
    max_retries=3
)

# 读取Markdown文件
with open("your_file.md", "r", encoding="utf-8") as f:
    content = f.read()

# 翻译内容
translated_content = translator.translate_markdown(content)

# 保存翻译结果
with open("translated.md", "w", encoding="utf-8") as f:
    f.write(translated_content)
```

## 注意事项

1. 翻译质量取决于使用的Ollama模型
2. 大型文档可能需要较长时间处理
3. 对于格式非常复杂的Markdown文档，可能需要手动调整一些翻译结果
4. 请确保Ollama服务器正在运行并且可以访问

## 性能优化

- 增加`--workers`参数可以提高并发处理能力，但会增加系统资源使用
- 调整`--chunk-size`可以平衡翻译质量和处理速度
- 对于非常大的文档，建议使用更小的`--chunk-size`值

## 故障排除

- 如果遇到连接错误，请检查Ollama服务器是否正在运行
- 如果翻译结果不理想，尝试使用功能更强大的模型，如`llama3`或`mistral`
- 如果处理速度太慢，尝试增加并发工作线程数或减小块大小 