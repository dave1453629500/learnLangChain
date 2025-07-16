#!/usr/bin/env python3
"""
测试 Ollama Markdown 翻译器
"""
import os
import argparse
from ollama_markdown_translator import OllamaMarkdownTranslator

def create_test_markdown():
    """创建一个用于测试的Markdown文件"""
    test_md = """# Test Markdown File for Translation

This is a simple markdown file used to test the Ollama Markdown translator.

## Features

- **Preserves** markdown formatting
- Handles code blocks properly
- Supports _inline formatting_
- Manages [links](https://example.com) correctly

## Code Example

```python
def hello_world():
    print("Hello, world!")
    return True
```

### Another section with a table

| Header 1 | Header 2 |
|----------|----------|
| Value 1  | Value 2  |
| Value 3  | Value 4  |

## Image Example

![Sample Image](https://example.com/image.jpg)

## Conclusion

This test file contains various markdown elements to ensure the translator works correctly with different types of content.
"""
    
    test_file = "test_markdown.md"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_md)
    
    return test_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试Ollama Markdown翻译器')
    parser.add_argument('--file', '-f', type=str, help='输入Markdown文件路径(如果未指定，将创建一个测试文件)')
    parser.add_argument('--model', '-m', type=str, default='llama3:latest', help='Ollama模型名称（默认为llama3:latest）')
    parser.add_argument('--url', '-u', type=str, default='http://localhost:11434', help='Ollama服务器URL')
    parser.add_argument('--source', '-s', type=str, default='英文', help='源语言')
    parser.add_argument('--target', '-t', type=str, default='中文', help='目标语言')
    
    args = parser.parse_args()
    
    # 如果未指定文件，创建测试文件
    if args.file:
        input_file = args.file
    else:
        input_file = create_test_markdown()
        print(f"创建了测试文件: {input_file}")
    
    # 创建输出文件路径
    output_file = os.path.splitext(input_file)[0] + "_translated.md"
    
    # 创建翻译器
    translator = OllamaMarkdownTranslator(
        ollama_url=args.url,
        model=args.model,
        source_lang=args.source,
        target_lang=args.target
    )
    
    try:
        print(f"读取文件: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print("开始翻译...")
        translated_content = translator.translate_markdown(content)
        
        print(f"写入输出文件: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(translated_content)
        
        print(f"翻译完成! 结果已保存到 {output_file}")
        
    except Exception as e:
        print(f"错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 