import torch
from transformers import AutoProcessor, AutoTokenizer, SeamlessM4TModel
import logging
import re

# 配置日志
logging.basicConfig(level=logging.INFO)

# 加载模型、处理器和分词器
def load_model():
    model_name = "facebook/hf-seamless-m4t-medium"
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = SeamlessM4TModel.from_pretrained(model_name)
    logging.info("Model loaded successfully!")
    return processor, tokenizer, model

# 检查是否有 GPU
def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device

# 提取Markdown中的文本（保留Markdown语法）
def extract_text_from_markdown(md_file):
    with open(md_file, "r", encoding="utf-8") as f:
        md_content = f.read()
    return md_content

# 翻译函数
def translate_text(target_lang: str, texts: list, processor=None, tokenizer=None, model=None, device=None):
    translated_texts = []

    # 按批次翻译文本
    batch_size = 8  # 可以根据 GPU 内存调整批次大小
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        # 对输入文本进行分词
        text_inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 将输入张量移至GPU（如果有）
        text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

        # 获取输入的input_ids
        input_ids = text_inputs["input_ids"]

        # 生成翻译后的tokens
        output_tokens = model.generate(input_ids, tgt_lang=target_lang, generate_speech=False)

        # 解码翻译后的tokens
        output_texts = [processor.decode(tokens.tolist(), skip_special_tokens=True) for tokens in output_tokens]

        translated_texts.extend(output_texts)

    return translated_texts

# 替换Markdown中的文本为翻译后的文本
def replace_text_in_markdown(md_content, translated_texts):
    # 使用正则表达式分割Markdown内容，将Markdown语法和纯文本分开
    text_blocks = re.split(r"([#>\*-+`])", md_content)  # 按Markdown语法符号进行分割

    translated_text_index = 0
    for i in range(len(text_blocks)):
        if text_blocks[i] not in ['#', '>', '*', '-', '+', '`']:  # 识别纯文本部分
            text_blocks[i] = translated_texts[translated_text_index]
            translated_text_index += 1
            if translated_text_index >= len(translated_texts):
                break

    # 合并为最终的Markdown内容
    return ''.join(text_blocks)

# 主函数
def main():
    # 加载模型和组件
    processor, tokenizer, model = load_model()

    # 检查设备（GPU或CPU）
    device = get_device()
    model.to(device)

    # 读取Markdown文件内容
    md_file = "/home/dave/PycharmProjects/learnLangChain/output/abc.md"  # 这里填写你的Markdown文件路径
    md_content = extract_text_from_markdown(md_file)

    # 提取Markdown中的文本（去除Markdown语法部分）
    text_to_translate = re.findall(r"([^\n#>\*-+`][^#>\*-+`]+)", md_content)

    # 翻译文本到中文（中文的语言代码为 "zh"）
    translated_texts = translate_text(target_lang="cmn", texts=text_to_translate, processor=processor, tokenizer=tokenizer, model=model, device=device)

    # 替换Markdown中的文本为翻译后的文本
    translated_md_content = replace_text_in_markdown(md_content, translated_texts)

    # 输出翻译后的Markdown内容到新文件
    output_file = "translated_example.md"  # 输出文件路径
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(translated_md_content)

    logging.info(f"Translation completed! Translated file saved as {output_file}")

if __name__ == "__main__":
    main()
