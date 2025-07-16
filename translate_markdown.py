import torch
from transformers import AutoTokenizer, AutoProcessor, SeamlessM4TModel
import logging
from huggingface_hub import login
import os

# 配置日志
logging.basicConfig(level=logging.INFO)

access_token = "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"

# 登录 Hugging Face
login(token=access_token)  # 使用你从 Hugging Face 获取的 Token

# 加载模型、tokenizer 和处理器
model_name = "facebook/hf-seamless-m4t-medium"
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = SeamlessM4TModel.from_pretrained(model_name)
logging.info("模型加载成功！")

# 检查是否可以使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 启用 FP16 来加速计算（如果支持）
if device.type == "cuda":
    model = model.half()

# 翻译函数：将输入文本翻译为目标语言（默认为中文）
def translate_text(text, target_lang="cmn"):  # 目标语言默认为简体中文 (cmn)
    try:
        # 处理输入文本
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 移动数据到GPU
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # 强制使用目标语言的 BOS token ID
        target_lang_id = tokenizer.lang2id.get(target_lang, None)  # 使用 lang2id 获取语言的 ID

        if target_lang_id is None:
            raise ValueError(f"目标语言 {target_lang} 不受支持！")

        # 翻译
        translated_tokens = model.generate(inputs["input_ids"], forced_bos_token_id=target_lang_id, tgt_lang=target_lang, generate_speech=False)

        # 解码翻译后的文本
        translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
        return translated_text
    except Exception as e:
        logging.error(f"翻译过程中发生错误: {str(e)}")
        return None

# 批量翻译函数：翻译多个文本
def translate_batch(text_batch, target_lang="cmn"):
    try:
        inputs = tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # 移动数据到GPU
        inputs = {key: value.to(device) for key, value in inputs.items()}

        # 强制使用目标语言的 BOS token ID
        target_lang_id = tokenizer.lang2id.get(target_lang, None)  # 使用 lang2id 获取语言的 ID

        if target_lang_id is None:
            raise ValueError(f"目标语言 {target_lang} 不受支持！")

        # 翻译
        translated_tokens = model.generate(inputs["input_ids"], forced_bos_token_id=target_lang_id, tgt_lang=target_lang, generate_speech=False)

        # 解码翻译后的文本
        translated_texts = [tokenizer.decode(tokens, skip_special_tokens=True) for tokens in translated_tokens]
        return translated_texts
    except Exception as e:
        logging.error(f"批量翻译过程中发生错误: {str(e)}")
        return []

# 读取Markdown文件并翻译其中的英文内容
def translate_markdown(input_md, output_md, target_lang="cmn"):
    try:
        with open(input_md, "r", encoding="utf-8") as f:
            md_content = f.read()

        # 将Markdown分割成一批一批的文本
        lines = md_content.splitlines()
        batch_size = 10  # 每次翻译10行
        translated_content = ""

        for i in range(0, len(lines), batch_size):
            batch = lines[i:i + batch_size]
            translated_batch = translate_batch(batch, target_lang)
            translated_content += "\n".join(translated_batch) + "\n"

        # 保存翻译后的内容
        with open(output_md, "w", encoding="utf-8") as f:
            f.write(translated_content)

        logging.info(f"翻译完成！结果已保存到 {output_md}")

    except Exception as e:
        logging.error(f"翻译过程中发生错误: {str(e)}")

# 主程序
if __name__ == "__main__":
    # 输入和输出文件路径
    input_md = "./output/abc.md"  # 输入的Markdown文件路径
    output_md = "translated_facebook.md"  # 输出翻译后的Markdown文件路径

    # 目标语言（默认为简体中文）
    target_lang = "cmn"  # 简体中文 (cmn)

    # 检查输入文件是否存在
    if not os.path.exists(input_md):
        logging.error(f"输入文件 {input_md} 不存在！")
    else:
        # 翻译并保存结果
        translate_markdown(input_md, output_md, target_lang)
