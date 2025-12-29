from transformers import pipeline
import torch

# 设置模型文件路径
# model_dir = '/home/wangshuo/resource/AI-Models/NLP/bart-large-mnli'
model_dir = '/home/wangshuo/resource/AIModels/NLP/bart-large-mnli'

# 确保模型运行在 GPU 上（如果 GPU 可用）
device = 0 if torch.cuda.is_available() else -1  # 使用 GPU (0) 或者 CPU (-1)

# 加载文本推断 pipeline，指定使用 GPU（device=0）或 CPU（device=-1）
nli_pipeline = pipeline("text-classification",
                        model=model_dir, tokenizer=model_dir,
                        device=device)

def check_nli(topic, post):
    """
    使用文本推断模型判断 post 是否符合 topic
    """
    # 构造前提和假设
    premise = post  # 文章内容
    hypothesis = f"This text is about {topic}."  # 主题假设
    # 使用 NLI 模型进行推断
    result = nli_pipeline(f"{premise} [SEP] {hypothesis}")
    return result

# 示例输入
topic = "AI and machine learning"
post = "This is an example post related to AI and machine learning."

# 调用推断函数
result = check_nli(topic, post)
print(f"推断结果: {result}")

# 判断是否符合
if result[0]['label'] == "entailment" and result[0]['score'] > 0.5:
    print("post 与 topic 相符合")
else:
    print("post 与 topic 不相符合")
