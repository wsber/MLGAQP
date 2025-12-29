from transformers import pipeline
# 设置模型文件路径
model_dir = '/home/wangshuo/ws/AI_models/huggingface/bart-large-mnli'
# 加载 Zero-Shot Classification pipeline
classifier = pipeline("zero-shot-classification",
                      model=model_dir, tokenizer=model_dir)
def classify_post(post, candidate_labels):
    """
    使用 Zero-Shot Classification 模型对 post 进行分类
    """
    # 使用 Zero-Shot Classification 进行推断
    result = classifier(post, candidate_labels)
    return result
# 示例输入文本和候选类别
post = "The team won the championship game in a thrilling match."
candidate_labels = [
    "sports",
    "politics",
    "technology"
]
# 调用分类函数
result = classify_post(post, candidate_labels)
# 输出推断结果
print("推断结果:")
for label, score in zip(result['labels'], result['scores']):
    print(f"  类别: {label}, 概率: {score:.4f}")
# 获取最可能的类别
most_likely_label = result['labels'][0]
most_likely_score = result['scores'][0]
print("\n最终分类结果:")
print(f"最可能的类别: {most_likely_label}, 概率: {most_likely_score:.4f}")
