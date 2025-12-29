from transformers import pipeline

# 设置本地模型路径
model_dir = "/home/wangshuo/ws/AI_models/huggingface/distilbart-mnli-12-6"
# 加载 zero-shot-classification pipeline 并指定模型路径
classifier = pipeline("zero-shot-classification", model=model_dir, tokenizer=model_dir)
def check_nli(topic, post):
    # 使用零样本分类模型判断 post 是否与 topic 相关
    result = classifier(post, [topic], multi_label=False)
    return result
# 示例输入
topic = "support"
post = "Biden is making significant progress in economic policies and reforms."
# 调用模型进行推断
result = check_nli(topic, post)
# 打印结果
print(f"推断结果: {result}")
if result['labels'][0] == topic and result['scores'][0] > 0.5:
    print("post 与 topic 相符合")
else:
    print("post 与 topic 不相符合")
