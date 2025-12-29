from transformers import ElectraTokenizer, ElectraModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的 ELECTRA 模型和分词器
model_path = "/home/wangshuo/ws/AI_models/huggingface/electra-small-discriminator"  # 替换为模型下载路径
tokenizer = ElectraTokenizer.from_pretrained(model_path)
model = ElectraModel.from_pretrained(model_path)

def get_embedding(text):
    # 编码文本并获取输出嵌入向量
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    # 取最后一层隐藏状态的平均池化作为文本嵌入
    embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embedding

# 示例输入
# topic = "climate change policy"
# post = "The government announced new initiatives to reduce carbon emissions and combat climate change."
topic = "i support biden"
post = "Biden is making significant progress in economic policies and reforms."

# 计算嵌入向量
topic_embedding = get_embedding(topic)
post_embedding = get_embedding(post)

# 计算余弦相似度
similarity = cosine_similarity(topic_embedding, post_embedding)
print(f"相似性分数: {similarity[0][0]}")
