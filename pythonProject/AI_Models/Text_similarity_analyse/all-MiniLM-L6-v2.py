from sentence_transformers import SentenceTransformer, util
import numpy as np

# 设置模型路径
model_path = "/home/wangshuo/resource/AIModels/NLP/all-MiniLM-L6-v2"
model = SentenceTransformer(model_path)

def calculate_relevance(topic, article):
    """
    判断文章与主题的相关性。

    :param topic: str, 主题（简短的一句话）
    :param article: str, 文章（较长文本）
    :return: float, 相关性得分（0-1）
    """
    # 对文章进行分段（按句子切分，或者每200字一段）
    paragraphs = article.split("\n")  # 简单按段分割
    if not paragraphs:
        paragraphs = [article]  # 如果无段落，直接使用全文

    # 编码主题和文章段落
    topic_embedding = model.encode(topic, convert_to_tensor=True)
    paragraph_embeddings = model.encode(paragraphs, convert_to_tensor=True)

    # 计算每段与主题的相似性
    scores = util.cos_sim(topic_embedding, paragraph_embeddings).squeeze()

    # 综合得分（取平均或最大值）
    relevance_score = float(scores.max())  # 或 np.mean(scores)
    return relevance_score

# 示例数据
topic = "可持续发展"
article = """在现代社会中，气候变化和环境保护已成为全球关注的重点。许多国家正在推动可持续发展的政策，通过清洁能源、减少碳排放等方式实现长远的生态平衡。"""

# 计算相关性
score = calculate_relevance(topic, article)
print(f"文章与主题的相关性得分：{score:.2f}")
