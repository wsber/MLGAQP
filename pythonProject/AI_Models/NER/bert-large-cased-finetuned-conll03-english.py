from transformers import pipeline

# 加载已经下载并保存在本地的 BERT NER 模型
nlp_ner = pipeline("ner",
                   model="/home/wangshuo/ws/AI_models/huggingface/bert-large-cased-finetuned-conll03-english",
                   device=0)


# 测试文本
text = "Barack Obama was born in Hawaii and visited Microsoft headquarters in Seattle."

# 进行命名实体识别
result = nlp_ner(text)

# 打印识别结果
for entity in result:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']}")
