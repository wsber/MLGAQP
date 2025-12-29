from transformers import pipeline

# 加载本地 NER 模型
ner_model_path = '/home/wangshuo/ws/AI_models/huggingface/albert-base-v2-finetuned-ner'
nlp_ner = pipeline("ner", model=ner_model_path)

# 示例文本进行实体识别
text = "Barack Obama was born in Hawaii and visited Microsoft headquarters in Seattle."

# 进行命名实体识别
result = nlp_ner(text)

# 打印识别结果
for entity in result:
    print(f"Entity: {entity['word']}, Label: {entity['entity']}, Score: {entity['score']}")


