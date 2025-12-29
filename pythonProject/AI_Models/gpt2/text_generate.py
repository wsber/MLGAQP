from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 模型快照路径
model_hub_path = "/home/wangshuo/ws/AI_models/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

# 加载分词器和模型
tokenizer = GPT2Tokenizer.from_pretrained(model_hub_path)
tokenizer.pad_token = tokenizer.eos_token  # 设置填充值为 <eos>
model = GPT2LMHeadModel.from_pretrained(model_hub_path)

print("模型加载成功！")

# 输入文本
input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors="pt", padding=True)

# 文本生成
outputs = model.generate(
    inputs,
    max_length=50,
    num_return_sequences=1,
    temperature=0.7,
    do_sample=True,
    no_repeat_ngram_size=2,  # 防止重复
    pad_token_id=tokenizer.eos_token_id  # 填充值为 <eos>
)

# 输出生成结果
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
