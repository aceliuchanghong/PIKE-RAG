import os
from openai import OpenAI

client = OpenAI(
    api_key="ollama",
    base_url=os.getenv("OLLAMA_BASE_URL"),
)
messages = [
    {
        "role": "user",
        "content": "1+1=?",
    }
]
completion = client.chat.completions.create(
    model=os.getenv("OLLAMA_MODEL"),
    messages=messages,
    temperature=0.7,
)

"""
uv run z_utils/test_llm.py

# 要预加载模型并常驻
curl http://localhost:11434/api/generate  -d '{
  "model": "gemma3:27b",
  "keep_alive": -1
}'
# 要卸载模型并释放内存
curl http://localhost:11434/api/generate -d '{
    "model": "gemma3:27b", 
    "keep_alive": 0
}'
"""
print(completion.choices[0].message.content)
