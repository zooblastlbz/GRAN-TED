# pip install accelerate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

tokenizer = AutoTokenizer.from_pretrained("/ytech_m2v5_hdd/workspace/kling_mm/Models/t5gemma-2b-2b-ul2")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "/ytech_m2v5_hdd/workspace/kling_mm/Models/t5gemma-2b-2b-ul2",
    device_map="auto",
)

input_text = "Write me a poem about Machine Learning. Answer:"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=32)
print(tokenizer.decode(outputs[0]))
