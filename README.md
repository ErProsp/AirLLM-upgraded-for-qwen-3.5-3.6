How to use it:

1. click code and then download as .zip
2. estract it in your site-packages folder Example for Windows: "C:\Users\[username]\AppData\Local\Python\pythoncore-3.14-64\Lib\site-packages"
3. if you had arleady installed airllm just delete the old folder or replace it with the new one
4. in your code you can use it almost like before.
Example of usage:

Python: "

from transformers import AutoTokenizer
from airllm import AutoModel


model_id = "Qwen/Qwen3.5-9B"  # or anything else from hugging face

model = AutoModel.from_pretrained(
    model_id,
    torch_dtype="auto",      # or torch.float16
    device_map="cuda",       # or "auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

input_text = "hello, tell me what a bit is"
input_tokens = tokenizer(
    input_text,
    return_tensors="pt",
    padding=False,
    truncation=True,
    max_length=512,
)

generation_output = model.generate(
    input_tokens["input_ids"].to(model.device),
    attention_mask=input_tokens["attention_mask"].to(model.device),
    max_new_tokens=128,
    use_cache=False,
    return_dict_in_generate=True,
)

print(tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True))

" <- end fo the code

Leave a comment if you liked this repository. I'll apprecciate it :)
