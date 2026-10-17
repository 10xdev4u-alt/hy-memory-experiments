import subprocess, os, sys, time

def sh(cmd, check=True):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout: print(r.stdout[-3000:])
    if r.stderr: print("STDERR:", r.stderr[-3000:])
    if check and r.returncode != 0:
        print(f"FAILED rc={r.returncode}")
    return r

print("=== GPU ===")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

print("=== Load model 4-bit with CPU offload ===")
code = '''
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4", llm_int8_enable_fp32_cpu_offload=True)
tok = AutoTokenizer.from_pretrained("tencent/Hy-Memory-30B-A3B")
model = AutoModelForCausalLM.from_pretrained(
    "tencent/Hy-Memory-30B-A3B",
    quantization_config=bnb,
    device_map="auto",
    max_memory={0: "13GiB", 1: "13GiB", "cpu": "28GiB"},
)
print("LOADED OK")
msgs = [{"role":"system","content":"Extract explicit user facts and preferences. Return JSON with memories list."},{"role":"user","content":"My name is Alex. I live in Seattle and work as a software engineer. I mainly use Python. I'm vegetarian."}]
inp = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
out = model.generate(**inp, max_new_tokens=256, temperature=0.0, do_sample=False)
print("OUTPUT:", tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True))
'''
with open("/kaggle/working/run.py","w") as f: f.write(code)
sh("python /kaggle/working/run.py 2>&1 | tail -40")
print("DONE")
