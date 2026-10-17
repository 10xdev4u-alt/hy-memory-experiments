# Hy-Memory-30B-A3B Q4 experiment on Kaggle 2x T4
import subprocess, os, sys, time

def sh(cmd, check=True):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout: print(r.stdout[-2000:])
    if r.stderr: print("STDERR:", r.stderr[-2000:])
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd}")
    return r

# 1. Build llama.cpp with hy_v3 support
print("=== Building llama.cpp ===")
sh("git clone https://github.com/ggerganov/llama.cpp.git /kaggle/working/llama.cpp 2>&1 | tail -3", check=False)
os.chdir("/kaggle/working/llama.cpp")
sh("git checkout f3363593f457169cdfbdfe22ea399b1cc139ba18 2>&1 | tail -3", check=False)
sh("cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 2>&1 | tail -5")
sh("cmake --build build --config Release -j4 2>&1 | tail -10")
assert os.path.exists("/kaggle/working/llama.cpp/build/bin/llama-cli"), "build failed"
print("BUILD OK")

# 2. Download Q4_K_M GGUF
print("=== Downloading Q4_K_M ===")
os.chdir("/kaggle/working")
sh("pip install -q huggingface_hub 2>&1 | tail -2")
from huggingface_hub import hf_hub_download
p = hf_hub_download("rapatsky/Hy-Memory-30B-A3B-GGUF", "Hy-Memory-30B-A3B-Q4_K_M.gguf", local_dir="/kaggle/working")
print("MODEL:", p, os.path.getsize(p)/1e9, "GB")

# 3. Run inference — memory extraction prompt
print("=== Inference ===")
prompt = """<|im_start|>user
Extract durable user facts and preferences from this conversation as JSON memory items.

Conversation:
User: I'm a backend engineer at Tencent, prefer Python and FastAPI, allergic to peanuts, currently learning Rust on weekends.
Assistant: Got it, noted.
User: Also my timezone is UTC+8 and I drink coffee black.

Output JSON only.<|end|><|start|>im_end|>
<|im_start|>assistant
"""
with open("/kaggle/working/prompt.txt","w") as f: f.write(prompt)
sh(f"/kaggle/working/llama.cpp/build/bin/llama-cli -m {p} -f /kaggle/working/prompt.txt -n 512 --temp 0.1 -ngl 99 2>&1 | tail -40")
print("DONE")
