import subprocess, os, sys, time

def sh(cmd, check=True):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.stdout: print(r.stdout[-3000:])
    if r.stderr: print("STDERR:", r.stderr[-3000:])
    if check and r.returncode != 0:
        raise SystemExit(f"FAILED rc={r.returncode}: {cmd}")
    return r

print("=== Install vLLM + deps ===")
sh("pip install -q vllm==0.6.3 2>&1 | tail -5", check=False)
sh("pip install -q bitsandbytes accelerate 2>&1 | tail -5", check=False)
sh("python -c 'import vllm; print(\"vllm\", vllm.__version__)'")

print("=== GPU info ===")
sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")

print("=== Start vLLM server (bitsandbytes 4bit) ===")
import threading
def run_server():
    subprocess.run([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "tencent/Hy-Memory-30B-A3B",
        "--served-model-name", "hy-memory",
        "--quantization", "bitsandbytes",
        "--load-format", "bitsandbytes",
        "--dtype", "float16",
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.9",
        "--port", "8000",
    ], capture_output=True, text=True)
t = threading.Thread(target=run_server, daemon=True)
t.start()
time.sleep(120)

print("=== Test inference ===")
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
resp = client.chat.completions.create(
    model="hy-memory",
    messages=[
        {"role": "system", "content": "Extract explicit user facts and preferences from the conversation. Write each memory as a concise, self-contained statement. Do not infer or invent information. Return only a JSON object with a \"memories\" list of strings."},
        {"role": "user", "content": "My name is Alex. I live in Seattle and work as a software engineer. I mainly use Python and prefer concise answers with code examples. I'm vegetarian, so please keep that in mind when suggesting meals."},
    ],
    temperature=0.0,
    max_tokens=512,
)
print("RESPONSE:", resp.choices[0].message.content)
print("DONE")
