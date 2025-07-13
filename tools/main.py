from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch
import torch
import os

# CUDA_VISIBLE_DEVICES=0,2,3,4,5,6,7 python -m uvicorn main:app --host 0.0.0.0 --port 8000
# ssh -L 8000:localhost:8000 smerrill@152.2.134.51

app = FastAPI()
templates = Jinja2Templates(directory="/playpen-ssd/smerrill/llm_decisions/templates")

model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
model_name = '/playpen-ssd/smerrill/trained_models/meta-llama/Meta-Llama-3-70B-Instruct/jonnoalcaro_12_0.2'
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
model.eval()

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "generated_text": ""})

@app.post("/generate", response_class=HTMLResponse)
async def generate(request: Request, prompt: str = Form(...)):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return templates.TemplateResponse("index.html", {"request": request, "generated_text": generated_text, "prompt": prompt})
