import argparse
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Parse command line argument for quantization


app = FastAPI()
templates = Jinja2Templates(directory="/playpen-ssd/smerrill/llm_decisions/templates")

model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
quantize = True
# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model {'in 4-bit quantized mode' if quantize else 'in full precision'}...")

if quantize:
    from transformers import BitsAndBytesConfig

    # Requires bitsandbytes installed: pip install bitsandbytes
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=quant_config,
    )
    
else:
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
        max_new_tokens=1000,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return templates.TemplateResponse("index.html", {"request": request, "generated_text": generated_text, "prompt": prompt})
