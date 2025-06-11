
# command line: python3 main_baseline.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, logging
import json
from tqdm import tqdm

def build_prompt(question, full_text):
    return f"""\
    <|begin_of_text|><|start_header_id|>user<|end_header_id|>
    Return only the final JSON answer. Do not repeat the question or any dialogue tags.
    You are a paper answering assistant. Generate the concise answer only.

    Question:
    {question}


    Please output:
    <your concise answer>
    <|end_of_text|>
    """

def generate_answer(device, model, tokenizer, prompt, max_new_tokens):
    # 將 prompt 轉成 tensor
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    # 生成
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    # 解碼並取出答案
    gen = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text

def main():
    logging.set_verbosity_error()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入模型（同 RAG 版本）
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 讀取資料
    with open("dataset/public_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for sample in tqdm(data, desc="Direct LLM answering", ncols=120):
        question = sample["question"]
        full_text = sample.get("full_text", "").replace("\n", " ")
        prompt = build_prompt(question, full_text)
        
        answer = generate_answer(device, model, tokenizer, prompt, max_new_tokens=256)
        results.append({
            "title": sample.get("title", ""),
            "answer": answer,
            "evidence": []
        })

    # 存檔
    with open("output/output_baseline.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Results saved to output_baseline.json")

if __name__ == "__main__":
    main()