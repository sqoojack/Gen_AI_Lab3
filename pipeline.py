import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, BitsAndBytesConfig
import json
import faiss
import numpy as np
import re
from tqdm import tqdm
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ---------------------------- 超參數 ----------------------------
class HyperParameters:
    def __init__(self, top_k, top_p, temperature, chunk_size, chunk_overlap, max_length, dist_threshold):
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_length = max_length
        self.dist_threshold = dist_threshold

config = HyperParameters(
    top_k=13,
    top_p=0.9,
    temperature=0.7,
    chunk_size=256,
    chunk_overlap=128,
    max_length=256,
    dist_threshold=0.6
)

# ---------------------------- 輔助函式 ----------------------------
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_by_chapter(text):
    pattern = re.compile(r"(Abstract|Introduction|Related Works|Proposed method|Results|Discussion|Conclusion|Acknowledgment)", re.IGNORECASE)
    parts = re.split(pattern, text)
    chapters = []
    if len(parts) <= 1:
        chapters.append(text)
    else:
        if parts[0].strip():
            chapters.append(parts[0].strip())
        for i in range(1, len(parts), 2):
            header = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            chapters.append(header + "\n" + content)
    return chapters

def split_in_chapter(chapter_text, chunk_size):
    if len(chapter_text) <= chunk_size:
        return [chapter_text]
    sentences = re.split(r'(?<=[\.!?])\s+', chapter_text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) <= chunk_size:
            current += s + " "
        else:
            if current: chunks.append(current.strip())
            current = s + " "
    if current: chunks.append(current.strip())
    return chunks

def split_text(full_text, chunk_size, chunk_overlap):
    cleaned = clean_text(full_text)
    chapters = split_by_chapter(cleaned)
    all_chunks = []
    for chap in chapters:
        segs = split_in_chapter(chap, chunk_size)
        if len(segs) > 1:
            merged = [segs[0]]
            for i in range(1, len(segs)):
                prev = merged[-1]
                overlap = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev
                merged.append(overlap + " " + segs[i])
            segs = merged
        all_chunks.extend(segs)
    return all_chunks

def build_prompt(question: str, retrieved_chunks):
    evidence = "\n".join(retrieved_chunks)
    return f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
Return only the final JSON answer. Do not repeat the question, the evidence, or any dialogue tags.
You need to based on provided question and evidence. To generate the final answer. Rememeber to use the CoT(Chain of Thought) to help you solve problem.
Question:
{question}

Evidence:
{evidence}

Please output:
<your concise answer>

<|end_of_text|>
"""

# ---------------------------- 核心產生並解析答案 ----------------------------
def batch_generate_answer(device, prompts, model, tokenizer, max_new_tokens, gen_batch_size, top_k, top_p, temperature):
    results = []
    for i in range(0, len(prompts), gen_batch_size):
        batch = prompts[i:i+gen_batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 1. 記錄 prompt token 長度
        input_len = inputs["input_ids"].shape[1]

        # 2. 產生
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, 
        num_beams=5, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)

        # 3. 切片並解析
        for out in outputs:
            gen = out[input_len:]  # 從 prompt 之後開始
            raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
            # 把 '"answer": ["English", "Russian"]' → 加大括號成合法 JSON
            try:
                data = json.loads("{" + raw + "}")
                answers = data.get("answer", [])
            except Exception:
                # 若無法解析，就用簡單字串擷取中括號
                m = re.search(r"\[.*?\]", raw)
                if m:
                    answers = json.loads(m.group(0))
                else:
                    answers = [raw]
            results.append(answers)
    return results

# ---------------------------- 主流程 ----------------------------
def main():
    device = torch.device("cuda:1")
    logging.set_verbosity_error()

    # 讀資料
    # with open("test.json", "r", encoding="utf-8") as f:
    # with open("public_dataset.json", "r", encoding="utf-8") as f:
    with open("private_dataset.json", "r", encoding="utf-8") as f:    
        samples = json.load(f)

    # 載模型
    embed_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, load_in_8bit=True, device_map="auto",
        torch_dtype=torch.float16, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = []
    for sample in tqdm(samples, desc="Per-paper RAG", ncols=120):
        chunks = split_text(sample.get("full_text", ""),
                            config.chunk_size, config.chunk_overlap)
        embeds = embed_model.encode(chunks, convert_to_tensor=False)
        embeds = np.array(embeds, dtype="float32")
        index = faiss.IndexFlatL2(embeds.shape[1])
        index.add(embeds)

        q_emb = embed_model.encode([sample["question"]], convert_to_tensor=False)
        dist, idx = index.search(np.array(q_emb, dtype="float32"), config.top_k)
        retrieved = [chunks[i] for d, i in zip(dist[0], idx[0]) if d <= config.dist_threshold]

        prompt = build_prompt(sample["question"], retrieved)
        answers = batch_generate_answer(
            device, [prompt], model, tokenizer,
            config.max_length, 1,
            config.top_k, config.top_p, config.temperature
        )[0]  # 取第一筆

        results.append({
            "title": sample["title"],
            "evidence": retrieved,
            "answer": answers  # 已經是 list 了
        })

        del index
        torch.cuda.empty_cache()

    # 存檔
    with open("313552049.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Successfully stored the answer to output.json")

if __name__ == "__main__":
    main()