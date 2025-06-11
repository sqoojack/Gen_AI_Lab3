
# Command line: python3 main.py
# not use langchain
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, BitsAndBytesConfig
import json
import faiss
import numpy as np
import re
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

""" Set Hyperparameter """
class HyperParameters:
    def __init__(self, chunk_size, chunk_overlap, max_length, dist_threshold, top_k, top_p, temperature):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_length = max_length
        self.dist_threshold = dist_threshold
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

config = HyperParameters(
    chunk_size=800,
    chunk_overlap=200,
    max_length=256,     # maximum number of tokens to create
    dist_threshold=0.4,
    top_k=13,
    top_p=0.9,
    temperature=0.7
)

""" Preprocessing the full_text """
def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)     # remove HTML label (ex: <div>, <p>)
    text = re.sub(r'\[\d+\]', '', text)     # remove citation of paper (ex: [1], [2])
    text = re.sub(r'\s+', ' ', text)     # replace multiple space with one space
    return text.strip()

def split_by_chapter(text):
    # Used to detect and split section titles in a paper. re.IGNORECASE: to ignore the uppercase or lowercase
    pattern = re.compile(r"(Abstract|Introduction|Related Works|Proposed method|Results|Discussion|Conclusion|Acknowledgment)", re.IGNORECASE)      
    parts = re.split(pattern, text)     # This is used with re.compile()
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

""" 在chapter內依句點, 問號來切句子, 組成長度不超過chunk_size的chunk"""
def split_in_chapter(chapter_text, chunk_size):
    if len(chapter_text) <= chunk_size:
        return [chapter_text]
    sentence = re.split(r'(?<=[.!?]) +', chapter_text)  # split by sentence
    chunks, current = [], ""   # chunks: is a list used to store the split chunks, current: is a string used to store the current chunk
    
    # Iterate through each sentence and check if adding it to the current chunk exceeds the chunk size
    for s in sentence:  
        if len(current) + len(s) <= chunk_size:
            current += s + " "
        else:
            chunks.append(current.strip())
            current = s + " "
    if current:
        chunks.append(current.strip())
    return chunks

def split_text(full_text, chunk_size, chunk_overlap):
    cleaned = clean_text(full_text)
    chapters = split_by_chapter(cleaned)
    all_chunks = []
    for chapter in chapters:
        segments = split_in_chapter(chapter, chunk_size)    # split the chapter into segments
        
        # address the chunk overlap
        if len(segments) > 1:
            merged = [segments[0]]      # initialize merged with the first segment
            for i in range(1, len(segments)):
                prev = merged[-1]     # get the last merged segment
                overlap = prev[-chunk_overlap:] if len(prev) > chunk_overlap else prev      # get the last chunk with overlap
                merged.append(overlap + " " + segments[i])      # merge the overlap with the current segment
            segments = merged   # update segments with the merged segments
        all_chunks.extend(segments)
    return all_chunks

def build_prompts(question, retrieved_chunks):
    evidence = "\n".join(retrieved_chunks)  # combine the list into a single string by new line
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
            
def generate_answer(device, prompts, tokenizer, model, max_new_tokens):
    results = []
    for prompt in prompts: 
        input_ids = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=config.max_length)
        input_ids = {k: v.to(device) for k, v in input_ids.items()}  # move input_ids to the same device as the model
        input_lens = input_ids["input_ids"].shape[1]  # get the length of prompt_tokens
        
        outputs = model.generate(
            **input_ids,    # inputs contains input_ids and attention_mask, used as input for the model
            max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1,
            temperature=None, top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
    # address the outputs
    for output in outputs:
        generated_text = output[input_lens:]   # remove the prompt tokens from the output
        raw = tokenizer.decode(generated_text, skip_special_tokens=True).strip()  # decode the output
        
        # try to parse(解析) the JSON output
        try:
            data = json.loads("{" + raw + "}")  # parse the JSON output
            answers = data.get("answer", "")  # get the answer from the JSON output
        except Exception:
            m = re.search(r"\[.*?\]", raw)  # find the first JSON-like string
            if m:
                answers = json.loads(m.group(0))  # parse the JSON-like string
            else:
                answers = [raw]
                
        results.append(answers)
    return results


def main():
    device = torch.device("cuda:0")
    logging.set_verbosity_error()  # Suppress warnings from transformers
    
    # with open('test.json', 'r', encoding='utf-8') as f:
    with open('dataset/public_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    embed_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,           # can be fine-tuned depending on the memory usage
        llm_int8_has_fp16_weight=False   # set to True for higher performance
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", torch_dtype=torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    results = []
    for sample in tqdm(data, desc="Per-paper RAG addressing", ncols=120):
        text = sample.get("full_text", "")
        chunks = split_text(text, config.chunk_size, config.chunk_overlap)
            
        embeddings = embed_model.encode(chunks, convert_to_tensor=False, normalize_embeddings=True)
        embeddings = np.array(embeddings, dtype=np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        
        question_embedding = embed_model.encode([sample["question"]], convert_to_tensor=False, normalize_embeddings=True)
        sim_scores, idx = index.search(np.array(question_embedding, dtype=np.float32), config.top_k)
        retrieved_chunks = [chunks[i] for score, i in zip(sim_scores[0], idx[0]) if score >= config.dist_threshold]   # get the top k chunks
        
        prompt = build_prompts(sample["question"], retrieved_chunks)
        answers = generate_answer(device, [prompt], tokenizer, model, config.max_length)[0]
        answers_str = answers[0] if isinstance(answers, list) and len(answers) > 0 else str(answers)     # get the first answer as string
        
        results.append({
            "title" : sample["title"],
            "answer" : answers_str,
            "evidence": retrieved_chunks
        })
        
        del index
        torch.cuda.empty_cache()
    
    # Save the results to a JSON file
    with open('output/output.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Results saved to output.json")
    
if __name__ == "__main__":
    main()

