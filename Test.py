import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, logging
import pickle
import faiss    # for fast vector similarity search and clustering
import numpy as np
import json
from tqdm import tqdm
import os
import re

from langchain.docstore.document import Document    # store and manage documents
from langchain_text_splitters import RecursiveCharacterTextSplitter     # recursively split long text to smaller segments

from sentence_transformers import SentenceTransformer

""" To centralize hyperparameter """
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
    top_k=20,
    top_p=0.9,
    temperature=0.7,
    chunk_size=256,
    chunk_overlap=128,
    max_length=256,
    dist_threshold = 0.5
)

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)     # remove the HTML label
    text = re.sub(r'\[\d+\]', '', text)     # remove paper labels such as [1], [2]
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

""" split it by [Abstract], [Introduction].... """
def split_by_chapter(text):
    pattern = re.compile(r"(Abstract|Introduction|Related Works|Proposed method|Results|Discussion|Conclusion|Acknowledgment)", re.IGNORECASE)
    parts = re.split(pattern, text)
    chapters = []
    if len(parts) <= 1:
        chapters.append(text)
    else:
        if parts[0].strip():
            chapters.append(parts[0].strip())
        for i in range(1, len(parts), 2):   # consists of every two elements, title and content
            header = parts[i].strip()
            content = parts[i+1].strip() if i+1 < len(parts) else ""
            chapters.append(header + "\n" + content)
    return chapters

def split_in_chapter(chapter_text, chunk_size):
    if len(chapter_text) <= chunk_size:
        return [chapter_text]
    else:
        sentences = re.split(r'(?<=[\.!?])\s+', chapter_text)   # split it by (\ . ! ?)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "  # re-set the current chunk because has added to chunks already
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks


""" Load and split documents using langchain """
def split_text(full_text, chunk_size, chunk_overlap):     # each chunk will share "chunk_overlap" characters with the previous chunk
    cleaned_text = clean_text(full_text)
    chapters = split_by_chapter(cleaned_text)
    chunks = []
    for chapter in chapters:
        chapter_chunks = split_in_chapter(chapter, chunk_size)
        # If the same chapter is split into multiple chunks, append the overlapping segment to the end of the previous chunk
        if len(chapter_chunks) > 1:
            new_chunks = [chapter_chunks[0]]
            for i in range(1, len(chapter_chunks)):
                previous = new_chunks[-1]
                overlap_text = previous[-chunk_overlap:] if len(previous) > chunk_overlap else previous
                combined_chunk = overlap_text + " " + chapter_chunks[i]
                new_chunks.append(combined_chunk)
            chapter_chunks = new_chunks
        chunks.extend(chapter_chunks)
    return chunks

""" Build a FAISS index for the text chunks,
    The FAISS index organizes these vectors to enable quick retrieval of the most similar chunks during semanic search """
def build_faiss_index(chunks, embed_model, cache_path="embeddings_cache.pkl", index_path="faiss_index.idx"):
    """ Encodes each chunks into an embedding vector using the provided embedding model """
    if os.path.exists(cache_path) and os.path.exists(index_path):
        print("Loading embedding and FAISS index from cache...")
        with open(cache_path, "rb") as f:
            chunk_embeddings = pickle.load(f)
        index = faiss.read_index(index_path)
    else:
        print("Computing embedding...")   
        chunk_embeddings = embed_model.encode(chunks, batch_size=64, convert_to_tensor=False)
        chunk_embeddings = np.array(chunk_embeddings).astype("float32")   # faiss expect the data in NumPy array format
        dim = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim) # using L2 distance for similarity
        index.add(chunk_embeddings)
        with open(cache_path, "wb") as f:
            pickle.dump(chunk_embeddings, f)
        faiss.write_index(index, index_path)
    return index, chunk_embeddings

""" Retrive relevant chunks based on a query """
def batch_retrieve_chunks(queries, embed_model, index, chunks, top_k, dist_threshold):
    query_embedding = embed_model.encode(queries, batch_size=16, convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32")
    distance, indices = index.search(query_embedding, top_k)
    retrieved_list = []
    for i in range(len(queries)):
        retrieved = []
        seen = set()
        for dist, idx in zip(distance[i], indices[i]):
            if dist_threshold is not None and dist > dist_threshold:
                continue
            chunk = chunks[idx]
            if chunk not in seen:
                retrieved.append(chunk)
                seen.add(chunk)
        retrieved_list.append(retrieved)
    return retrieved_list

def batch_generate_answer(device, prompts, model_llama, tokenizer, max_new_tokens, gen_batch_size, top_k, top_p, temperature):
    answers = []
    for i in tqdm(range(0, len(prompts), gen_batch_size), desc="Generating answers", ncols=120):    # len(prompts) = 100
        batch_prompts = prompts[i:i+gen_batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model_llama.generate(**inputs, max_new_tokens=max_new_tokens,
                                    do_sample=True, top_k=top_k, top_p=top_p, temperature=temperature)
        batch_answers = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        answers.extend(batch_answers)
    return answers

def main():
    device = torch.device("cuda:1")
    logging.set_verbosity_error()   # only display the error message
    # RAG_file_path = "public_dataset.json"
    test_file_path = "private_dataset.json"
    output_file = "313552049_2.json"

    """ Process the private dataset """
    with open(test_file_path, "r", encoding="utf-8") as f:
        test_dataset = json.load(f)

    test_chunks = []
    for sample in tqdm(test_dataset, desc="Processing private dataset full_text", ncols=120):
        if sample.get("full_text"):
            full_text = sample["full_text"]
            chunks = split_text(full_text, config.chunk_size, config.chunk_overlap)
            test_chunks.extend(chunks)

    embed_model = SentenceTransformer('intfloat/multilingual-e5-large', device=device)   # convert text into vectors in order to perform similarity retrieval and build a knowledge base
    rag_index, _ = build_faiss_index(test_chunks, embed_model, "embeddings_cache.pkl", "faiss_index.idx")

    titles = [sample["title"] for sample in test_dataset]
    queries = [sample["question"] for sample in test_dataset]

    retrieved_chunks_list = batch_retrieve_chunks(queries, embed_model, rag_index, test_chunks, config.top_k, config.dist_threshold)
    prompts = []
    for query, retrieved_chunks in zip(queries, retrieved_chunks_list):
        prompt = "Evidence:\n"
        # print(f"查詢: {query}")
        for idx, chunk in enumerate(retrieved_chunks):
            prompt += f"[{idx+1}] {chunk}\n"
            # print(f"檢索到片段: {idx+1}: {chunk}")
        prompt += (
            "\nPlease do the following:\n"
            "1. Summarize the above evidence in 1~2 sentences and label it as 'Evidence Summary:'.\n"
            "2. Based on that summary, answer the question below in 1~4 concise sentences and label it as 'Final Answer:'.\n\n"
            f"Question: \"{query}\"\n\n"
            "Evidence Summary:"
        )
        prompts.append(prompt)
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"     # used to generate outcome
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model_llama = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    model_llama.config.do_sample = False
    model_llama.config.temperature = 1.0
    model_llama.config.top_p = 1.0
        
    predicted_ans = batch_generate_answer(device, prompts, model_llama, tokenizer, config.max_length, gen_batch_size=4,
                                        top_k=config.top_k, top_p=config.top_p, temperature=config.temperature)
    results = []
    for title, output in zip(titles, predicted_ans):
        if "Final Answer" in output:
            parts = output.split("Final Answer")
            evidence_summary = parts[0].replace("Evidence Summary:", "").strip()
            final_answer = parts[1].strip()
        else:
            evidence_summary = ""
            final_answer = output.strip()

        results.append({
            "title": title,
            "answer": final_answer,
            "evidence": evidence_summary
        })
        
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Successfully stored the answer to {output_file}")

if __name__ == "__main__":
    main()
        
