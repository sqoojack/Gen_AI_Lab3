
# python3 pipeline_v2.py

import json
from typing import List
from tqdm import tqdm
import gc

# 新增 PromptTemplate
from langchain.prompts import PromptTemplate

# LangChain 核心
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline

# Transformers 相關
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline as hf_pipeline,
    logging
)


def load_json_dataset(path: str) -> List[Document]:
    """從 JSON 讀取 full_text 與 title, 轉成 Document 列表"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return [
        Document(page_content=sample.get("full_text", ""),
                metadata={"title": sample.get("title", "")})
        for sample in data
    ]

def build_vectorstore(docs: List[Document], embeddings) -> FAISS:
    """使用 RecursiveCharacterTextSplitter 與 FAISS 建立向量庫"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    return vectorstore

def main():
    # 1. 載入資料集
    logging.set_verbosity_error()
    
    raw_docs = load_json_dataset("test.json")
    # raw_docs = load_json_dataset("public_dataset.json")
    # raw_docs = load_json_dataset("private_dataset.json")

    # 2. 嵌入模型（e5-large 多語版）
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # 3. 建立 FAISS 向量資料庫
    vectorstore = build_vectorstore(raw_docs, embeddings)

    # 清理 CUDA 緩存以釋放記憶體
    gc.collect()
    torch.cuda.empty_cache()

    # 4. 載入 Llama 3.2-Vision-Instruct 模型（4-bit 量化）
    model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    # model_name = "mistralai/Mistral-7B-Instruct-v0.3"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
        llm_int8_enable_fp32_cpu_offload=True
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        offload_folder="offload",
        offload_state_dict=True,
        max_memory={1: "20GB", "cpu": "64GB"},
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    hf_tokenizer.pad_token = hf_tokenizer.eos_token
    hf_tokenizer.pad_token_id = hf_tokenizer.eos_token_id

    # 5. 建立 Transformers 生成管線，並包裝為 LangChain LLM
    text_gen = hf_pipeline(
        task="text-generation",
        model=hf_model,
        tokenizer=hf_tokenizer,
        max_new_tokens=128,
        do_sample=False,
        top_p=1.0,
        temperature=1.0,
        num_beams=1,
        use_cache=False,
        eos_token_id=hf_tokenizer.eos_token_id,
        pad_token_id=hf_tokenizer.pad_token_id
    )
    llm = HuggingFacePipeline(pipeline=text_gen)

    # 定義只輸出答案的 PromptTemplate
    qa_template = """Use the following context to answer the question.
        And use COT(Chain Of Thought) to reason the answer.
        The most important: Only output the final answer with no explanation or reasoning.

        {context}

        Question: {question}
        Answer:"""
    
    # Map/Combine prompts for map_reduce chain
    map_prompt = PromptTemplate(template="""Use the following context to answer the question. {context}
    Question: {question}
    Answer:""", input_variables=["context", "question"])
    
    combine_prompt = PromptTemplate(template="""
    Here are the partial answers you generated:
    {summaries}

    Now, in one sentence, answer the question:
    “{question}”
    Return only the final answer:
    """, input_variables=["summaries", "question"])

    # 6. 優化檢索器：使用 MMR 降低冗餘並提升多樣性
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 50, "lambda_mult": 0.7}
    )

    # 7. 建立 RetrievalQAChain，改用 map_reduce 以分段整合並優化回答
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,                     # 您的 LLM 實例
        chain_type="map_reduce",     # 指定 MapReduce 模式
        retriever=retriever,         # 您的檢索器
        verbose=False,
        chain_type_kwargs={
            "question_prompt": map_prompt,                       # 原 map_prompt
            "combine_prompt": combine_prompt,                    # 原 combine_prompt
            "combine_document_variable_name": "summaries",       # Reduce 階段接收 summaries
        }
    )

    # 8. 對每個問題執行 RAG 並收集結果
    with open("test.json", 'r', encoding='utf-8') as f:
    # with open("public_dataset.json", 'r', encoding='utf-8') as f:
    # with open("private_dataset.json", 'r', encoding='utf-8') as f:
        samples = json.load(f)

    results = []
    for sample in tqdm(samples, desc="Langchain_version addressing", ncols=120):
        question = sample.get("question", "")
        output = qa_chain({"query": question})
        # 只截取 “Answer:” 之後的文字
        raw_answer = output["result"]
        final_answer = (
            raw_answer.split("Answer:", 1)[1].strip()
            if "Answer:" in raw_answer
            else raw_answer.strip()
        )
        evidence_docs = qa_chain.retriever.get_relevant_documents(question)
        evidence = [doc.page_content for doc in evidence_docs]

        results.append({
            "title": sample.get("title", ""),
            "answer": final_answer,
            "evidence": evidence
        })

    # 9. 儲存結果
    with open("output.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("result saved to output.json")

if __name__ == "__main__":
    main()