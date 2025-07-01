# Command line: python3 main_langchain.py -d dataset/public_dataset.json

import json
import argparse
from tqdm import tqdm
import gc   # garbage collection -> to save the memory
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, logging

from langchain.prompts import PromptTemplate  # define the prompt template like {question}, {full_text}
from langchain.schema import Document   # used to encapsulation(封裝) a context and metadata
from langchain_community.vectorstores import FAISS      # used to transfrom from text to vector
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline  #  connect embedding model on Hugging face
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.chains import RetrievalQA

""" Load json file and convert each entry into a Document.
    For each item in the json array, extract the "full_text" and "title" field as content and metadata. Return a list of Document objects. """
def load_json_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return [Document(page_content=sample.get("full_text", ""),
                    metadata={"title": sample.get("title", "")})    # metadata is type of dict
            for sample in data]

""" split the text to the chunk, then turn into the FAISS vector"""
def build_FAISS(docs, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)     # overlap is used to ensure connectivity of context
    split_docs = splitter.split_documents(docs)     
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embeddings)    # build FAISS vector
    return vectorstore      # used to vector similarity search

def main():
    logging.set_verbosity_error()   # only log errors
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", "-d", default="dataset/public_dataset.json")    # "-d" is short option, can command: -d
    parser.add_argument("--model_name", "-m", default="meta-llama/Llama-3.2-11B-Vision-Instruct")
    args = parser.parse_args()
    
    raw_docs = load_json_dataset(args.data_path)
    embeddings = HuggingFaceEmbeddings(model_name='intfloat/multilingual-e5-large')
    vectorstore = build_FAISS(raw_docs, embeddings)
    
    gc.collect()    # release memory
    torch.cuda.empty_cache()
    
    # 4-bit quantization: convert weight from float32/float16 to 4-bit integers
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",    # use NormalFloat4
                                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)  # double_quant: let quant_config(scale, zero_point) to 4bit
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=bnb_config, device_map={"": "cuda:0"},
                                                offload_folder="offload", offload_state_dict=True,    # if memory space is limited, store parameter to offload_folder
                                                torch_dtype=torch.float16, low_cpu_mem_usage=True)     # to save more CPU memory
    
    """ Tokenizer: text -> token -> ID -> add special token -> attention_mask(區分真實token和填充token) -> decode to context """
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)  # load tokenizer
    tokenizer.pad_token_id = tokenizer.eos_token_id

    """ Create generation pipeline from Transformer """
    text_generation = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256,
                            do_sample=False, num_beams=1, top_p=None, temperature=None, use_cache=True, return_full_text=False,   # remember to set return_full_text=False, or it will return full_prompt
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    
    llm = HuggingFacePipeline(pipeline=text_generation)   # becomes an LLM object which compatible with langchain
    
    """ k: number of passages fed to the LLM eventually
        fetch_k: number of candidates retrieved before picking k
        lambda_mult: MMR's relevance-vs-diversity weighting """
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 60, "lambda_mult": 0.2})
    
    prompt = PromptTemplate(
        template="""
    You are an answer-only assistant. Reply with the final answer only, no explanation.

    Example
    Context: .....
    Question: Which languages are explored in this paper?
    Final Answer: English
    ---
    Context:
    {context}

    Question:
    {question}

    Final Answer:
    """,
        input_variables=["context", "question"]
    )
    
    # LangChain’s RetrievalQA will automatically fill the retrieved results into {context} in prompt
    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=False,
        return_source_documents=True, chain_type_kwargs={"prompt": prompt})
    
    with open(args.data_path, 'r', encoding="utf-8") as f:
        samples = json.load(f)
        
    results = []
    for sample in tqdm(samples, desc="Langchain version addressing", ncols=120):
        question = sample.get("question", "")
        output = qa_chain({"query": question})
        raw_answer = output["result"].strip()
        clean_answer = raw_answer.split('---')[0].splitlines()[0].strip()   # only answer first paragraph, stop by "---"" and "/n"
        
        evidence_docs = output["source_documents"]
        evidence = [doc.page_content for doc in evidence_docs]
        
        results.append({
            "title": sample.get("title", ""),
            "answer": clean_answer,
            "evidence": evidence
        })
    
    with open("output/output_langchain.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Result save to output_langchain.json successfully")
    
if __name__ == "__main__":
    main()