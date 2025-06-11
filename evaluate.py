
# python3 evaluate.py -d output/output.json
import gc
import json
import logging
import re
import torch
from tqdm import tqdm
import argparse

from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StoppingCriteriaList, logging as hf_logging

logging.basicConfig(
    format="%(message)s",
    level=logging.DEBUG
)

DEBUG = False

USING_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_MAX_TOK = 64  # max_new_tokens for judge

PROMPT_JUDGEMENT = (
    """Assume you are a human expert in grading predictions given by a model. You are given a document, a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
    1: Take it as granted that the Ground Truth is always correct.
    2: If the Prediction indicates it is not sure about the answer, "score" should be "0"; otherwise, go the next step.
    3: If the Prediction exactly matches the Ground Truth, "score" is 1.
    4: If the Prediction does not exactly match the Ground Truth, go through the following steps.
    5: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
    6: If the Prediction is self-contradictory, "score" must be 0.
    7: If the prediction is not answering the question, "score" must be 0.
    8: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
    9: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
    10: Otherwise, "score" is 0.
    Keep the answer concise. Don't provide irrelevant information."""
)

PROMPT_JUDGE_CONTENT = (
    "document: {document}\n"
    "question: {question}\n"
    "Ground Truth: {ground_truth}\n"
    "Prediction: {prediction}\n"
)

CHAT_JUDGE_TEMPLATE = (
    f"system: {PROMPT_JUDGEMENT}\n"
    "assistant: You must output ONLY '0' or '1', with no other text.\n"
    f"human: {PROMPT_JUDGE_CONTENT}\n"
    "assistant: "
)

class LLMJudge:
    def __init__(self, model_name, max_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map={"": "cuda:0"},
                                                offload_folder="offload", offload_state_dict=True,    # if memory space is limited, store parameter to offload_folder
                                                torch_dtype=torch.float16, low_cpu_mem_usage=True)     # to save more CPU memory
        self.device = next(self.model.parameters()).device
        self.max_tokens = max_tokens
        
    def correctness_score(self, document, question, ground_truth, prediction):
        prompt = CHAT_JUDGE_TEMPLATE.format(document=document, question=question, ground_truth=ground_truth, prediction=prediction)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=10,   # to let LLM output the answer of 0 or 1
                                            eos_token_id=self.tokenizer.eos_token_id, do_sample=False, temperature=None, top_p=None)
        
        # inputs["input_ids"].shape[-1]: the length of prompt
        gen_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        clean_ans = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()   # token -> character
        
        if clean_ans == "1":
            return 1
        if clean_ans == "0":
            return 0
        
        leading_num = re.match(r"[01]", clean_ans)    # only extract the leading 0 or 1
        if leading_num:
            return int(leading_num.group())
        logging.warning(f"unexpected LLM output: {clean_ans!r}")
        return 0

""" To calculate every data's Rouge-L score and correctness """
def evaluate_pair(gt, pred, llm_judge, scorer):
    title = gt["title"]
    question = gt["question"]
    full_text = gt.get("full_text", "")
    gt_evidences = gt.get("evidence", [])
    gt_answer = gt.get("answer", "")
    if isinstance(gt_answer, list):  # if gt_answer is list, turn into string
        gt_answer = " ".join(gt_answer)
    
    pred_evidences = pred.get("evidence", [])
    pred_answer = pred.get("answer", [])
    if isinstance(pred_answer, list):
        pred_answer = " ".join(pred_answer)
    
    scores = []
    
    for gt_evid in gt_evidences:
        score = [scorer.score(gt_evid, pred_evid)["rougeL"].fmeasure for pred_evid in pred_evidences]
        if score:
            scores.append(max(score))   # only record the max score of these pred_evidences
    RougeL = sum(scores) / len(scores) if scores else 0.0
    
    
    Correctness = llm_judge.correctness_score(full_text, question, gt_answer, pred_answer)
    return RougeL, Correctness

def main():
    hf_logging.set_verbosity_error()
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_filepath", "-d", default="output_no_rag.json")
    args = parser.parse_args()
    
    with open("dataset/public_dataset.json", "r", encoding="utf-8") as f:
        gt_file = json.load(f)
    gt_dict = {data["title"]: data for data in gt_file}  # create dict of gt_file, the key is "title" in order to find the pair data quickly

    with open(args.pred_filepath, "r", encoding="utf-8") as f:
        preds = json.load(f)
    
    llm_judge = LLMJudge(USING_MODEL, MODEL_MAX_TOK)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    total_rouge, total_correct, count = 0.0, 0, 0
    with tqdm(preds, desc="Evaluating score", leave=False, ncols=120) as pbar:
        for pred in pbar:
            title = pred["title"]
            RougeL, Correctness = evaluate_pair(gt_dict[title], pred, llm_judge, scorer)
            pbar.set_postfix({"Rouge-L": f"{RougeL:.3f}", "Correctness": f"{Correctness}"})
        
            total_rouge += RougeL
            total_correct += Correctness
            count += 1
            torch.cuda.empty_cache()
    
    avg_rouge = total_rouge / count
    avg_correct = total_correct / count
    print(f"Average ROUGE-L: {avg_rouge:.4f}")
    print(f"Average Correctness: {avg_correct:.4f}")
    print("Evaluation successfully complete")
    
if __name__ == "__main__":
    main()
    
    
            