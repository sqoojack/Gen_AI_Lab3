#!/usr/bin/env python
"""
evaluate.py

This script evaluates retrieval evidence quality using ROUGE-L and answer correctness using an LLM-based judge.
Integrates memory optimizations: 8/4-bit quantization, device_map auto, CPU offload, and cache cleanup.
"""

import json
import re
import gc
import logging as std_logging
from rich.console import Console
from rich.logging import RichHandler
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, StoppingCriteriaList
from transformers.generation.stopping_criteria import StopOnTokens
from transformers import logging as tf_logging

# ---------------------------- Logging Configuration ----------------------------
console = Console(stderr=True, record=True)
log_handler = RichHandler(rich_tracebacks=True, console=console, markup=True)
std_logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[log_handler])
log = std_logging.getLogger("rich")
log.setLevel(std_logging.DEBUG)
tf_logging.set_verbosity_error()

DEBUG = False
TOP_K = 5  # only score top-K retrieved evidences per sample

# ---------------------------- Model and API Parameters ----------------------------
USING_MODEL   = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_TEMP    = 0.3
MODEL_MAX_TOK = 64  # reduce max_new_tokens to save memory

# ---------------------------- Judge Prompt Template ----------------------------
PROMPT_JUDGEMENT: str = (
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
Keep the answer concise. Don't provide irrelevant information.
""")

PROMPT_JUDGE_CONTENT = (
"""document: {document}
question: {question}
Ground Truth: {answer}
Prediction: {prediction}
""")

CHAT_JUDGE_TEMPLATE = (
    f"system: {PROMPT_JUDGEMENT}\n"
    f"human: {PROMPT_JUDGE_CONTENT}\n"
    "assistant: The score is "
)

class LLMJudge:
    def __init__(self, model_name, temperature, max_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            load_in_4bit=False,             # switch to True for 4-bit
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        # Auto device_map + CPU offload
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        self.device = next(self.model.parameters()).device
        self.temperature = temperature
        self.max_tokens = max_tokens

    def score(self, document, question, answer, prediction):
        prompt = CHAT_JUDGE_TEMPLATE.format(
            document=document[:2000],
            question=question,
            answer=answer,
            prediction=prediction
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        stop_criteria = StoppingCriteriaList([StopOnTokens([self.tokenizer.eos_token_id])])
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                stopping_criteria=stop_criteria,
                do_sample=False,
                temperature=0.0
            )
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # cleanup
        del inputs, output_ids
        gc.collect()
        torch.cuda.empty_cache()
        resp = response.strip()
        if resp == "0":
            return 0
        elif resp == "1":
            return 1
        else:
            log.warning(f"Unexpected judge output: {response!r}")
            return 0

# Initialize components once
llm_judge = LLMJudge(USING_MODEL, MODEL_TEMP, MODEL_MAX_TOK)
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def evaluate_pair(gt, pred):
    title    = gt["title"]
    question = gt["question"]
    document = gt.get("full_text", title)

    # GT evidence & answer
    gt_list   = gt.get("evidence", [])
    raw_gt    = gt.get("answer", "")
    gt_answer = " ".join(raw_gt) if isinstance(raw_gt, list) else raw_gt

    # Pred evidence & answer (limit top-K)
    pred_list    = pred.get("evidence", [])[:TOP_K]
    raw_pred     = pred.get("answer", "")
    pred_answer  = " ".join(raw_pred) if isinstance(raw_pred, list) else raw_pred

    # ROUGE-L: iterate over GT, then over top-K
    scores = []
    for gt_ev in gt_list:
        fms = [
            scorer.score(gt_ev, chunk)["rougeL"].fmeasure
            for chunk in pred_list
        ]
        if fms:
            scores.append(max(fms))
    rouge_l = sum(scores) / len(scores) if scores else 0.0

    # Answer correctness
    correct = llm_judge.score(document, question, gt_answer, pred_answer)
    return rouge_l, correct

def main():
    # gt_file = "test.json"
    gt_file   = "public_dataset.json"
    
    pred_file = "output.json"
    # pred_file = "313552049_v2.json"

    with open(gt_file, "r", encoding="utf-8") as f:
        gts = json.load(f)
    gt_map = {e["title"]: e for e in gts}

    with open(pred_file, "r", encoding="utf-8") as f:
        preds = json.load(f)

    total_rouge = 0.0
    total_corr  = 0
    cnt          = 0

    for pred in preds:
        title = pred["title"]
        if title not in gt_map:
            log.warning(f"No GT for {title}")
            continue
        gt       = gt_map[title]
        rl, corr = evaluate_pair(gt, pred)
        pred["final_evidence_rouge_score"] = rl
        pred["llm_answer_correctness"]     = corr

        log.info(f"{title}")
        log.info(f"  ROUGE-L: {rl:.4f}")
        log.info(f"  Correctness: {corr}")
        log.info("-" * 40)

        total_rouge += rl
        total_corr  += corr
        cnt         += 1

    avg_rouge = total_rouge / cnt if cnt else 0.0
    avg_corr  = total_corr / cnt if cnt else 0.0
    log.info(f"Average ROUGE-L: {avg_rouge:.4f}")
    log.info(f"Average Correctness: {avg_corr:.4f}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print("Evaluation complete. Results saved to eval_results.json")

if __name__ == "__main__":
    main()