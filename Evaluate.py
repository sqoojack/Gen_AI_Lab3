#!/usr/bin/env python
import json
import re
import logging as std_logging
from rich.console import Console
from rich.logging import RichHandler
from rouge_score import rouge_scorer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import logging as tf_logging

# ---------------------------- Logging 設定 ----------------------------
console = Console(stderr=True, record=True)
log_handler = RichHandler(rich_tracebacks=True, console=console, markup=True)
std_logging.basicConfig(format="%(message)s", datefmt="[%X]", handlers=[log_handler])
log = std_logging.getLogger("rich")
log.setLevel(std_logging.DEBUG)
tf_logging.set_verbosity_error()

DEBUG = False

# ---------------------------- 模型及 API 參數 ----------------------------
USING_MODEL    = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_TEMP     = 0.3
MODEL_MAX_TOK  = 128

# ---------------------------- LLM 判斷 Prompt 模板 ----------------------------
PROMPT_JUDGEMENT = """Assume you are a human expert in grading predictions given by a model. You are given a document, a question and a model prediction. Judge if the prediction matches the ground truth answer by following these steps:
1: Take it as granted that the Ground Truth is always correct.
2: If the Prediction indicates it is not sure about the answer, "score" should be "0"; otherwise, go to the next step.
3: If the Prediction exactly matches the Ground Truth, "score" is 1.
4: If the Prediction does not exactly match the Ground Truth, go through the following steps.
5: If the Ground Truth is a number, "score" is 1 if and only if the Prediction gives a number that almost exactly matches the ground truth.
6: If the Prediction is self-contradictory, "score" must be 0.
7: If the prediction is not answering the question, "score" must be 0.
8: If the prediction is a concise and correct summary of the ground truth, "score" is 1.
9: If ground truth contains a set of items, prediction must contain exactly same items for the score to be 1.
10: Otherwise, "score" is 0.
Keep the answer concise. Don't provide irrelevant information.
"""

PROMPT_JUDGE_CONTENT = """document: {document}
question: {question}
Ground Truth: {answer}
Prediction: {prediction}
"""

CHAT_JUDGE_TEMPLATE = (
    f"system: {PROMPT_JUDGEMENT.strip()}\n"
    f"human: {PROMPT_JUDGE_CONTENT}\n"
    "assistant:"
)

# ---------------------------- LLM 判斷相關元件 ----------------------------
class VLLMOpenAI:
    def __init__(self, model, temperature, max_tokens):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,           # 可微調
                llm_int8_has_fp16_weight=False
            )
        self.model = AutoModelForCausalLM.from_pretrained(model, quantization_config=bnb_config, device_map={"": "cuda:1"}, torch_dtype=torch.float16)
        self.device = "cuda:1"
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def score(self, document, question, answer, prediction):
        prompt = CHAT_JUDGE_TEMPLATE.format(
            document=document[:2000],
            question=question,
            answer=answer,
            prediction=prediction
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=self.temperature>0,
                temperature=self.temperature
            )
        out = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 用正則找第一個獨立的 0 或 1
        m = re.search(r'\b([01])\b', out)
        if m:
            return int(m.group(1))
        else:
            # 找不到就印出來 debug
            print("Judge model output unexpected:", repr(out))
            return 0

llm_judge = VLLMOpenAI(USING_MODEL, MODEL_TEMP, MODEL_MAX_TOK)

# ---------------------------- ROUGE 設定 ----------------------------
scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# ---------------------------- 評估函式 ----------------------------
def evaluate_pair(gt, pred):
    """
    gt: dict from public_dataset.json
    pred: dict from RAG output file
    """
    title    = gt["title"]
    question = gt["question"]
    document = gt.get("full_text", title)

    # Ground‑Truth evidence & answer
    gt_evidences   = gt.get("evidence", [])
    raw_gt_answer  = gt.get("answer", "")
    if isinstance(raw_gt_answer, list):
        gt_answer = " ".join(raw_gt_answer)
    else:
        gt_answer = raw_gt_answer

    # Retrieved evidence & predicted answer
    ret_evidences  = pred.get("evidence", [])
    raw_pred_ans   = pred.get("answer", "")
    if isinstance(raw_pred_ans, list):
        pred_answer = " ".join(raw_pred_ans)
    else:
        pred_answer = raw_pred_ans

    # --- ROUGE-L over evidence ---
    rouge_scores = []
    for gt_ev in gt_evidences:
        # compare each gt_ev against all ret_evidences, take max
        sims = [scorer.score(gt_ev, r)["rougeL"].fmeasure for r in ret_evidences]
        if sims:
            rouge_scores.append(max(sims))
    rouge_l = sum(rouge_scores)/len(rouge_scores) if rouge_scores else 0.0

    # --- LLM 判斷答案正確性 ---
    correctness = llm_judge.score(
        document=document,
        question=question,
        answer=gt_answer,
        prediction=pred_answer
    )

    return rouge_l, correctness

# ---------------------------- 主程式 ----------------------------
def main():
    gt_file = "public_dataset.json"
    test_file = "output.json"
    # 讀 ground truth
    with open(gt_file, "r", encoding="utf-8") as f:
        public = json.load(f)
    gt_map = {r["title"]: r for r in public}

    # 讀 RAG Output
    with open(test_file, "r", encoding="utf-8") as f:
        preds = json.load(f)

    overall_rouge = 0.0
    overall_corr  = 0
    N = 0

    for pred in preds:
        title = pred["title"]
        if title not in gt_map:
            log.warning(f"No GT entry for title: {title}")
            continue
        gt = gt_map[title]
        rouge_l, corr = evaluate_pair(gt, pred)

        pred["final_evidence_rouge_score"] = rouge_l
        pred["llm_answer_correctness"]     = corr

        log.info(f"{title}")
        log.info(f"  ROUGE-L (evidence): {rouge_l:.4f}")
        log.info(f"  Answer Correctness: {corr}")
        log.info("-"*40)

        overall_rouge += rouge_l
        overall_corr  += corr
        N += 1
        torch.cuda.empty_cache()

    avg_rouge = overall_rouge / N if N else 0.0
    avg_corr  = overall_corr  / N if N else 0.0
    log.info(f"Average ROUGE-L: {avg_rouge:.4f}")
    log.info(f"Average Correctness: {avg_corr:.4f}")

    # 存回新的檔案
    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    
    print(f"Successfully stored the answer to eval_results.json")

if __name__ == "__main__":
    main()