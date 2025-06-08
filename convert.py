import json

# 1. 讀入原始 JSON（假設它是一個 list of dicts）
with open('313552049_out.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 2. 遍歷列表，對每一筆資料做轉換
converted = []
for item in data:
    # 取出 title
    title = item.get("title", "")
    # 如果 answer 是列表，就取第一個元素，否則設為空字串
    ans_list = item.get("answer", [])
    answer = ans_list[0] if isinstance(ans_list, list) and ans_list else ""
    # 直接取 evidence（保證是列表）
    evidence = item.get("evidence", [])

    converted.append({
        "title": title,
        "answer": answer,
        "evidence": evidence
    })

# 3. 將結果寫回 output.json
with open('output.json', 'w', encoding='utf-8') as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

# 4. （可選）在終端印出結果檢查
print(json.dumps(converted, ensure_ascii=False, indent=2))