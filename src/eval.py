"""Eval harness for RAG quality measurement."""
from .query import answer, load_vectorstore


EVAL_SET = [
    {
        "question": "What does Tesla say about its Robotaxi business?",
        "expected_keywords": ["robotaxi", "autonomous", "driver"],
    },
    {
        "question": "How does Nvidia describe its data center business?",
        "expected_keywords": ["data center", "AI", "GPU"],
    },
    {
        "question": "What are Apple\'s main revenue segments?",
        "expected_keywords": ["iPhone", "services", "wearables"],
    },
    {
        "question": "What does Meta say about Reality Labs?",
        "expected_keywords": ["reality labs", "metaverse", "vr"],
    },
    {
        "question": "What regulatory risks does Tesla face internationally?",
        "expected_keywords": ["regulation", "international", "compliance"],
    },
]


def evaluate():
    """Run all eval questions and score by keyword presence."""
    vectorstore = load_vectorstore()
    results = []
    
    for item in EVAL_SET:
        ans, _ = answer(item["question"], vectorstore=vectorstore, verbose=False)
        ans_lower = ans.lower()
        hits = [kw for kw in item["expected_keywords"] if kw.lower() in ans_lower]
        score = len(hits) / len(item["expected_keywords"])
        results.append({
            "question": item["question"],
            "score": score,
            "hits": hits,
            "expected": item["expected_keywords"],
            "answer": ans,
        })
    
    avg = sum(r["score"] for r in results) / len(results)
    return results, avg


if __name__ == "__main__":
    results, avg = evaluate()
    print(f"Average score: {avg:.2f}")
    for r in results:
        print(f"\n{r[\'question\']}: {r[\'score\']:.2f}")
