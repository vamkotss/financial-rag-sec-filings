"""Demo: load the existing FAISS index and answer a few questions."""
import os
from src import answer


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY environment variable")
        return
    
    questions = [
        "What are Tesla's main risks related to autonomous driving?",
        "How does Nvidia describe its data center business?",
        "What does Apple say about its services segment?",
        "What does Meta say about Reality Labs investments?",
    ]
    
    for q in questions:
        print("=" * 70)
        answer(q)
        print()


if __name__ == "__main__":
    main()
