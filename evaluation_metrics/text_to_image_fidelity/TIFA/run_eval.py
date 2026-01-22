from eval.tifa_score import compute_tifa

overall, per_prompt = compute_tifa("vqa_results.json")

print("====== TIFA Score ======")
print("Overall TIFA:", overall)

# print score for first 5 prompts, for demo
for i, (pid, score) in enumerate(per_prompt.items()):
    if i >= 5:
        break
    print(f"Prompt {pid} score: {score}")
