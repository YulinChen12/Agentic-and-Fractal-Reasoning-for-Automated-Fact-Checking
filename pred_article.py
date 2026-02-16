from preditive_models_training.predictors import analyze_complete_article

article_text = "Title: ...\n\nBody: ..." # Replace with an article text
results = analyze_complete_article(article_text)
print(results["predictions"])

pred = results["predictions"]

def clean_label(x):
    s = str(x)
    return "Unknown" if s.lower() in {"nan", "none"} else s

def fmt(x):
    return "" if x is None else f"{float(x):.3f}"

print("\n=== Predictions ===")
print(f"{'factor':<15}  {'label':<12}  {'conf':>6}")
print("-"*38)

print(f"{'news_coverage':<15}  {clean_label(pred['news_coverage']['label']):<12}  {fmt(pred['news_coverage'].get('confidence')):>6}")
print(f"{'intent':<15}  {clean_label(pred['intent']['label']):<12}  {fmt(pred['intent'].get('confidence')):>6}")
print(f"{'sensationalism':<15}  {clean_label(pred['sensationalism']['label']):<12}  {fmt(pred['sensationalism'].get('confidence')):>6}")
print(f"{'sentiment':<15}  {clean_label(pred['sentiment']):<12}  {'':>6}")
print(f"{'reputation':<15}  {clean_label(pred['reputation']['label']):<12}  {fmt(pred['reputation'].get('confidence')):>6}")
print(f"{'stance':<15}  {clean_label(pred['stance']['label']):<12}  {fmt(pred['stance'].get('confidence')):>6}")

print("\nReputation votes:", pred["reputation"]["sentence_votes"])
print("Stance votes:", pred["stance"]["sentence_votes"])
