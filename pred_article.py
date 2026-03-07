from pprint import pprint
import nltk

from pred_models_training.predictors import (
    predict_news_coverage,
    predict_intent,
    predict_sensationalism,
    predict_article_stance,
)

article_title = "Scientists announce a major climate study" # Replace with the article title
article_text = """
A new climate report says global temperatures may continue rising over the next decade.
Researchers analyzed historical emissions and projected future warming scenarios.
Some policymakers say the report should lead to stronger environmental regulations.
Critics argue the predictions are too uncertain and may overstate the near-term risks.
""" # Replace with article text


# sentence split
sentences = nltk.sent_tokenize(article_text)


# 1. News coverage
print("1. predict_news_coverage")
try:
    out = predict_news_coverage(article_text)
    pprint(out)
except Exception as e:
    print("ERROR:", e)

print("\n" + "="*60 + "\n")

# 2. Intent
print("2. predict_intent")
try:
    out = predict_intent(title=article_title, body=article_text)
    pprint(out)
except Exception as e:
    print("ERROR:", e)

print("\n" + "="*60 + "\n")

# 3. Sensationalism
print("3. predict_sensationalism")
try:
    out = predict_sensationalism(article_text)
    pprint(out)
except Exception as e:
    print("ERROR:", e)

print("\n" + "="*60 + "\n")

# 4. Stance
print("5. predict_article_stance")
try:
    out = predict_article_stance(sentences=sentences)
    pprint(out)
except Exception as e:
    print("ERROR:", e)