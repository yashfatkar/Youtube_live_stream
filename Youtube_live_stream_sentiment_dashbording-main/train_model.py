import pickle
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Simple sentiment model
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression())
])

# Dummy training data (enough to make the app work)
X = [
    "I love this video",
    "This is amazing",
    "Very bad experience",
    "I hate this",
    "Not good",
    "Excellent content",
    "Worst video ever",
    "So helpful and nice"
]

y = [
    "Positive",
    "Positive",
    "Negative",
    "Negative",
    "Negative",
    "Positive",
    "Negative",
    "Positive"
]

pipeline.fit(X, y)

# Save model
with open("stacking_model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("stacking_model.pkl created successfully")
