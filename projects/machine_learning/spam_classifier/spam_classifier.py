from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example data
texts = ["You won a prize!", "Let's have a meeting tomorrow."]
labels = [1, 0]  # 1 = spam, 0 = not spam

# Convert text to numeric form
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Train the model
model = MultinomialNB()
model.fit(X, labels)

# Predict new message
test = vectorizer.transform(["Claim your free vacation now!"])
print(model.predict(test))  # Output: [1] → spam
