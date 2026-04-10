import pandas as pd
import re
import nltk
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create models folder
os.makedirs('models', exist_ok=True)

print("="*60)
print("TRAINING TICKET CLASSIFIER")
print("="*60)

# Training data - Simple and clear examples
ticket_texts = []
priorities = []

# HIGH PRIORITY tickets (urgent words: down, crash, urgent, emergency)
high_examples = [
    "website is down",
    "server crashed",
    "urgent help needed",
    "emergency system broken",
    "cannot login at all",
    "payment failed money deducted",
    "site not working",
    "error 500 everywhere",
    "database connection lost",
    "everything is broken"
]

# MEDIUM PRIORITY tickets (billing, refund, charge words)
medium_examples = [
    "wrong amount charged",
    "need refund please",
    "update credit card",
    "subscription cancelled but charged",
    "invoice has wrong amount",
    "duplicate charge on card",
    "refund not received",
    "billing address update",
    "charge more than expected",
    "payment method not working"
]

# LOW PRIORITY tickets (how, what, question words)
low_examples = [
    "how to change password",
    "what are business hours",
    "send me brochure",
    "discount for students",
    "how to contact support",
    "what features included",
    "how to upgrade plan",
    "where is my data stored",
    "how to delete account",
    "what is refund policy"
]

# Add all tickets
for text in high_examples:
    ticket_texts.append(text)
    priorities.append("High")

for text in medium_examples:
    ticket_texts.append(text)
    priorities.append("Medium")

for text in low_examples:
    ticket_texts.append(text)
    priorities.append("Low")

# Create dataframe
df = pd.DataFrame({
    'ticket_text': ticket_texts,
    'priority': priorities
})

print(f"\nCreated {len(df)} training tickets")
print("\nPriority distribution:")
print(df['priority'].value_counts())

# Download NLTK data (needed for text processing)
print("\nDownloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Function to clean text
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Split into words
    words = text.split()
    # Remove short words
    clean_words = [w for w in words if len(w) > 2]
    return ' '.join(clean_words)

# Clean all tickets
print("Cleaning text...")
df['cleaned'] = df['ticket_text'].apply(clean_text)

# Show example
print(f"\nExample cleaning:")
print(f"Original: {df['ticket_text'][0]}")
print(f"Cleaned: {df['cleaned'][0]}")

# Convert text to numbers
print("\nConverting text to numbers...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['priority']

print(f"Created {X.shape[0]} tickets with {X.shape[1]} features")

# Split data for training and testing
print("\nSplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
print("Training model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print(f"MODEL ACCURACY: {accuracy * 100:.1f}%")
print("="*60)

# Save model
print("\nSaving model...")
joblib.dump(model, 'models/ticket_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("\n✅ SUCCESS! Model saved to:")
print("   - models/ticket_model.pkl")
print("   - models/vectorizer.pkl")

# Quick test
print("\n" + "="*60)
print("QUICK TEST - Predicting new tickets")
print("="*60)

test_tickets = [
    "My website is down urgently need help",
    "How do I reset my password",
    "I was charged twice please refund"
]

for ticket in test_tickets:
    cleaned = clean_text(ticket)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    print(f"\nTicket: {ticket}")
    print(f"Predicted: {prediction} (Confidence: {confidence*100:.1f}%)")

print("\n" + "="*60)
print("Training complete! Run: streamlit run app.py")
print("="*60)