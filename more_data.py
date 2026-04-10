import pandas as pd
import re
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("="*60)
print("ADDING MORE TRAINING DATA")
print("="*60)

# HIGH PRIORITY tickets - 20 examples
high_tickets = [
    "website down", "server crashed", "urgent help", "emergency situation",
    "cannot login", "payment failed", "site broken", "error 500",
    "database down", "system crash", "everything broken", "critical issue",
    "site not loading", "app crashing", "blank screen", "error message",
    "account locked", "password not working", "access denied", "timeout error"
]

# MEDIUM PRIORITY tickets - 20 examples
medium_tickets = [
    "wrong charge", "need refund", "update card", "billing issue",
    "duplicate payment", "refund please", "invoice wrong", "charge back",
    "subscription issue", "payment problem", "credit card update",
    "billing address change", "receipt not sent", "order problem",
    "shipping delay", "wrong item sent", "cancel subscription", "price changed",
    "tax calculation wrong", "discount not applied"
]

# LOW PRIORITY tickets - 20 examples
low_tickets = [
    "how to change password", "business hours", "send brochure",
    "student discount", "contact support", "features included",
    "upgrade plan", "data storage", "delete account", "refund policy",
    "video tutorial", "integration options", "language support",
    "demo schedule", "privacy policy", "mobile app", "API key",
    "what is your return policy", "how to invite team members", "where is documentation"
]

# Make sure all lists have exactly 20 items
print(f"High tickets: {len(high_tickets)}")
print(f"Medium tickets: {len(medium_tickets)}")
print(f"Low tickets: {len(low_tickets)}")

# Combine all tickets
all_tickets = high_tickets + medium_tickets + low_tickets
all_priorities = ['High']*20 + ['Medium']*20 + ['Low']*20

print(f"\nTotal tickets: {len(all_tickets)}")
print(f"Total priorities: {len(all_priorities)}")

# Create dataframe
df = pd.DataFrame({
    'ticket_text': all_tickets,
    'priority': all_priorities
})

print(f"\nPriority distribution:")
print(df['priority'].value_counts())

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    clean_words = [w for w in words if len(w) > 2]
    return ' '.join(clean_words)

# Clean all tickets
print("\nCleaning tickets...")
df['cleaned'] = df['ticket_text'].apply(clean_text)

# Show example
print(f"\nExample:")
print(f"Original: {df['ticket_text'][0]}")
print(f"Cleaned: {df['cleaned'][0]}")

# Convert to numbers
print("\nConverting to numbers...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned'])
y = df['priority']

print(f"Created {X.shape[0]} tickets with {X.shape[1]} features")

# Split and train
print("\nTraining model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "="*60)
print(f"NEW MODEL ACCURACY: {accuracy*100:.1f}%")
print("="*60)

# Save model
print("\nSaving improved model...")
joblib.dump(model, 'models/ticket_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')

print("\n✅ SUCCESS! Model updated with 60 training tickets!")
print("\nTesting the new model:")
print("-"*40)

test_tickets = [
    "URGENT my website is completely down",
    "How do I reset my password",
    "I was charged twice please refund me"
]

for ticket in test_tickets:
    cleaned = clean_text(ticket)
    features = vectorizer.transform([cleaned])
    pred = model.predict(features)[0]
    conf = max(model.predict_proba(features)[0])
    print(f"\nTicket: {ticket}")
    print(f"Predicted: {pred} (Confidence: {conf*100:.1f}%)")

print("\n" + "="*60)
print("Run: streamlit run app.py")
print("="*60)