import pandas as pd
import re
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

print("Adding EVEN MORE training data...")

# More diverse examples
high = [
    "website down", "server crashed", "urgent help", "emergency",
    "cannot login", "payment failed", "site broken", "error 500",
    "database down", "system crash", "critical issue", "site not loading",
    "app crashing", "blank screen", "account locked", "access denied",
    "timeout error", "connection failed", "service unavailable", "404 error"
]

medium = [
    "wrong charge", "need refund", "update card", "billing issue",
    "duplicate payment", "refund please", "invoice wrong", "charge back",
    "subscription issue", "payment problem", "credit card update",
    "billing address change", "receipt not sent", "order problem",
    "shipping delay", "wrong item sent", "cancel subscription", "price changed",
    "tax wrong", "discount missing"
]

low = [
    "how to change password", "business hours", "send brochure",
    "student discount", "contact support", "features included",
    "upgrade plan", "data storage", "delete account", "refund policy",
    "video tutorial", "integration options", "language support",
    "demo schedule", "privacy policy", "mobile app", "API key",
    "return policy", "invite members", "documentation"
]

all_tickets = high + medium + low
all_priorities = ['High']*20 + ['Medium']*20 + ['Low']*20

df = pd.DataFrame({'text': all_tickets, 'priority': all_priorities})

def clean(t):
    t = t.lower()
    t = re.sub(r'[^a-zA-Z\s]', '', t)
    return ' '.join([w for w in t.split() if len(w) > 2])

df['clean'] = df['text'].apply(clean)

vec = TfidfVectorizer()
X = vec.fit_transform(df['clean'])
y = df['priority']

model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)

joblib.dump(model, 'models/ticket_model.pkl')
joblib.dump(vec, 'models/vectorizer.pkl')

print(f"✅ Updated with {len(df)} total tickets!")
print("Restart your website to see improvements!")