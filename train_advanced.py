import pandas as pd
import re
import nltk
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 ADVANCED TICKET CLASSIFIER WITH HIGH ACCURACY")
print("="*70)

# Create comprehensive training data
tickets = []
priorities = []

# HIGH PRIORITY - Emergency words
high_examples = [
    ("website completely down customers cannot access", "High"),
    ("server crashed all systems offline emergency", "High"),
    ("critical security breach hackers detected", "High"),
    ("urgent payment gateway failing losing sales", "High"),
    ("database corrupted data loss happening now", "High"),
    ("system not responding for 3 hours urgent", "High"),
    ("cannot login any account authentication broken", "High"),
    ("emergency maintenance required immediately", "High"),
    ("production server down critical issue", "High"),
    ("customer data exposed security risk", "High"),
    ("api completely broken integration failing", "High"),
    ("website showing error 500 all pages", "High"),
    ("mobile app crashing on open for everyone", "High"),
    ("backup failed data recovery needed", "High"),
    ("ssl certificate expired website unsafe", "High"),
    ("memory leak server crashing hourly", "High"),
    ("ddos attack website unavailable", "High"),
    ("database connection pool exhausted", "High"),
    ("queue system backlog critical", "High"),
    ("replication lag affecting all users", "High")
]

# MEDIUM PRIORITY - Billing/Update words
medium_examples = [
    ("wrong amount charged on credit card", "Medium"),
    ("need refund for duplicate transaction", "Medium"),
    ("update billing information on profile", "Medium"),
    ("subscription cancelled but still charged", "Medium"),
    ("invoice has incorrect billing address", "Medium"),
    ("payment method expired need update", "Medium"),
    ("receipt not received for purchase", "Medium"),
    ("order status not updating correctly", "Medium"),
    ("shipping address cannot be modified", "Medium"),
    ("discount code not applied to order", "Medium"),
    ("loyalty points missing from account", "Medium"),
    ("tax calculation seems incorrect", "Medium"),
    ("product shipped to wrong address", "Medium"),
    ("delivery delayed by several days", "Medium"),
    ("cancel my subscription immediately", "Medium"),
    ("price changed after purchase", "Medium"),
    ("billing cycle wrong date", "Medium"),
    ("upgrade not reflecting on account", "Medium"),
    ("downgrade plan billing issue", "Medium"),
    ("refund status not updated", "Medium")
]

# LOW PRIORITY - Question words
low_examples = [
    ("how to reset my password", "Low"),
    ("what are customer support hours", "Low"),
    ("send me product catalog please", "Low"),
    ("student discount available", "Low"),
    ("how to contact technical support", "Low"),
    ("what features in basic plan", "Low"),
    ("how to upgrade my account", "Low"),
    ("where is my data stored", "Low"),
    ("how to delete my account", "Low"),
    ("what is your refund policy", "Low"),
    ("video tutorials available", "Low"),
    ("integrate with slack possible", "Low"),
    ("what languages supported", "Low"),
    ("schedule product demo", "Low"),
    ("privacy policy where to find", "Low"),
    ("mobile app available", "Low"),
    ("how to generate api key", "Low"),
    ("what is your return policy", "Low"),
    ("how to invite team members", "Low"),
    ("where is documentation located", "Low")
]

for text, priority in high_examples:
    tickets.append(text)
    priorities.append(priority)

for text, priority in medium_examples:
    tickets.append(text)
    priorities.append(priority)

for text, priority in low_examples:
    tickets.append(text)
    priorities.append(priority)

df = pd.DataFrame({'ticket': tickets, 'priority': priorities})

print(f"\n📊 Dataset created: {len(df)} tickets")
print(f"   High: {sum(df['priority'] == 'High')}")
print(f"   Medium: {sum(df['priority'] == 'Medium')}")
print(f"   Low: {sum(df['priority'] == 'Low')}")

# Download NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    clean_words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(clean_words)

print("\n🧹 Cleaning tickets...")
df['cleaned'] = df['ticket'].apply(clean_text)

# Advanced vectorization
vectorizer = TfidfVectorizer(
    max_features=500,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['priority']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train multiple models
print("\n🤖 Training advanced models...")

rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

gb_model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
gb_acc = accuracy_score(y_test, gb_pred)

# Use best model
if rf_acc >= gb_acc:
    best_model = rf_model
    best_acc = rf_acc
    model_name = "Random Forest"
else:
    best_model = gb_model
    best_acc = gb_acc
    model_name = "Gradient Boosting"

print(f"\n✅ Best Model: {model_name}")
print(f"✅ Accuracy: {best_acc*100:.1f}%")

# Cross-validation
cv_scores = cross_val_score(best_model, X, y_encoded, cv=5)
print(f"✅ Cross-validation score: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")

# Save everything
print("\n💾 Saving model and components...")
joblib.dump(best_model, 'models/ticket_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(cv_scores, 'models/cv_scores.pkl')
joblib.dump(best_acc, 'models/accuracy.pkl')

print("\n" + "="*70)
print("✅ MODEL TRAINING COMPLETE!")
print(f"   Final Accuracy: {best_acc*100:.1f}%")
print("="*70)

# Quick test
print("\n📝 Quick Test:")
test_tickets = [
    "URGENT website completely down help",
    "how to reset my password",
    "wrong amount charged on my card"
]

for test in test_tickets:
    cleaned = clean_text(test)
    vec = vectorizer.transform([cleaned])
    pred_encoded = best_model.predict(vec)[0]
    pred = le.inverse_transform([pred_encoded])[0]
    prob = max(best_model.predict_proba(vec)[0])
    print(f"   '{test}' -> {pred} ({prob*100:.1f}%)")

print("\n🚀 Run: streamlit run app.py")