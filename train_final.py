import pandas as pd
import re
import nltk
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🎯 ULTIMATE TICKET CLASSIFIER - HIGH ACCURACY")
print("="*70)

# ============ HIGH PRIORITY - Strong emergency keywords ============
high_tickets = [
    "URGENT website is completely down customers cannot access",
    "EMERGENCY server crashed all systems offline need immediate help",
    "CRITICAL security breach hackers accessed customer data",
    "URGENT payment gateway failing we are losing sales every minute",
    "EMERGENCY database corrupted data loss happening right now",
    "CRITICAL system not responding for 3 hours urgent fix needed",
    "URGENT cannot login to admin account authentication broken",
    "EMERGENCY production server down critical issue affecting all users",
    "CRITICAL customer data exposed security risk urgent action required",
    "URGENT API completely broken all integrations failing",
    "EMERGENCY website showing error 500 on every page",
    "CRITICAL mobile app crashing on open for all users",
    "URGENT backup system failed data recovery needed immediately",
    "EMERGENCY ssl certificate expired website showing security warning",
    "CRITICAL memory leak causing server to crash every hour",
    "URGENT ddos attack making website unavailable",
    "EMERGENCY database connection pool exhausted no new connections",
    "CRITICAL queue system backlog thousands of pending messages",
    "URGENT replication lag affecting all database reads",
    "EMERGENCY disk full database cannot write new data",
    "CRITICAL load balancer failed all traffic going to dead server",
    "URGENT cache server down website extremely slow",
    "EMERGENCY search index corrupted no results showing",
    "CRITICAL email service down no notifications sending",
    "URGENT file upload system broken users cannot upload documents",
    "EMERGENCY authentication service completely down no one can login",
    "CRITICAL payment processor returning errors for all transactions",
    "URGENT checkout page not working customers cannot complete purchase",
    "EMERGENCY shopping cart not adding items losing sales",
    "CRITICAL user accounts being randomly deleted data loss risk"
]

# ============ MEDIUM PRIORITY - Billing/account keywords ============
medium_tickets = [
    "BILLING wrong amount charged on my credit card statement",
    "REFUND request need money back for duplicate transaction",
    "ACCOUNT update my billing information on profile",
    "SUBSCRIPTION cancelled but I was still charged this month",
    "INVOICE has incorrect billing address please correct",
    "PAYMENT method expired need to update credit card",
    "RECEIPT not received for my last months purchase",
    "ORDER status not updating correctly in dashboard",
    "SHIPPING address cannot be modified after order placed",
    "DISCOUNT code not applied to my recent order total",
    "LOYALTY points missing from my account after purchase",
    "TAX calculation seems incorrect on invoice amount",
    "PRODUCT shipped to wrong address need correction",
    "DELIVERY delayed by several days need status update",
    "CANCEL my subscription immediately please process",
    "PRICE changed after I made the purchase need adjustment",
    "BILLING cycle showing wrong date on invoice",
    "UPGRADE not reflecting on my account after payment",
    "DOWNGRADE plan billing issue please investigate",
    "REFUND status not updated for 2 weeks",
    "CREDIT card declined but my account shows charged",
    "SUBSCRIPTION renewed early before end date",
    "PLAN features missing after upgrade to premium",
    "INVOICE not generated for current billing cycle",
    "PAYMENT receipt not emailed to my address",
    "ACCOUNT charged after cancellation request",
    "WRONG plan showing on my invoice total",
    "TAX exemption not applied to business account",
    "INTERNATIONAL fee charged unexpectedly",
    "CURRENCY conversion amount seems wrong",
    "PROMOTIONAL credit not showing in my account",
    "REFERRAL bonus missing after friend signed up",
    "GIFT card balance shows incorrect amount",
    "BILLING address mismatch with credit card"
]

# ============ LOW PRIORITY - Question/help keywords ============
low_tickets = [
    "HOW TO reset my password I forgot it",
    "WHAT ARE customer support hours on weekends",
    "PLEASE send me your product catalog",
    "DO YOU offer student discount for university students",
    "HOW TO contact technical support team",
    "WHAT features are included in basic plan",
    "HOW TO upgrade my account to premium",
    "WHERE IS my data stored and is it secure",
    "HOW TO delete my account permanently",
    "WHAT IS your refund policy for subscriptions",
    "DO YOU have video tutorials for beginners",
    "CAN I integrate your service with slack",
    "WHAT languages does your platform support",
    "HOW TO schedule a product demo with sales",
    "WHERE IS your privacy policy located",
    "DO YOU have a mobile app for iOS and Android",
    "HOW TO generate an API key for development",
    "WHAT IS your return policy for products",
    "HOW TO invite team members to my account",
    "WHERE IS your API documentation located",
    "HOW TO change my email address on file",
    "HOW TO enable two factor authentication",
    "WHAT browsers are officially supported",
    "HOW TO export my data to CSV format",
    "WHEN IS the next feature release scheduled",
    "HOW TO mute email notifications",
    "WHAT IS your uptime guarantee percentage",
    "HOW TO add team members to organization",
    "WHERE TO find my invoice history",
    "HOW TO set up webhook for events",
    "WHAT IS maximum file upload size",
    "HOW TO change my timezone settings",
    "HOW TO view system audit logs",
    "WHAT IS the API rate limit per minute",
    "HOW TO test integration in sandbox mode"
]

# Create balanced dataset
all_tickets = high_tickets + medium_tickets + low_tickets
all_priorities = ['High']*len(high_tickets) + ['Medium']*len(medium_tickets) + ['Low']*len(low_tickets)

df = pd.DataFrame({'ticket': all_tickets, 'priority': all_priorities})

print(f"\n📊 Dataset created: {len(df)} tickets")
print(f"   🔴 High: {len(high_tickets)} tickets")
print(f"   🟠 Medium: {len(medium_tickets)} tickets")
print(f"   🟢 Low: {len(low_tickets)} tickets")

# Download NLTK
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def clean_text(text):
    """Enhanced text cleaning"""
    text = str(text).lower()
    # Remove special characters but keep important words
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    # Keep important words only
    clean_words = [w for w in words if w not in stop_words and len(w) > 2]
    return ' '.join(clean_words)

print("\n🧹 Cleaning tickets...")
df['cleaned'] = df['ticket'].apply(clean_text)

# Advanced vectorization with more features
vectorizer = TfidfVectorizer(
    max_features=1500,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.85,
    sublinear_tf=True
)

X = vectorizer.fit_transform(df['cleaned'])
y = df['priority']

print(f"✅ Created {X.shape[0]} tickets with {X.shape[1]} features")

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train multiple models with better parameters
print("\n🤖 Training advanced models...")

models = {
    'Random Forest (Best)': RandomForestClassifier(
        n_estimators=500, 
        max_depth=25, 
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=300, 
        learning_rate=0.15,
        max_depth=5,
        random_state=42
    )
}

best_model = None
best_accuracy = 0
best_name = ""

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"   {name}: {acc*100:.1f}%")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_name = name

# Cross-validation
cv_scores = cross_val_score(best_model, X, y_encoded, cv=5)

print(f"\n{'='*70}")
print(f"✅ BEST MODEL: {best_name}")
print(f"✅ ACCURACY: {best_accuracy*100:.1f}%")
print(f"✅ CROSS-VALIDATION: {cv_scores.mean()*100:.1f}% (+/- {cv_scores.std()*100:.1f}%)")
print(f"{'='*70}")

# Save everything
print("\n💾 Saving model...")
joblib.dump(best_model, 'models/ticket_model.pkl')
joblib.dump(vectorizer, 'models/vectorizer.pkl')
joblib.dump(le, 'models/label_encoder.pkl')
joblib.dump(best_accuracy, 'models/accuracy.pkl')

print("\n✅ Model saved successfully!")

# Comprehensive test
print("\n" + "="*70)
print("📝 COMPREHENSIVE TEST - New Tickets")
print("="*70)

test_cases = [
    ("URGENT! My website is completely down! Customers cannot access anything!", "High"),
    ("EMERGENCY! Server crashed, everything is broken!", "High"),
    ("How do I reset my password? I forgot it.", "Low"),
    ("I was charged twice for my subscription this month", "Medium"),
    ("What are your business hours on weekends?", "Low"),
    ("CRITICAL! Payment gateway is failing, losing sales!", "High"),
    ("Please refund my duplicate payment", "Medium"),
    ("Can you send me your product catalog?", "Low"),
    ("Security breach detected on our server!", "High"),
    ("Need to update my billing address", "Medium")
]

correct = 0
for test, expected in test_cases:
    cleaned = clean_text(test)
    vec = vectorizer.transform([cleaned])
    pred_encoded = best_model.predict(vec)[0]
    pred = le.inverse_transform([pred_encoded])[0]
    prob = max(best_model.predict_proba(vec)[0])
    status = "✅" if pred == expected else "❌"
    print(f"\n{status} Ticket: {test[:50]}...")
    print(f"   Predicted: {pred} | Expected: {expected} | Confidence: {prob*100:.1f}%")
    if pred == expected:
        correct += 1

print(f"\n{'='*70}")
print(f"✅ TEST ACCURACY: {correct/len(test_cases)*100:.1f}%")
print(f"{'='*70}")

print("\n🚀 Run: streamlit run app.py")