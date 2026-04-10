import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import csv

# Page configuration
st.set_page_config(
    page_title="Enterprise Support Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PREMIUM DARK THEME CSS
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(ellipse at top, #1a1a2e, #0a0a0f);
    }
    
    .premium-header {
        background: rgba(18, 18, 30, 0.8);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 180, 216, 0.2);
        border-radius: 24px;
        padding: 2rem 2rem;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .premium-header h1 {
        background: linear-gradient(135deg, #00b4d8 0%, #90e0ef 50%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
    }
    
    .premium-header p {
        color: #8a8a9e;
        margin-top: 0.8rem;
    }
    
    .premium-card {
        background: rgba(18, 18, 30, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 1.2rem;
        transition: all 0.3s ease;
    }
    
    .premium-card:hover {
        transform: translateY(-4px);
        border-color: rgba(0, 180, 216, 0.3);
    }
    
    .card-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #00b4d8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .card-label {
        color: #6c6c84;
        font-size: 0.85rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .priority-premium-high {
        background: linear-gradient(135deg, #ff006e, #d90429);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 0, 110, 0.3);
    }
    
    .priority-premium-medium {
        background: linear-gradient(135deg, #ffb703, #fb8500);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(255, 183, 3, 0.3);
    }
    
    .priority-premium-low {
        background: linear-gradient(135deg, #06d6a0, #059669);
        border-radius: 16px;
        padding: 1rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(6, 214, 160, 0.3);
    }
    
    .priority-title {
        font-size: 1.3rem;
        font-weight: 800;
        color: white;
    }
    
    .premium-action {
        background: linear-gradient(135deg, rgba(0, 180, 216, 0.1), rgba(0, 119, 182, 0.05));
        border: 1px solid rgba(0, 180, 216, 0.2);
        border-radius: 20px;
        padding: 1.2rem;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 180, 216, 0.4);
    }
    
    .stTextArea textarea {
        background: rgba(18, 18, 30, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        color: #e0e0e0;
        padding: 1rem;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a0f 0%, #12121e 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background: rgba(18, 18, 30, 0.6);
        border-radius: 60px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        color: #8a8a9e;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        color: white;
    }
    
    .premium-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0, 180, 216, 0.3), transparent);
        margin: 1.5rem 0;
    }
    
    .premium-footer {
        text-align: center;
        padding: 2rem;
        color: #5a5a70;
        font-size: 0.8rem;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 2rem;
    }
    
    .stProgress > div > div {
        background: linear-gradient(90deg, #00b4d8, #90e0ef);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/ticket_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        le = joblib.load('models/label_encoder.pkl')
        accuracy = joblib.load('models/accuracy.pkl')
        return model, vectorizer, le, accuracy
    except:
        return None, None, None, None

# ============================================================================
# TEXT CLEANING
# ============================================================================
def clean_text(text):
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        clean_words = [w for w in words if w not in stop_words and len(w) > 2]
        return ' '.join(clean_words)
    except:
        return ' '.join([w for w in str(text).lower().split() if len(w) > 2])

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_single(ticket_text, model, vectorizer, le):
    cleaned = clean_text(ticket_text)
    features = vectorizer.transform([cleaned])
    pred_encoded = model.predict(features)[0]
    priority = le.inverse_transform([pred_encoded])[0]
    confidence = max(model.predict_proba(features)[0])
    return priority, confidence

# ============================================================================
# SAFE CSV READER FUNCTION
# ============================================================================
def safe_read_csv(uploaded_file):
    """Safely read CSV file with error handling"""
    try:
        # Try normal read first
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        try:
            # Try with different encoding
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin1')
            return df, None
        except:
            try:
                # Try with error handling
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, on_bad_lines='skip')
                return df, None
            except:
                try:
                    # Manual parsing
                    uploaded_file.seek(0)
                    content = uploaded_file.getvalue().decode('utf-8')
                    lines = content.split('\n')
                    tickets = []
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith(',') and len(line) > 5:
                            # Remove quotes and clean
                            clean_line = line.replace('"', '').replace("'", "")
                            tickets.append(clean_line)
                    
                    if len(tickets) > 1:
                        df = pd.DataFrame({'ticket': tickets[1:]})
                        return df, None
                    else:
                        return None, "No valid data found"
                except Exception as e2:
                    return None, f"Error reading file: {str(e2)}"

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'analytics' not in st.session_state:
    st.session_state.analytics = {
        'total': 0,
        'high': 0,
        'medium': 0,
        'low': 0,
        'history': []
    }

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="premium-header">
    <h1>⚡ Enterprise Support Intelligence</h1>
    <p>AI-Powered Ticket Classification & Prioritization Platform</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL
# ============================================================================
model, vectorizer, le, accuracy = load_model()

if model is None:
    st.error("⚠️ Model not found! Please run: python train_final.py")
    st.stop()

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🎯 Platform Overview")
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model Accuracy", f"{accuracy*100:.0f}%")
    with col2:
        st.metric("Total Tickets", st.session_state.analytics['total'])
    
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Priority Guide")
    st.markdown("🔴 **HIGH** - System down, urgent, emergency")
    st.markdown("🟠 **MEDIUM** - Billing, refund, account update")
    st.markdown("🟢 **LOW** - Questions, how-to, general info")
    
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("### 🔍 Search Tickets")
    search_term = st.text_input("", placeholder="Enter keyword...", label_visibility="collapsed")
    
    if search_term and len(st.session_state.analytics['history']) > 0:
        df_search = pd.DataFrame(st.session_state.analytics['history'])
        results = df_search[df_search['ticket'].str.contains(search_term, case=False, na=False)]
        if len(results) > 0:
            st.markdown(f"**Found {len(results)} matches:**")
            for _, row in results.head(5).iterrows():
                icon = "🔴" if row['priority'] == 'High' else "🟠" if row['priority'] == 'Medium' else "🟢"
                st.markdown(f"- {icon} {row['ticket'][:35]}...")
    
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
    
    if st.button("🔄 Reset Analytics", use_container_width=True):
        st.session_state.analytics = {'total': 0, 'high': 0, 'medium': 0, 'low': 0, 'history': []}
        st.rerun()

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🎯 CLASSIFY", "📊 ANALYTICS", "📁 BATCH"])

# ============================================================================
# TAB 1: CLASSIFY
# ============================================================================
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### ✨ Customer Ticket Input")
        
        ticket_text = st.text_area(
            "",
            height=140,
            placeholder="Paste customer ticket here...\n\nExample: URGENT! My website is completely down. Customers cannot access anything!",
            label_visibility="collapsed"
        )
        
        st.markdown("**Quick Test Examples:**")
        ex1, ex2, ex3 = st.columns(3)
        
        with ex1:
            if st.button("🔴 Urgent Issue", use_container_width=True):
                ticket_text = "URGENT! Website crashed, system down, customers cannot access anything!"
        
        with ex2:
            if st.button("🟠 Billing Issue", use_container_width=True):
                ticket_text = "I was charged twice for my subscription this month. Please refund immediately."
        
        with ex3:
            if st.button("🟢 General Question", use_container_width=True):
                ticket_text = "How do I reset my password? I forgot my current password."
        
        if st.button("🚀 Classify Ticket", use_container_width=True):
            if ticket_text.strip():
                with st.spinner("Analyzing ticket..."):
                    priority, confidence = predict_single(ticket_text, model, vectorizer, le)
                    
                    st.session_state.analytics['total'] += 1
                    if priority == 'High':
                        st.session_state.analytics['high'] += 1
                    elif priority == 'Medium':
                        st.session_state.analytics['medium'] += 1
                    else:
                        st.session_state.analytics['low'] += 1
                    
                    st.session_state.analytics['history'].append({
                        'ticket': ticket_text[:60],
                        'priority': priority,
                        'confidence': confidence,
                        'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    st.session_state['result'] = {
                        'priority': priority,
                        'confidence': confidence,
                        'ticket': ticket_text
                    }
                    st.rerun()
            else:
                st.warning("Please enter a ticket description")
    
    with col2:
        st.markdown("### 📈 System Status")
        st.markdown(f"**Model:** Random Forest Classifier")
        st.markdown(f"**Accuracy:** {accuracy*100:.0f}%")
        st.markdown(f"**Status:** 🟢 Operational")
        st.markdown(f"**Predictions:** {st.session_state.analytics['total']}")
    
    if 'result' in st.session_state:
        res = st.session_state['result']
        
        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔮 Classification Result")
        
        if res['priority'] == 'High':
            st.markdown(f"""
            <div class="priority-premium-high">
                <div class="priority-title">🔴 PRIORITY: HIGH - CRITICAL</div>
                <div class="priority-sub">Immediate action required | Response within 1 hour</div>
            </div>
            """, unsafe_allow_html=True)
        elif res['priority'] == 'Medium':
            st.markdown(f"""
            <div class="priority-premium-medium">
                <div class="priority-title">🟠 PRIORITY: MEDIUM</div>
                <div class="priority-sub">Action required | Response within 4 hours</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="priority-premium-low">
                <div class="priority-title">🟢 PRIORITY: LOW</div>
                <div class="priority-sub">Normal handling | Response within 24 hours</div>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f'<div class="premium-action"><strong>📊 Confidence Score</strong></div>', unsafe_allow_html=True)
            st.progress(res['confidence'])
            st.caption(f"{res['confidence']*100:.1f}% confident")
        
        with col2:
            st.markdown(f'<div class="premium-action"><strong>⚡ Recommended Actions</strong></div>', unsafe_allow_html=True)
            if res['priority'] == 'High':
                st.markdown("• Notify manager immediately")
                st.markdown("• Escalate to senior support")
                st.markdown("• 4-hour resolution SLA")
            elif res['priority'] == 'Medium':
                st.markdown("• Assign to support team")
                st.markdown("• Send email acknowledgment")
                st.markdown("• 24-hour resolution SLA")
            else:
                st.markdown("• Add to regular queue")
                st.markdown("• No urgent notification")
                st.markdown("• 72-hour resolution SLA")
        
        with st.expander("📝 View Original Ticket"):
            st.write(res['ticket'])

# ============================================================================
# TAB 2: ANALYTICS
# ============================================================================
with tab2:
    st.markdown("### 📊 Analytics Dashboard")
    
    total = st.session_state.analytics['total']
    
    if total > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-value">{total}</div>
                <div class="card-label">Total Tickets</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            high_pct = (st.session_state.analytics['high'] / total) * 100
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-value" style="background: linear-gradient(135deg, #ff006e, #ff6b6b); -webkit-background-clip: text;">{st.session_state.analytics['high']}</div>
                <div class="card-label">High Priority ({high_pct:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            med_pct = (st.session_state.analytics['medium'] / total) * 100
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-value" style="background: linear-gradient(135deg, #ffb703, #ffd166); -webkit-background-clip: text;">{st.session_state.analytics['medium']}</div>
                <div class="card-label">Medium Priority ({med_pct:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            low_pct = (st.session_state.analytics['low'] / total) * 100
            st.markdown(f"""
            <div class="premium-card">
                <div class="card-value" style="background: linear-gradient(135deg, #06d6a0, #6bcf7f); -webkit-background-clip: text;">{st.session_state.analytics['low']}</div>
                <div class="card-label">Low Priority ({low_pct:.0f}%)</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Priority Distribution")
            priority_data = pd.DataFrame({
                'Priority': ['High', 'Medium', 'Low'],
                'Count': [st.session_state.analytics['high'], st.session_state.analytics['medium'], st.session_state.analytics['low']]
            })
            fig = px.bar(priority_data, x='Priority', y='Count', color='Priority',
                        color_discrete_map={'High': '#ff006e', 'Medium': '#ffb703', 'Low': '#06d6a0'},
                        text='Count')
            fig.update_layout(plot_bgcolor='rgba(18,18,30,0.5)', paper_bgcolor='rgba(0,0,0,0)', font_color='#a0a0b5', height=400)
            fig.update_traces(textposition='outside')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Model Performance")
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=accuracy*100,
                title={'text': "Accuracy Score", 'font': {'color': '#a0a0b5'}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#00b4d8"},
                    'steps': [
                        {'range': [0, 50], 'color': 'rgba(255,0,110,0.2)'},
                        {'range': [50, 75], 'color': 'rgba(255,183,3,0.2)'},
                        {'range': [75, 100], 'color': 'rgba(6,214,160,0.2)'}
                    ]
                }
            ))
            fig.update_layout(height=400, paper_bgcolor='rgba(0,0,0,0)', font_color='#a0a0b5')
            st.plotly_chart(fig, use_container_width=True)
        
        if len(st.session_state.analytics['history']) > 0:
            st.markdown("#### 📋 Recent Activity")
            df_recent = pd.DataFrame(st.session_state.analytics['history'][-10:][::-1])
            df_recent['confidence'] = df_recent['confidence'].apply(lambda x: f"{x*100:.0f}%")
            df_recent.columns = ['Ticket', 'Priority', 'Confidence', 'Time']
            st.dataframe(df_recent[['Time', 'Ticket', 'Priority', 'Confidence']], use_container_width=True, hide_index=True)
        
        st.markdown("#### 📥 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export CSV", use_container_width=True):
                if len(st.session_state.analytics['history']) > 0:
                    df_export = pd.DataFrame(st.session_state.analytics['history'])
                    df_export['confidence'] = df_export['confidence'].apply(lambda x: f"{x*100:.0f}%")
                    csv = df_export.to_csv(index=False).encode('utf-8')
                    st.download_button("Download", csv, f"ticket_data_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv")
        
        with col2:
            if st.button("📋 Generate Report", use_container_width=True):
                report = f"""
                **ENTERPRISE SUPPORT REPORT**
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                Total Tickets: {total}
                High Priority: {st.session_state.analytics['high']} ({(st.session_state.analytics['high']/total)*100:.0f}%)
                Medium Priority: {st.session_state.analytics['medium']} ({(st.session_state.analytics['medium']/total)*100:.0f}%)
                Low Priority: {st.session_state.analytics['low']} ({(st.session_state.analytics['low']/total)*100:.0f}%)
                Model Accuracy: {accuracy*100:.0f}%
                """
                st.info(report)
    else:
        st.info("📊 No data yet. Classify some tickets to see analytics!")

# ============================================================================
# TAB 3: BATCH UPLOAD (FIXED VERSION)
# ============================================================================
with tab3:
    st.markdown("### 📁 Batch Processing")
    st.markdown("Upload a CSV or TXT file with tickets to classify in bulk")
    
    st.markdown("""
    <div style="background: rgba(0,180,216,0.1); padding: 1rem; border-radius: 12px; margin: 1rem 0;">
        <strong>📋 Accepted Formats:</strong><br>
        • <strong>CSV:</strong> File with column named 'ticket', 'description', 'text', or 'issue'<br>
        • <strong>TXT:</strong> One ticket per line<br>
        • <strong>Simple CSV:</strong> Just a list of tickets (no header)
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=['csv', 'txt'], label_visibility="collapsed")
    
    if uploaded_file is not None:
        # Check file type
        if uploaded_file.name.endswith('.txt'):
            # Handle TXT file
            content = uploaded_file.getvalue().decode('utf-8')
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            df_batch = pd.DataFrame({'ticket': lines})
            st.success(f"✅ Loaded {len(df_batch)} tickets from TXT file")
            text_col = 'ticket'
        else:
            # Handle CSV with safe parsing
            df_batch, error = safe_read_csv(uploaded_file)
            
            if error:
                st.error(f"Error reading file: {error}")
                st.stop()
            
            # Find the text column
            text_col = None
            possible_columns = ['ticket', 'description', 'text', 'issue', 'message', 'Ticket', 'Description', 'Text']
            
            for col in possible_columns:
                if col in df_batch.columns:
                    text_col = col
                    break
            
            # If no standard column found, use first column
            if text_col is None and len(df_batch.columns) > 0:
                text_col = df_batch.columns[0]
                st.info(f"Using column: '{text_col}' for tickets")
            
            st.success(f"✅ Loaded {len(df_batch)} tickets from CSV file")
        
        if text_col and len(df_batch) > 0:
            # Limit batch size
            if len(df_batch) > 100:
                st.warning(f"⚠️ Limiting to first 100 tickets (found {len(df_batch)})")
                df_batch = df_batch.head(100)
            
            # Preview
            with st.expander("Preview first 5 tickets"):
                st.dataframe(df_batch[text_col].head(), use_container_width=True)
            
            if st.button("🚀 Process Batch", use_container_width=True):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, ticket in enumerate(df_batch[text_col]):
                    status_text.text(f"Processing ticket {i+1} of {len(df_batch)}...")
                    
                    try:
                        ticket_str = str(ticket).strip()
                        if ticket_str and len(ticket_str) > 5:
                            priority, confidence = predict_single(ticket_str, model, vectorizer, le)
                            results.append({
                                'Ticket': ticket_str[:100],
                                'Priority': priority,
                                'Confidence': f"{confidence*100:.0f}%"
                            })
                            
                            # Update analytics
                            st.session_state.analytics['total'] += 1
                            if priority == 'High':
                                st.session_state.analytics['high'] += 1
                            elif priority == 'Medium':
                                st.session_state.analytics['medium'] += 1
                            else:
                                st.session_state.analytics['low'] += 1
                            
                            st.session_state.analytics['history'].append({
                                'ticket': ticket_str[:60],
                                'priority': priority,
                                'confidence': confidence,
                                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                        else:
                            results.append({'Ticket': ticket_str[:100], 'Priority': 'Skipped', 'Confidence': 'Empty'})
                    except Exception as e:
                        results.append({'Ticket': ticket_str[:100], 'Priority': 'Error', 'Confidence': str(e)[:20]})
                    
                    progress_bar.progress((i + 1) / len(df_batch))
                
                status_text.text("✅ Batch processing complete!")
                
                st.markdown("### 📊 Batch Results")
                df_results = pd.DataFrame(results)
                st.dataframe(df_results, use_container_width=True, hide_index=True)
                
                # Summary stats
                if 'Priority' in df_results.columns:
                    st.markdown("### 📈 Summary")
                    col1, col2, col3 = st.columns(3)
                    valid_results = df_results[df_results['Priority'].isin(['High', 'Medium', 'Low'])]
                    if len(valid_results) > 0:
                        summary = valid_results['Priority'].value_counts()
                        with col1:
                            high_count = summary.get('High', 0)
                            st.metric("🔴 High Priority", high_count, f"{high_count/len(valid_results)*100:.0f}%")
                        with col2:
                            med_count = summary.get('Medium', 0)
                            st.metric("🟠 Medium Priority", med_count, f"{med_count/len(valid_results)*100:.0f}%")
                        with col3:
                            low_count = summary.get('Low', 0)
                            st.metric("🟢 Low Priority", low_count, f"{low_count/len(valid_results)*100:.0f}%")
                
                # Download results
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Batch Results",
                    csv,
                    f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        else:
            st.error("No valid tickets found in the file")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="premium-footer">
    <p>⚡ Enterprise Support Intelligence Platform | Powered by Random Forest ML</p>
</div>
""", unsafe_allow_html=True)