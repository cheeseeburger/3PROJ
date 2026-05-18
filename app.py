import streamlit as st
import pandas as pd
import plotly.express as px
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download necessary internal AI resources cleanly
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

# Set up clean web page presentation
st.set_page_config(page_title="Enterprise Intelligence Dashboard", layout="wide")

st.title("💼 Enterprise Active Intelligence Dashboard")
st.subheader("Simulated Real-Time Risk & Project Blocker Detection Pipeline")
st.markdown("Inspired by production-grade industry workflows utilized by modern enterprise systems like OKAtlas.ai.")

# 1. Simulate data ingestion from corporate tools
@st.cache_data
def load_mock_corporate_data():
    data = {
        "Timestamp": ["2026-05-25 09:15", "2026-05-25 09:30", "2026-05-25 10:02", "2026-05-25 10:14", "2026-05-25 10:28"],
        "Source Channel": ["Slack (#dev-team)", "Email (Client-Success)", "Jira Ticket", "Slack (#marketing)", "Email (Finance)"],
        "Communication Log": [
            "The database server migration is blocked because our primary API credentials expired.",
            "Client confirmed they love the prototype! Moving forward to final contract signing.",
            "Critical security bug discovered in the user authentication loop. High deployment risk.",
            "The marketing campaign assets are delayed by a couple of days, adjusting timeline.",
            "Quarterly budget allocation approved. Funding is cleared for the engineering expansion."
        ]
    }
    return pd.DataFrame(data)

df = load_mock_corporate_data()

# 2. Initialize the AI Sentiment Analysis Engine
sia = SentimentIntensityAnalyzer()

def analyze_corporate_risks(dataframe):
    status_list = []
    confidence_scores = []
    
    for log in dataframe["Communication Log"]:
        score = sia.polarity_scores(log)["compound"]
        confidence_scores.append(abs(score))
        
        # Flag text based on underlying sentiment metrics
        if score <= -0.1:
            status_list.append("🔴 CRITICAL BLOCKER")
        elif score > -0.1 and score < 0.1:
            status_list.append("🟡 MINOR DELAY")
        else:
            status_list.append("🟢 HEALTHY PROGRESS")
            
    dataframe["System Status"] = status_list
    dataframe["AI Confidence Score"] = confidence_scores
    return dataframe

processed_df = analyze_corporate_risks(df)

# 3. Create Visual Layout Analytics
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### 📊 Operational Health Summary")
    status_counts = processed_df["System Status"].value_counts().reset_index()
    status_counts.columns = ["Status", "Total Count"]
    
    fig = px.pie(status_counts, values="Total Count", names="Status", 
                 color="Status", color_discrete_map={
                     "🔴 CRITICAL BLOCKER": "#EF553B",
                     "🟢 HEALTHY PROGRESS": "#00CC96",
                     "🟡 MINOR DELAY": "#FECB52"
                 })
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### 🔍 Real-Time Active Data Stream")
    st.dataframe(processed_df, use_container_width=True)

st.success("Data pipeline is executing successfully in real-time environment.") 
