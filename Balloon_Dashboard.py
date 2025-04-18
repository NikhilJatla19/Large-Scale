import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from plotly.subplots import make_subplots
import re

# Set layout config FIRST
st.set_page_config(layout="wide")

# Proper button navigation
if st.button("🔄 Go to In-Detailed Dashboard"):
    st.switch_page("pages/In-Detailed Dashboard.py")

st.title("🎈 Webex Sentiment Balloon – Three Sentiment Views")

# Load data
@st.cache_data
def load_data():
    df = pd.read_excel("Webex_Traceability_Merged_Cleaned.xlsx")
    df['at'] = pd.to_datetime(df['at'])
    df['Release Date'] = pd.to_datetime(df['Release Date'])

    def classify_sentiment(text):
        text = str(text).lower()
        if any(word in text for word in ["good", "great", "excellent", "love", "happy","nice","perfect", "clear", "easy", "Perfectoo"]):
            return "Positive"
        elif any(word in text for word in ["bad", "terrible", "hate", "issue", "problem", "poor", "worst"]):
            return "Negative"
        return "Neutral"

    df['Sentiment'] = df['content'].apply(classify_sentiment)
    return df

df = load_data()

# Sidebar filter
def version_key(v):
    nums = re.findall(r'\d+', str(v))
    return [int(n) for n in nums] if nums else [0]

version_list = sorted(df['appVersion'].dropna().unique().tolist(), key=version_key)
version_list.insert(0, "All Versions")
selected_version = st.sidebar.selectbox("Choose App Version", version_list)

# Filter based on version
if selected_version == "All Versions":
    version_df = df.copy()
else:
    version_df = df[df['appVersion'] == selected_version]

version_display = selected_version if selected_version != 'All Versions' else 'All Versions'

# Sentiment plotting setup
sentiment_order = ['Positive', 'Neutral', 'Negative']
sentiment_colors = {'Positive': 'green', 'Neutral': 'gold', 'Negative': 'red'}
score_min = 3.2
score_max = 5.0
min_size = 30
max_size = 80
pixel_to_data_ratio = 2 / 600

# Create subplots
fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.1, subplot_titles=sentiment_order)

for i, sentiment in enumerate(sentiment_order, start=1):
    sentiment_df = version_df[version_df['Sentiment'] == sentiment]
    if sentiment_df.empty:
        continue

    reviews = len(sentiment_df)
    avg_score = round(sentiment_df['score'].mean(), 2)
    duration_days = (sentiment_df['at'].max() - sentiment_df['at'].min()).days
    duration_scaled = max(duration_days / 15, 1.5)

    score_norm = (avg_score - score_min) / (score_max - score_min)
    score_norm = max(0, min(score_norm, 1))
    circle_pixel_radius = min_size + score_norm * (max_size - min_size)
    circle_data_radius = (circle_pixel_radius / 2) * pixel_to_data_ratio

    stick_visible_height = max(duration_scaled - circle_data_radius, 0.5)
    circle_y = stick_visible_height + circle_data_radius

    # Stick line
    fig.add_trace(go.Scatter(
        x=[0, 0],
        y=[0, stick_visible_height],
        mode='lines',
        line=dict(color='gray', width=4),
        showlegend=False,
        hoverinfo='skip'
    ), row=1, col=i)

    # Transparent markers for hover
    hover_y = np.linspace(0, stick_visible_height, 30)
    fig.add_trace(go.Scatter(
        x=[0]*len(hover_y),
        y=hover_y,
        mode='markers',
        marker=dict(size=10, color='rgba(0,0,0,0)'),
        hovertemplate=f"<b>Duration:</b> {duration_days} days",
        showlegend=False
    ), row=1, col=i)

    # Lollipop head (circle)
    fig.add_trace(go.Scatter(
        x=[0],
        y=[circle_y],
        mode='markers',
        marker=dict(size=circle_pixel_radius, color=sentiment_colors[sentiment], line=dict(color='black', width=1)),
        hovertemplate=f"""
        <b>Sentiment:</b> {sentiment}<br>
        <b>Avg Score:</b> {avg_score}<br>
        <b>Reviews:</b> {reviews}<br>
        <b>Duration:</b> {duration_days} days
        """,
        showlegend=False
    ), row=1, col=i)

# Layout update
fig.update_layout(
    height=600,
    width=1000,
    title_text=f"Ballon Visualization by Sentiment – {version_display}",
    plot_bgcolor='white',
    margin=dict(t=80, l=30, r=30, b=40)
)

for i in range(1, 4):
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=i)

# Display chart
st.plotly_chart(fig, use_container_width=True)

# Feature Descriptions
# Feature Descriptions
# Feature Descriptions
feature_raw = version_df['Feature Description'].dropna().tolist()

# Split descriptions by "#" and flatten the list, deduplicate, and sort
feature_set = set()
for entry in feature_raw:
    features = [desc.strip() for desc in entry.split('#') if desc.strip()]
    feature_set.update(features)

feature_list = sorted(feature_set)

if feature_list:
    st.markdown("---")
    st.markdown("### 🧩 Feature Descriptions")
    st.markdown(f"""
    <div style='max-height: 300px; overflow-y: auto; padding: 10px; border: 1px solid #eee; background-color: #fafafa; border-radius: 8px;'>
        <ul style='font-size: 15px; color: #333;'>
            {''.join([f"<li>{desc}</li>" for desc in feature_list])}
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("No feature descriptions found for the selected version(s).")
