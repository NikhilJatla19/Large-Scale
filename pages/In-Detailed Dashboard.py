import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import re

# Load Data
@st.cache_data
def load_data():
    return pd.read_excel("Webex_Traceability_Merged_Cleaned.xlsx")

df = load_data()

# Convert dates
df['at'] = pd.to_datetime(df['at'], errors='coerce')
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')

# Title
st.title("📈 Webex Traceability Dashboard")

# Sidebar Filters
st.sidebar.header("Filters")
date_range = st.sidebar.date_input("Review Date Range", [df['at'].min(), df['at'].max()])

# Helper to extract numeric parts for proper sorting
def version_key(v):
    if pd.isna(v):
        return []
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', str(v))]

sorted_versions = sorted(df['Release Version'].dropna().unique(), key=version_key)

version_filter = st.sidebar.multiselect("Filter by Release Version", sorted_versions)


filtered_df = df[
    (df['at'] >= pd.to_datetime(date_range[0])) & 
    (df['at'] <= pd.to_datetime(date_range[1]))
]

if version_filter:
    filtered_df = filtered_df[filtered_df['Release Version'].isin(version_filter)]

# Sentiment Analysis (basic scoring)
def classify_sentiment(text):
    text = str(text).lower()
    if any(word in text for word in ["good", "great", "excellent", "love", "happy","nice"]):
        return "Positive"
    elif any(word in text for word in ["bad", "terrible", "hate", "issue", "problem", "poor"]):
        return "Negative"
    else:
        return "Neutral"

filtered_df["Sentiment"] = filtered_df["content"].apply(classify_sentiment)

# Chart 1: Reviews Over Time
st.subheader("📅 Reviews Over Time")
timeline = filtered_df.groupby(filtered_df['at'].dt.to_period("M")).size().reset_index(name='count')
timeline['at'] = timeline['at'].astype(str)
fig1 = px.line(timeline, x='at', y='count', title='Review Volume per Month')
st.plotly_chart(fig1)

# Chart 2: Sentiment Distribution
st.subheader("😀 Sentiment Breakdown")
fig2 = px.pie(filtered_df, names='Sentiment', title="Sentiment Distribution")
st.plotly_chart(fig2)

# Chart 3: Score Distribution
st.subheader("⭐ Score Distribution")
fig3 = px.histogram(filtered_df, x="score", nbins=6)
st.plotly_chart(fig3)


# Table: Traceability Between Features and Reviews
st.subheader("🔍 Traceability Mapping (Reviews ↔ Features)")
traceability = filtered_df[[
    "at", "content", "score", "Sentiment", "reviewCreatedVersion", "Feature Description", "Release Date"
]]

st.dataframe(traceability)


# Feature Impact Heatmap
st.subheader("🚀 Feature Impact Heatmap (Sentiment Over Time)")

# Prepare data
heatmap_data = filtered_df.copy()
heatmap_data['Month'] = heatmap_data['at'].dt.to_period('M').astype(str)
heatmap_pivot = (
    heatmap_data
    .groupby(['Release Version', 'Month'])
    .agg(avg_score=('score', 'mean'), review_count=('content', 'count'))
    .reset_index()
)

# Create Heatmap
fig4 = px.density_heatmap(
    heatmap_pivot, 
    x="Month", 
    y="Release Version", 
    z="avg_score",
    color_continuous_scale="RdYlGn", 
    hover_data=["review_count"],
    title="Feature Impact Heatmap: Average Sentiment by Version and Month"
)

st.plotly_chart(fig4)

heatmap_data = filtered_df.copy()
heatmap_data['Month'] = heatmap_data['at'].dt.to_period('M').astype(str)
heatmap_data['Sentiment'] = heatmap_data['content'].apply(classify_sentiment)

sentiment_pivot = (
    heatmap_data.groupby(['Release Version', 'Month', 'Sentiment'])
    .size()
    .reset_index(name='count')
)

fig5 = px.sunburst(
    sentiment_pivot,
    path=["Release Version", "Month", "Sentiment"],
    values="count",
    color="count",
    color_continuous_scale="RdBu",
    title="Sunburst: Sentiment Spread by Version & Time"
)

st.plotly_chart(fig5)

st.subheader("📅 Gantt Chart: Feature Relevance Timeline")

# Prepare Gantt data
gantt_df = (
    filtered_df.groupby("Release Version")
    .agg(
        Start=('Release Date', 'min'),
        End=('at', 'max'),
        Description=('Feature Description', 'first'),
        Reviews=('content', 'count')
    )
    .reset_index()
)

# Cap entries for better visual clarity
gantt_df = gantt_df.sort_values("Start").head(20)

# Format for Gantt
gantt_data = [
    dict(Task=row['Release Version'],
         Start=str(row['Start']),
         Finish=str(row['End']),
         Resource=row['Description'][:40] + '...',
         Reviews=row['Reviews'])
    for _, row in gantt_df.iterrows()
]

fig_gantt = ff.create_gantt(
    gantt_data,
    group_tasks=True,
    title="Feature Timeline by Review Activity",
    bar_width=0.2,
    showgrid_x=True,
    showgrid_y=True
)

st.plotly_chart(fig_gantt)

# Download Option
st.download_button("Download Filtered Data", data=traceability.to_csv(index=False), file_name="filtered_traceability.csv")
