import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from influxdb_client import InfluxDBClient
from datetime import datetime, timedelta
import json

# Configure the page
st.set_page_config(
    page_title="GPT Service Metrics Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize InfluxDB client
INFLUX_URL = os.getenv("INFLUX_URL")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN")
INFLUX_ORG = os.getenv("INFLUX_ORG")
INFLUX_BUCKET = os.getenv("INFLUX_BUCKET")
client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
query_api = client.query_api()

def fetch_metrics(service_name: str, metric_name: str = None, limit: int = 1000):
    """Fetch metrics from InfluxDB based on service and (optionally) metric name"""
    # Time range based on 'limit' in minutes
    start_duration = f"-{limit}m"
    # Build Flux query
    filter_measure = f'r["_measurement"] == "{metric_name}"' if metric_name else "true"
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start_duration})
  |> filter(fn: (r) => r["service_name"] == "{service_name}" and {filter_measure})
  |> sort(columns: ["_time"])
'''  
    try:
        tables = query_api.query(org=INFLUX_ORG, query=flux)
        metrics = []
        for table in tables:
            for record in table.records:
                metrics.append({
                    "timestamp": record.get_time().isoformat(),
                    "metric_name": record.get_measurement(),
                    "metric_value": record.get_value(),
                    "metadata": None,
                    "tags": {}
                })
        return metrics
    except Exception as e:
        st.error(f"Error querying InfluxDB: {e}")
        return []

def fetch_events(service_name: str, event_type: str = None, limit: int = 1000):
    """Fetch events from InfluxDB"""
    # Fetch events from InfluxDB
    start_duration = f"-{limit}m"
    filter_meas = 'r["_measurement"] == "events"'
    filter_meas += f' and r["service_name"] == "{service_name}"'
    if event_type:
        filter_meas += f' and r["event_type"] == "{event_type}"'
    flux = f'''
from(bucket: "{INFLUX_BUCKET}")
  |> range(start: {start_duration})
  |> filter(fn: (r) => {filter_meas})
  |> sort(columns: ["_time"])
'''
    try:
        tables = query_api.query(org=INFLUX_ORG, query=flux)
        events = []
        for table in tables:
            for record in table.records:
                fields = {k: v for k, v in record.values.items()
                          if k not in ("_value", "_time", "_measurement", "service_name", "event_type")}
                events.append({
                    "timestamp": record.get_time().isoformat(),
                    "event_type": record.values.get("event_type"),
                    "event_data": fields,
                    "status": record.values.get("status")
                })
        return events
    except Exception as e:
        st.error(f"Error querying InfluxDB events: {e}")
        return []

def create_time_series_plot(df, metric_name, title):
    """Create a time series plot using plotly"""
    fig = px.line(
        df,
        x="timestamp",
        y="metric_value",
        title=title,
        labels={"metric_value": metric_name, "timestamp": "Time"}
    )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title=metric_name,
        hovermode="x unified"
    )
    return fig

def create_heatmap(df, x_col, y_col, value_col, title):
    """Create a heatmap using plotly"""
    fig = px.density_heatmap(
        df,
        x=x_col,
        y=y_col,
        z=value_col,
        title=title
    )
    return fig

# Main dashboard
st.title("GPT Service Metrics Dashboard")

# Sidebar filters
st.sidebar.header("Filters")
service_name = st.sidebar.selectbox(
    "Select Service",
    ["gpt-service", "rag-api", "metrics-service"]
)

time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last Hour", "Last 24 Hours", "Last 7 Days", "Last 30 Days"]
)

# Convert time range to limit
time_limits = {
    "Last Hour": 60,
    "Last 24 Hours": 1440,
    "Last 7 Days": 10080,
    "Last 30 Days": 43200
}
limit = time_limits[time_range]

# Fetch and display metrics
metrics = fetch_metrics(service_name, limit=limit)
if metrics:
    df = pd.DataFrame(metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["metric_value"] = pd.to_numeric(df["metric_value"], errors="coerce")
    
    # Group metrics by name
    metric_names = df["metric_name"].unique()
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Time Series", "Heatmap", "Raw Data"])
    
    with tab1:
        st.header("Time Series Metrics")
        for metric_name in metric_names:
            metric_df = df[df["metric_name"] == metric_name]
            fig = create_time_series_plot(metric_df, metric_name, f"{metric_name} Over Time")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Metric Heatmap")
        # Create a pivot table for the heatmap
        pivot_df = df.pivot_table(
            index=df["timestamp"].dt.hour,
            columns=df["timestamp"].dt.date,
            values="metric_value",
            aggfunc="mean"
        )
        fig = create_heatmap(
            df,
            x="timestamp",
            y="metric_name",
            value_col="metric_value",
            title="Metric Distribution Heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Raw Metrics Data")
        st.dataframe(df)

# Fetch and display events
st.header("Recent Events")
events = fetch_events(service_name, limit=limit)
if events:
    events_df = pd.DataFrame(events)
    events_df["timestamp"] = pd.to_datetime(events_df["timestamp"])
    
    # Display events in a table
    st.dataframe(events_df)
    
    # Create event timeline
    fig = px.timeline(
        events_df,
        x_start="timestamp",
        y="event_type",
        color="status",
        title="Event Timeline"
    )
    st.plotly_chart(fig, use_container_width=True)

# Add summary statistics
if metrics:
    st.header("Summary Statistics")
    summary_df = df.groupby("metric_name").agg({
        "metric_value": ["mean", "std", "min", "max"]
    }).round(2)
    st.dataframe(summary_df) 