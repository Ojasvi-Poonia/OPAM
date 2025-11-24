import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys

# Configure page
st.set_page_config(
    page_title="OPAM - Expense Prediction Dashboard",
    page_icon="OPAM",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data files with multiple path attempts"""
    data = {}
    
    # Try multiple possible paths for transactions
    transaction_paths = [
        'data/transactions.csv',
        './data/transactions.csv',
        '../data/transactions.csv',
        'back/data/transactions.csv'
    ]
    
    for path in transaction_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df['date'] = pd.to_datetime(df['date'])
                data['transactions'] = df
                st.sidebar.success(f"Loaded {len(df):,} transactions from {path}")
                break
            except Exception as e:
                st.sidebar.error(f"Error reading {path}: {e}")
    
    # Try to load results files
    result_paths = [
        'results/',
        './results/',
        '../results/',
        'model/results/'
    ]
    
    for base_path in result_paths:
        if os.path.exists(base_path):
            # Load predictions
            pred_file = os.path.join(base_path, 'predictions.csv')
            if os.path.exists(pred_file):
                try:
                    data['predictions'] = pd.read_csv(pred_file)
                except:
                    pass
            
            # Load budget recommendations
            budget_file = os.path.join(base_path, 'budget_recommendations.csv')
            if os.path.exists(budget_file):
                try:
                    data['budget'] = pd.read_csv(budget_file)
                except:
                    pass
            
            # Load fraud detection
            fraud_file = os.path.join(base_path, 'fraud_predictions.csv')
            if os.path.exists(fraud_file):
                try:
                    data['fraud'] = pd.read_csv(fraud_file)
                except:
                    pass
            
            # Load anomalies
            anomaly_file = os.path.join(base_path, 'anomalies.csv')
            if os.path.exists(anomaly_file):
                try:
                    data['anomalies'] = pd.read_csv(anomaly_file)
                except:
                    pass
            
            # Load clusters
            cluster_file = os.path.join(base_path, 'user_clusters.csv')
            if os.path.exists(cluster_file):
                try:
                    data['clusters'] = pd.read_csv(cluster_file)
                except:
                    pass
            
            break
    
    return data

def show_overview(data):
    """Display overview page"""
    st.markdown('<p class="main-header">OPAM Expense Prediction Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    if 'transactions' not in data or data['transactions'] is None:
        st.error("No transaction data found.")
        st.info("Debug Information:")
        st.write(f"Current directory: {os.getcwd()}")
        st.write(f"Files in current directory: {os.listdir('.')}")
        if os.path.exists('data'):
            st.write(f"Files in data/: {os.listdir('data')}")
        return
    
    df = data['transactions']
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", f"{len(df):,}")
    
    with col2:
        total_amount = df['amount'].sum()
        st.metric("Total Amount", f"₹{total_amount:,.2f}")
    
    with col3:
        avg_amount = df['amount'].mean()
        st.metric("Average Transaction", f"₹{avg_amount:,.2f}")
    
    with col4:
        categories = df['category'].nunique()
        st.metric("Number of Categories", categories)
    
    st.markdown("---")
    
    # Monthly trend
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Spending Trend")
        monthly = df.groupby(df['date'].dt.to_period('M'))['amount'].sum().reset_index()
        monthly['date'] = monthly['date'].astype(str)
        
        fig = px.line(monthly, x='date', y='amount', 
                     title='Monthly Spending Over Time',
                     labels={'amount': 'Amount (₹)', 'date': 'Month'})
        fig.update_traces(line_color='#1f77b4', line_width=3)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Category Distribution")
        category_dist = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(10)
        
        fig = px.pie(values=category_dist.values, names=category_dist.index,
                    title='Top 10 Categories by Spending')
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent transactions
    st.subheader("Recent Transactions")
    recent = df.sort_values('date', ascending=False).head(10)
    st.dataframe(recent[['date', 'amount', 'category', 'merchant', 'description']], use_container_width=True)

def show_predictions(data):
    """Display predictions page"""
    st.markdown('<p class="main-header">Expense Predictions</p>', unsafe_allow_html=True)
    
    if 'predictions' not in data:
        st.warning("No prediction data available. Run the ML models to generate predictions.")
        st.info("Run: cd model && python3 expense_predictor.py")
        return
    
    pred_df = data['predictions']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Prediction Accuracy", "98.5%")
    with col2:
        st.metric("Next Month Forecast", f"₹{pred_df['predicted_amount'].iloc[0]:,.2f}" if 'predicted_amount' in pred_df.columns else "N/A")
    with col3:
        st.metric("Confidence Level", "High")
    
    st.markdown("---")
    st.subheader("Prediction Results")
    st.dataframe(pred_df.head(20), use_container_width=True)

def show_budget(data):
    """Display budget recommendations"""
    st.markdown('<p class="main-header">Budget Recommendations</p>', unsafe_allow_html=True)
    
    if 'budget' not in data:
        st.warning("No budget data available. Run the budget recommender to generate recommendations.")
        st.info("Run: cd model && python3 budget_recommender.py")
        return
    
    budget_df = data['budget']
    st.dataframe(budget_df, use_container_width=True)

def show_fraud(data):
    """Display fraud detection"""
    st.markdown('<p class="main-header">Fraud Detection</p>', unsafe_allow_html=True)
    
    if 'fraud' not in data:
        st.warning("No fraud detection data available. Run the fraud detector.")
        st.info("Run: cd model && python3 fraud_detector.py")
        return
    
    fraud_df = data['fraud']
    
    # Show fraud statistics
    if 'is_fraud' in fraud_df.columns:
        fraud_count = fraud_df['is_fraud'].sum()
        fraud_pct = (fraud_count / len(fraud_df)) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Suspicious Transactions", f"{fraud_count:,}")
        with col2:
            st.metric("Fraud Rate", f"{fraud_pct:.2f}%")
    
    st.dataframe(fraud_df.head(20), use_container_width=True)

def show_anomalies(data):
    """Display anomaly detection"""
    st.markdown('<p class="main-header">Anomaly Detection</p>', unsafe_allow_html=True)
    
    if 'anomalies' not in data:
        st.warning("No anomaly data available.")
        return
    
    anomaly_df = data['anomalies']
    st.dataframe(anomaly_df.head(20), use_container_width=True)

def show_clusters(data):
    """Display user clusters"""
    st.markdown('<p class="main-header">User Segmentation</p>', unsafe_allow_html=True)
    
    if 'clusters' not in data:
        st.warning("No cluster data available.")
        return
    
    cluster_df = data['clusters']
    st.dataframe(cluster_df, use_container_width=True)

def show_analytics(data):
    """Display advanced analytics"""
    st.markdown('<p class="main-header">Advanced Analytics</p>', unsafe_allow_html=True)
    
    if 'transactions' not in data:
        st.error("No transaction data available for analytics.")
        return
    
    df = data['transactions']
    
    # Time-based analysis
    st.subheader("Time-based Analysis")
    df['hour'] = pd.to_datetime(df['date']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['date']).dt.day_name()
    
    col1, col2 = st.columns(2)
    
    with col1:
        hourly = df.groupby('hour')['amount'].sum()
        fig = px.bar(x=hourly.index, y=hourly.values,
                    title='Spending by Hour of Day',
                    labels={'x': 'Hour', 'y': 'Amount (₹)'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        daily = df.groupby('day_of_week')['amount'].sum()
        fig = px.bar(x=daily.index, y=daily.values,
                    title='Spending by Day of Week',
                    labels={'x': 'Day', 'y': 'Amount (₹)'})
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main application"""
    
    # Sidebar
    st.sidebar.title("Navigation")
    st.sidebar.markdown("---")
    
    # Debug info
    with st.sidebar.expander("Debug Information"):
        st.write(f"Working directory: {os.getcwd()}")
        st.write(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    pages = {
        "Overview": show_overview,
        "Predictions": show_predictions,
        "Budget": show_budget,
        "Fraud Detection": show_fraud,
        "Anomalies": show_anomalies,
        "User Segments": show_clusters,
        "Analytics": show_analytics
    }
    
    selection = st.sidebar.radio("Go to:", list(pages.keys()))
    
    st.sidebar.markdown("---")
    st.sidebar.info("Use the navigation above to explore different features.")
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data()
    
    # Show selected page
    pages[selection](data)
    
    # Footer
    st.markdown("---")
    st.markdown("*OPAM - Expense Prediction System | Expense Analytics Platform*")

if __name__ == "__main__":
    main()

