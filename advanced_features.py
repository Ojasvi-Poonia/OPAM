"""
OPAM - Advanced Features Module
Real-time alerts, data export, report generation, and monitoring

Author: Alife
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import base64
from typing import Dict, List, Tuple

class AdvancedFeatures:
    """Advanced dashboard features"""
    
    def __init__(self):
        self.alert_thresholds = {
            'high_spending': 50000,
            'fraud_score': 70,
            'anomaly_count': 10
        }
    
    def generate_alerts(self, data: Dict) -> List[Dict]:
        """Generate real-time alerts based on data"""
        alerts = []
        
        if 'transactions' in data:
            df = data['transactions']
            
            # High spending alert
            recent_spending = df[df['date'] >= df['date'].max() - timedelta(days=7)]['amount'].sum()
            if recent_spending > self.alert_thresholds['high_spending']:
                alerts.append({
                    'type': 'warning',
                    'title': '⚠️ High Spending Alert',
                    'message': f'Last 7 days spending: ₹{recent_spending:,.2f} exceeds threshold',
                    'timestamp': datetime.now()
                })
        
        if 'fraud_scores' in data:
            fraud_df = data['fraud_scores']
            high_risk = len(fraud_df[fraud_df['fraud_score'] >= self.alert_thresholds['fraud_score']])
            
            if high_risk > 0:
                alerts.append({
                    'type': 'error',
                    'title': '🚨 Fraud Alert',
                    'message': f'{high_risk} high-risk transactions detected',
                    'timestamp': datetime.now()
                })
        
        if 'anomalies' in data:
            anomaly_count = len(data['anomalies'])
            if anomaly_count >= self.alert_thresholds['anomaly_count']:
                alerts.append({
                    'type': 'info',
                    'title': '🔍 Anomaly Alert',
                    'message': f'{anomaly_count} anomalous transactions found',
                    'timestamp': datetime.now()
                })
        
        return alerts
    
    def display_alerts(self, alerts: List[Dict]):
        """Display alerts in dashboard"""
        if not alerts:
            st.success("✅ All systems normal - no alerts")
            return
        
        st.subheader("🔔 Active Alerts")
        
        for alert in alerts:
            if alert['type'] == 'error':
                st.error(f"{alert['title']}: {alert['message']}")
            elif alert['type'] == 'warning':
                st.warning(f"{alert['title']}: {alert['message']}")
            else:
                st.info(f"{alert['title']}: {alert['message']}")
    
    def export_to_csv(self, df: pd.DataFrame, filename: str) -> bytes:
        """Export dataframe to CSV"""
        return df.to_csv(index=False).encode('utf-8')
    
    def export_to_excel(self, data_dict: Dict[str, pd.DataFrame], filename: str) -> bytes:
        """Export multiple dataframes to Excel with sheets"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, df in data_dict.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output.getvalue()
    
    def create_download_link(self, data: bytes, filename: str, file_type: str) -> str:
        """Create download link for file"""
        b64 = base64.b64encode(data).decode()
        return f'<a href="data:{file_type};base64,{b64}" download="{filename}">Download {filename}</a>'
    
    def generate_summary_report(self, data: Dict) -> str:
        """Generate text summary report"""
        report = []
        report.append("="*80)
        report.append("OPAM EXPENSE ANALYSIS REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*80)
        report.append("")
        
        if 'transactions' in data:
            df = data['transactions']
            report.append("TRANSACTION SUMMARY")
            report.append("-"*80)
            report.append(f"Total Transactions: {len(df):,}")
            report.append(f"Total Spending: ₹{df['amount'].sum():,.2f}")
            report.append(f"Average Transaction: ₹{df['amount'].mean():,.2f}")
            report.append(f"Date Range: {df['date'].min()} to {df['date'].max()}")
            report.append("")
            
            # Top categories
            report.append("TOP 5 CATEGORIES BY SPENDING")
            report.append("-"*80)
            top_cats = df.groupby('category')['amount'].sum().sort_values(ascending=False).head(5)
            for i, (cat, amount) in enumerate(top_cats.items(), 1):
                report.append(f"{i}. {cat}: ₹{amount:,.2f}")
            report.append("")
        
        if 'budget_recs' in data:
            budget_df = data['budget_recs']
            report.append("BUDGET RECOMMENDATIONS")
            report.append("-"*80)
            total_savings = budget_df['potential_savings'].sum()
            report.append(f"Monthly Savings Potential: ₹{total_savings:,.2f}")
            report.append(f"Annual Savings Potential: ₹{total_savings * 12:,.2f}")
            report.append("")
        
        if 'fraud_scores' in data:
            fraud_df = data['fraud_scores']
            report.append("FRAUD DETECTION SUMMARY")
            report.append("-"*80)
            risk_counts = fraud_df['risk_level'].value_counts()
            for level in ['Critical', 'High', 'Medium', 'Low']:
                count = risk_counts.get(level, 0)
                report.append(f"{level} Risk: {count:,} transactions")
            report.append("")
        
        if 'cluster_summary' in data:
            cluster_df = data['cluster_summary']
            report.append("USER SEGMENTATION")
            report.append("-"*80)
            report.append(f"Total Segments: {len(cluster_df)}")
            for _, row in cluster_df.iterrows():
                report.append(f"\n{row['cluster_name']}")
                report.append(f"  Users: {row['size']:,}")
                report.append(f"  Avg Transaction: ₹{row['avg_amount']:,.2f}")
        
        report.append("")
        report.append("="*80)
        report.append("End of Report")
        report.append("="*80)
        
        return "\n".join(report)
    
    def calculate_kpis(self, data: Dict) -> Dict:
        """Calculate key performance indicators"""
        kpis = {}
        
        if 'transactions' in data:
            df = data['transactions']
            
            # Spending KPIs
            kpis['total_spending'] = df['amount'].sum()
            kpis['avg_transaction'] = df['amount'].mean()
            kpis['transaction_count'] = len(df)
            
            # Growth metrics
            df['year_month'] = df['date'].dt.to_period('M')
            monthly = df.groupby('year_month')['amount'].sum()
            if len(monthly) >= 2:
                current_month = monthly.iloc[-1]
                previous_month = monthly.iloc[-2]
                kpis['mom_growth'] = ((current_month - previous_month) / previous_month) * 100
            else:
                kpis['mom_growth'] = 0
            
            # Category concentration
            category_spending = df.groupby('category')['amount'].sum()
            top_category_pct = (category_spending.max() / category_spending.sum()) * 100
            kpis['category_concentration'] = top_category_pct
        
        if 'fraud_scores' in data:
            fraud_df = data['fraud_scores']
            high_risk_count = len(fraud_df[fraud_df['risk_level'].isin(['High', 'Critical'])])
            kpis['high_risk_transactions'] = high_risk_count
            kpis['fraud_rate'] = (high_risk_count / len(fraud_df)) * 100 if len(fraud_df) > 0 else 0
        
        if 'budget_recs' in data:
            budget_df = data['budget_recs']
            kpis['savings_potential'] = budget_df['potential_savings'].sum()
            kpis['budget_efficiency'] = (kpis['savings_potential'] / budget_df['current_monthly'].sum()) * 100 if budget_df['current_monthly'].sum() > 0 else 0
        
        return kpis
    
    def show_comparison(self, data: Dict, period1: str, period2: str):
        """Compare two time periods"""
        if 'transactions' not in data:
            st.warning("No transaction data available for comparison")
            return
        
        df = data['transactions']
        
        # Split data by periods
        df1 = df[df['date'].dt.to_period('M').astype(str) == period1]
        df2 = df[df['date'].dt.to_period('M').astype(str) == period2]
        
        if df1.empty or df2.empty:
            st.warning("One or both periods have no data")
            return
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                f"Spending - {period1}",
                f"₹{df1['amount'].sum():,.2f}",
                delta=None
            )
        
        with col2:
            st.metric(
                f"Spending - {period2}",
                f"₹{df2['amount'].sum():,.2f}",
                delta=None
            )
        
        with col3:
            change = df2['amount'].sum() - df1['amount'].sum()
            change_pct = (change / df1['amount'].sum()) * 100 if df1['amount'].sum() > 0 else 0
            st.metric(
                "Change",
                f"₹{change:,.2f}",
                delta=f"{change_pct:+.1f}%"
            )
    
    def predict_next_month_simple(self, df: pd.DataFrame) -> Tuple[float, float, float]:
        """Simple prediction using moving average"""
        df['year_month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('year_month')['amount'].sum()
        
        # Use last 3 months average
        last_3_avg = monthly.tail(3).mean()
        
        # Calculate standard deviation for confidence interval
        std_dev = monthly.tail(3).std()
        
        # Prediction with confidence interval
        prediction = last_3_avg
        lower_bound = prediction - std_dev
        upper_bound = prediction + std_dev
        
        return prediction, lower_bound, upper_bound
    
    def spending_health_score(self, data: Dict) -> Tuple[int, str]:
        """Calculate overall spending health score (0-100)"""
        score = 100
        factors = []
        
        if 'transactions' in data and 'budget_recs' in data:
            # Budget adherence
            budget_df = data['budget_recs']
            over_budget = len(budget_df[budget_df['current_monthly'] > budget_df['recommended_budget']])
            if over_budget > len(budget_df) * 0.5:
                score -= 20
                factors.append("Over budget in multiple categories")
        
        if 'fraud_scores' in data:
            # Fraud risk
            fraud_df = data['fraud_scores']
            high_risk_pct = (len(fraud_df[fraud_df['risk_level'].isin(['High', 'Critical'])]) / len(fraud_df)) * 100
            if high_risk_pct > 5:
                score -= 15
                factors.append("High fraud risk detected")
            elif high_risk_pct > 2:
                score -= 5
                factors.append("Moderate fraud risk")
        
        if 'anomalies' in data and 'transactions' in data:
            # Anomaly rate
            anomaly_rate = (len(data['anomalies']) / len(data['transactions'])) * 100
            if anomaly_rate > 5:
                score -= 15
                factors.append("High anomaly rate")
            elif anomaly_rate > 2:
                score -= 5
                factors.append("Some anomalies detected")
        
        if 'transactions' in data:
            # Spending volatility
            df = data['transactions']
            df['year_month'] = df['date'].dt.to_period('M')
            monthly = df.groupby('year_month')['amount'].sum()
            if len(monthly) >= 3:
                cv = monthly.std() / monthly.mean() if monthly.mean() > 0 else 0
                if cv > 0.3:
                    score -= 10
                    factors.append("High spending volatility")
        
        # Generate summary
        if score >= 90:
            summary = "Excellent - Healthy spending patterns"
        elif score >= 75:
            summary = "Good - Minor improvements needed"
        elif score >= 60:
            summary = "Fair - Several areas need attention"
        else:
            summary = "Poor - Significant improvements required"
        
        return max(0, score), summary

def show_advanced_features(data: Dict):
    """Display advanced features page"""
    st.title("🚀 Advanced Features")
    st.markdown("---")
    
    features = AdvancedFeatures()
    
    # Tabs for different features
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔔 Alerts", "📊 KPIs", "📥 Export", "🏥 Health Score", "📈 Comparison"
    ])
    
    with tab1:
        st.subheader("Real-Time Alerts")
        alerts = features.generate_alerts(data)
        features.display_alerts(alerts)
        
        st.markdown("---")
        st.subheader("⚙️ Alert Configuration")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            high_spending = st.number_input(
                "High Spending Threshold (₹)",
                value=50000,
                step=1000
            )
        with col2:
            fraud_threshold = st.slider(
                "Fraud Score Threshold",
                0, 100, 70
            )
        with col3:
            anomaly_threshold = st.number_input(
                "Anomaly Alert Count",
                value=10,
                step=1
            )
        
        if st.button("Update Thresholds"):
            features.alert_thresholds['high_spending'] = high_spending
            features.alert_thresholds['fraud_score'] = fraud_threshold
            features.alert_thresholds['anomaly_count'] = anomaly_threshold
            st.success("✅ Thresholds updated!")
    
    with tab2:
        st.subheader("Key Performance Indicators")
        
        kpis = features.calculate_kpis(data)
        
        if kpis:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Spending",
                    f"₹{kpis.get('total_spending', 0):,.0f}"
                )
            
            with col2:
                st.metric(
                    "MoM Growth",
                    f"{kpis.get('mom_growth', 0):+.1f}%"
                )
            
            with col3:
                st.metric(
                    "Fraud Rate",
                    f"{kpis.get('fraud_rate', 0):.2f}%"
                )
            
            with col4:
                st.metric(
                    "Savings Potential",
                    f"₹{kpis.get('savings_potential', 0):,.0f}"
                )
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Avg Transaction",
                    f"₹{kpis.get('avg_transaction', 0):,.2f}"
                )
            
            with col2:
                st.metric(
                    "Category Concentration",
                    f"{kpis.get('category_concentration', 0):.1f}%"
                )
        else:
            st.info("No KPI data available")
    
    with tab3:
        st.subheader("Data Export")
        
        export_format = st.radio(
            "Select Export Format",
            ["CSV", "Excel", "Text Report"]
        )
        
        if st.button("Generate Export"):
            if export_format == "CSV" and 'transactions' in data:
                csv_data = features.export_to_csv(data['transactions'], 'transactions.csv')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"opam_transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            
            elif export_format == "Excel":
                excel_dict = {}
                if 'transactions' in data:
                    excel_dict['Transactions'] = data['transactions']
                if 'budget_recs' in data:
                    excel_dict['Budget'] = data['budget_recs']
                if 'fraud_scores' in data:
                    excel_dict['Fraud Scores'] = data['fraud_scores']
                
                if excel_dict:
                    excel_data = features.export_to_excel(excel_dict, 'opam_export.xlsx')
                    st.download_button(
                        label="📥 Download Excel",
                        data=excel_data,
                        file_name=f"opam_export_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                else:
                    st.warning("No data available for export")
            
            elif export_format == "Text Report":
                report = features.generate_summary_report(data)
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name=f"opam_report_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )
                
                st.markdown("---")
                st.subheader("Report Preview")
                st.text(report)
    
    with tab4:
        st.subheader("Spending Health Score")
        
        score, summary = features.spending_health_score(data)
        
        # Display score with color
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Create gauge-like visualization
            if score >= 90:
                color = "green"
            elif score >= 75:
                color = "blue"
            elif score >= 60:
                color = "orange"
            else:
                color = "red"
            
            st.markdown(f"""
                <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                    <h1 style='color: {color}; font-size: 72px; margin: 0;'>{score}</h1>
                    <p style='font-size: 20px; margin: 0;'>Health Score</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.metric("Status", summary)
        
        with col2:
            st.markdown("### Score Breakdown")
            
            # Show progress bars for different factors
            factors = {
                "Budget Adherence": 25,
                "Fraud Risk": 20,
                "Anomaly Rate": 20,
                "Spending Stability": 15,
                "Category Balance": 10,
                "Growth Pattern": 10
            }
            
            for factor, max_score in factors.items():
                # Calculate actual score (simplified)
                actual = max_score if score >= 90 else max_score * (score / 100)
                st.progress(actual / max_score, text=f"{factor}: {actual:.0f}/{max_score}")
    
    with tab5:
        st.subheader("Period Comparison")
        
        if 'transactions' in data:
            df = data['transactions']
            df['year_month'] = df['date'].dt.to_period('M').astype(str)
            available_periods = sorted(df['year_month'].unique())
            
            if len(available_periods) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    period1 = st.selectbox("Select Period 1", available_periods, index=len(available_periods)-2)
                
                with col2:
                    period2 = st.selectbox("Select Period 2", available_periods, index=len(available_periods)-1)
                
                if st.button("Compare Periods"):
                    features.show_comparison(data, period1, period2)
            else:
                st.info("Need at least 2 months of data for comparison")
        else:
            st.warning("No transaction data available")
