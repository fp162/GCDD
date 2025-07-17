import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Fleet Derate Analysis Dashboard",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .unit-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #4caf50;
    }
    .alert-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #f44336;
    }
    .warning-card {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)

def map_equipment_to_units(equipment_id):
    """Map equipment_id to UNIT and CAR based on your fleet structure"""
    # Unit 180108 cars
    unit_180108 = {
        50908: {'UNIT': '180108', 'CAR': 1, 'CAR_ID': '50908'},
        54908: {'UNIT': '180108', 'CAR': 2, 'CAR_ID': '54908'},
        55908: {'UNIT': '180108', 'CAR': 3, 'CAR_ID': '55908'},
        56908: {'UNIT': '180108', 'CAR': 4, 'CAR_ID': '56908'},
        59908: {'UNIT': '180108', 'CAR': 5, 'CAR_ID': '59908'}
    }
    
    # Unit 180112 cars
    unit_180112 = {
        50912: {'UNIT': '180112', 'CAR': 1, 'CAR_ID': '50912'},
        54912: {'UNIT': '180112', 'CAR': 2, 'CAR_ID': '54912'},
        55912: {'UNIT': '180112', 'CAR': 3, 'CAR_ID': '55912'},
        56912: {'UNIT': '180112', 'CAR': 4, 'CAR_ID': '56912'},
        59912: {'UNIT': '180112', 'CAR': 5, 'CAR_ID': '59912'}
    }
    
    # Combine mappings
    equipment_mapping = {**unit_180108, **unit_180112}
    
    return equipment_mapping.get(equipment_id, {'UNIT': 'Unknown', 'CAR': 0, 'CAR_ID': str(equipment_id)})

def load_sample_derate_data():
    """Generate sample derate data matching your fleet structure"""
    np.random.seed(42)
    
    # Equipment IDs from your fleet
    equipment_ids = [50908, 54908, 55908, 56908, 59908,  # Unit 180108
                    50912, 54912, 55912, 56912, 59912]   # Unit 180112
    
    # Create hourly buckets for a week
    dates = pd.date_range('2025-07-07', '2025-07-13', freq='H')
    
    data = []
    for bucket in dates:
        for equipment_id in equipment_ids:
            # Generate realistic derate values (some cars perform worse)
            if equipment_id in [55908, 56912]:  # Make some cars have higher derate
                derate_gap = np.random.uniform(5, 35)
            elif equipment_id in [54908, 59912]:  # Some moderate issues
                derate_gap = np.random.uniform(2, 25)
            else:  # Generally good performance
                derate_gap = np.random.uniform(0, 15)
            
            data.append({
                'bucket': bucket,
                'equipment_id': equipment_id,
                'derate_gap': derate_gap
            })
    
    return pd.DataFrame(data)

def process_derate_data(df):
    """Process your derate data format"""
    try:
        # Convert bucket to datetime
        df['bucket'] = pd.to_datetime(df['bucket'])
        
        # Extract date and hour
        df['Date'] = df['bucket'].dt.date
        df['Hour'] = df['bucket'].dt.hour
        df['Day_of_Week'] = df['bucket'].dt.day_name()
        
        # Convert equipment_id to numeric
        df['equipment_id'] = pd.to_numeric(df['equipment_id'], errors='coerce')
        
        # Convert derate_gap to numeric
        df['derate_gap'] = pd.to_numeric(df['derate_gap'], errors='coerce')
        
        # Map equipment_id to UNIT and CAR
        unit_car_mapping = df['equipment_id'].apply(map_equipment_to_units)
        df['UNIT'] = [mapping['UNIT'] for mapping in unit_car_mapping]
        df['CAR'] = [mapping['CAR'] for mapping in unit_car_mapping]
        df['CAR_ID'] = [mapping['CAR_ID'] for mapping in unit_car_mapping]
        
        # Calculate additional metrics
        df['Power_Available'] = 100 - df['derate_gap']
        df['Status'] = df['derate_gap'].apply(
            lambda x: 'Normal' if x < 10 else 'Warning' if x < 20 else 'Critical'
        )
        
        # Create labels for visualization
        df['Car_Label'] = df['UNIT'] + '-C' + df['CAR'].astype(str)
        df['Full_Label'] = df['UNIT'] + '-C' + df['CAR'].astype(str) + ' (' + df['CAR_ID'] + ')'
        
        return df
    except Exception as e:
        st.error(f"Error processing derate data: {str(e)}")
        return None

def create_unit_heatmap(df, unit_id):
    """Create heatmap for a specific unit"""
    unit_data = df[df['UNIT'] == unit_id].copy()
    
    if unit_data.empty:
        return None
    
    # Create heatmap data: Date x Hour with average derate across all cars
    heatmap_data = unit_data.groupby(['Date', 'Hour'])['derate_gap'].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='Date', columns='Hour', values='derate_gap')
    heatmap_pivot = heatmap_pivot.fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=[f"{h:02d}:00" for h in heatmap_pivot.columns],
        y=[str(date) for date in heatmap_pivot.index],
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Avg Derate: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Avg Derate %")
    ))
    
    fig.update_layout(
        title=f'Unit {unit_id} - Average Derate by Hour',
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        height=500
    )
    
    return fig

def create_car_heatmap(df, unit_id):
    """Create heatmap showing individual car performance for a unit"""
    unit_data = df[df['UNIT'] == unit_id].copy()
    
    if unit_data.empty:
        return None
    
    # Create heatmap: Car x Hour with average derate
    car_heatmap = unit_data.groupby(['Car_Label', 'Hour'])['derate_gap'].mean().reset_index()
    car_pivot = car_heatmap.pivot(index='Car_Label', columns='Hour', values='derate_gap')
    car_pivot = car_pivot.fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=car_pivot.values,
        x=[f"{h:02d}:00" for h in car_pivot.columns],
        y=car_pivot.index,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Derate: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Derate %")
    ))
    
    fig.update_layout(
        title=f'Unit {unit_id} - Car Performance by Hour',
        xaxis_title="Hour of Day",
        yaxis_title="Car",
        height=400
    )
    
    return fig

def create_fleet_overview_heatmap(df):
    """Create overview heatmap for entire fleet"""
    # Fleet-wide heatmap: All cars x hours
    fleet_heatmap = df.groupby(['Full_Label', 'Hour'])['derate_gap'].mean().reset_index()
    fleet_pivot = fleet_heatmap.pivot(index='Full_Label', columns='Hour', values='derate_gap')
    fleet_pivot = fleet_pivot.fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=fleet_pivot.values,
        x=[f"{h:02d}:00" for h in fleet_pivot.columns],
        y=fleet_pivot.index,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Derate: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Derate %")
    ))
    
    fig.update_layout(
        title='Fleet Overview - All Cars Performance by Hour',
        xaxis_title="Hour of Day",
        yaxis_title="Car (Unit-CarNum-ID)",
        height=600
    )
    
    return fig

def create_fleet_by_units_heatmap(df):
    """Create heatmap showing fleet performance grouped by units"""
    # Create unit-based aggregated heatmap
    unit_heatmap = df.groupby(['UNIT', 'Hour'])['derate_gap'].mean().reset_index()
    unit_pivot = unit_heatmap.pivot(index='UNIT', columns='Hour', values='derate_gap')
    unit_pivot = unit_pivot.fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=unit_pivot.values,
        x=[f"{h:02d}:00" for h in unit_pivot.columns],
        y=unit_pivot.index,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='<b>Unit %{y}</b><br>Hour: %{x}<br>Avg Derate: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Avg Derate %")
    ))
    
    fig.update_layout(
        title='Fleet Performance by Units - Hourly Average',
        xaxis_title="Hour of Day",
        yaxis_title="Unit",
        height=300
    )
    
    return fig

def create_problem_identification_chart(df):
    """Create chart to identify problematic cars"""
    # Calculate metrics for each car
    car_metrics = df.groupby(['UNIT', 'Car_Label', 'CAR_ID']).agg({
        'derate_gap': ['mean', 'max', 'std', 'count'],
        'Status': lambda x: (x == 'Critical').sum()
    }).reset_index()
    
    # Flatten columns
    car_metrics.columns = ['UNIT', 'Car_Label', 'CAR_ID', 'Avg_Derate', 'Max_Derate', 'Std_Derate', 'Records', 'Critical_Count']
    
    # Calculate problem score (weighted combination of metrics)
    car_metrics['Problem_Score'] = (
        car_metrics['Avg_Derate'] * 0.4 +
        car_metrics['Max_Derate'] * 0.3 +
        car_metrics['Std_Derate'] * 0.2 +
        (car_metrics['Critical_Count'] / car_metrics['Records'] * 100) * 0.1
    )
    
    # Sort by problem score
    car_metrics = car_metrics.sort_values('Problem_Score', ascending=False)
    
    # Create bar chart
    fig = go.Figure()
    
    # Color bars by unit
    colors = {'180108': '#1f77b4', '180112': '#ff7f0e'}
    
    for unit in car_metrics['UNIT'].unique():
        unit_data = car_metrics[car_metrics['UNIT'] == unit]
        fig.add_trace(go.Bar(
            x=unit_data['Car_Label'],
            y=unit_data['Problem_Score'],
            name=f'Unit {unit}',
            marker_color=colors.get(unit, '#7f7f7f'),
            text=unit_data['Problem_Score'].round(1),
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Problem Score: %{y:.1f}<br>' +
                         'Avg Derate: %{customdata[0]:.1f}%<br>' +
                         'Max Derate: %{customdata[1]:.1f}%<br>' +
                         '<extra></extra>',
            customdata=unit_data[['Avg_Derate', 'Max_Derate']].values
        ))
    
    fig.update_layout(
        title='Problem Car Identification - Higher Score = More Issues',
        xaxis_title="Car",
        yaxis_title="Problem Score",
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig, car_metrics

def create_performance_summary_table(df):
    """Create performance summary table"""
    summary = df.groupby(['UNIT', 'Car_Label', 'CAR_ID']).agg({
        'derate_gap': ['mean', 'max', 'min', 'std'],
        'Status': lambda x: pd.Series({
            'Normal': (x == 'Normal').sum(),
            'Warning': (x == 'Warning').sum(),
            'Critical': (x == 'Critical').sum()
        })
    }).reset_index()
    
    # Flatten columns
    summary.columns = ['UNIT', 'Car_Label', 'CAR_ID', 'Avg_Derate', 'Max_Derate', 'Min_Derate', 'Std_Derate', 'Normal_Count', 'Warning_Count', 'Critical_Count']
    
    # Round numerical values
    numerical_cols = ['Avg_Derate', 'Max_Derate', 'Min_Derate', 'Std_Derate']
    for col in numerical_cols:
        summary[col] = summary[col].round(2)
    
    return summary

def main():
    # Header
    st.markdown('<div class="main-header">🚂 Fleet Derate Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Fleet Engineer Dashboard** - Identify problematic cars and analyze derate patterns across your fleet")
    
    # Sidebar
    st.sidebar.markdown("## 🔧 Fleet Analysis Options")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload Fleet Data", "Use Sample Data"]
    )
    
    df = None
    
    if data_source == "Upload Fleet Data":
        st.sidebar.markdown("### 📁 Upload Derate Data")
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV/Excel with derate data",
            type=['csv', 'xlsx', 'xls'],
            help="Should include columns: bucket, equipment_id, derate_gap"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                df = process_derate_data(df)
                
                if df is not None:
                    st.sidebar.success("✅ Data loaded successfully!")
                    st.sidebar.markdown(f"**Records:** {len(df)}")
                    st.sidebar.markdown(f"**Cars:** {df['equipment_id'].nunique()}")
                    st.sidebar.markdown(f"**Date Range:** {df['Date'].min()} to {df['Date'].max()}")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("👆 Please upload your fleet derate data to get started!")
            
    else:
        df = load_sample_derate_data()
        df = process_derate_data(df)
        st.sidebar.markdown("### 🎯 Sample Fleet Data")
        st.sidebar.info("Using sample data for Units 180108 and 180112")
    
    if df is not None:
        # Fleet overview
        st.markdown('<div class="sub-header">🚂 Fleet Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Units", df['UNIT'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Cars", df['equipment_id'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_derate = df['derate_gap'].mean()
            st.metric("Fleet Avg Derate", f"{avg_derate:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Fleet structure display
        st.markdown('<div class="unit-info">', unsafe_allow_html=True)
        st.markdown("**Fleet Structure:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Unit 180108:**")
            unit_180108_cars = df[df['UNIT'] == '180108']['CAR_ID'].unique()
            for car_id in sorted(unit_180108_cars):
                car_num = df[df['CAR_ID'] == car_id]['CAR'].iloc[0]
                avg_derate = df[df['CAR_ID'] == car_id]['derate_gap'].mean()
                st.markdown(f"• Car {car_num}: {car_id} (Avg: {avg_derate:.1f}%)")
        
        with col2:
            st.markdown("**Unit 180112:**")
            unit_180112_cars = df[df['UNIT'] == '180112']['CAR_ID'].unique()
            for car_id in sorted(unit_180112_cars):
                car_num = df[df['CAR_ID'] == car_id]['CAR'].iloc[0]
                avg_derate = df[df['CAR_ID'] == car_id]['derate_gap'].mean()
                st.markdown(f"• Car {car_num}: {car_id} (Avg: {avg_derate:.1f}%)")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Problem identification
        st.markdown('<div class="sub-header">🚨 Problem Car Identification</div>', unsafe_allow_html=True)
        
        problem_fig, problem_data = create_problem_identification_chart(df)
        st.plotly_chart(problem_fig, use_container_width=True)
        
        # Show top 3 problematic cars
        top_problems = problem_data.head(3)
        if len(top_problems) > 0:
            st.markdown('<div class="alert-card">', unsafe_allow_html=True)
            st.markdown("**⚠️ Top 3 Problematic Cars:**")
            for _, row in top_problems.iterrows():
                st.markdown(f"• **{row['Car_Label']} ({row['CAR_ID']})**: Problem Score {row['Problem_Score']:.1f} - Avg Derate: {row['Avg_Derate']:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📊 By Unit", "🔍 All Cars", "📈 Fleet Summary"])
        
        with tab1:
            st.markdown("### Unit-Specific Analysis")
            
            # Unit selection
            selected_unit = st.selectbox("Select Unit:", df['UNIT'].unique())
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Unit daily heatmap
                unit_daily_fig = create_unit_heatmap(df, selected_unit)
                if unit_daily_fig:
                    st.plotly_chart(unit_daily_fig, use_container_width=True)
            
            with col2:
                # Unit car performance heatmap
                unit_car_fig = create_car_heatmap(df, selected_unit)
                if unit_car_fig:
                    st.plotly_chart(unit_car_fig, use_container_width=True)
            
            # Unit statistics
            unit_stats = df[df['UNIT'] == selected_unit].groupby('Car_Label').agg({
                'derate_gap': ['mean', 'max', 'std'],
                'Status': lambda x: (x == 'Critical').sum()
            }).round(2)
            unit_stats.columns = ['Avg Derate', 'Max Derate', 'Std Dev', 'Critical Count']
            
            st.markdown(f"### Unit {selected_unit} - Car Performance Summary")
            st.dataframe(unit_stats, use_container_width=True)
        
        with tab2:
            st.markdown("### All Cars Analysis")
            
            # Fleet overview heatmap
            fleet_fig = create_fleet_overview_heatmap(df)
            if fleet_fig:
                st.plotly_chart(fleet_fig, use_container_width=True)
            
            # Performance summary table
            st.markdown("### Detailed Performance Summary")
            summary_table = create_performance_summary_table(df)
            st.dataframe(summary_table, use_container_width=True)
        
        with tab3:
            st.markdown("### Fleet Summary by Units")
            
            # Fleet by units heatmap
            fleet_units_fig = create_fleet_by_units_heatmap(df)
            if fleet_units_fig:
                st.plotly_chart(fleet_units_fig, use_container_width=True)
            
            # Unit comparison metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Unit Performance Comparison**")
                unit_comparison = df.groupby('UNIT').agg({
                    'derate_gap': ['mean', 'max', 'std'],
                    'Status': lambda x: (x == 'Critical').sum()
                }).round(2)
                unit_comparison.columns = ['Avg Derate', 'Max Derate', 'Std Dev', 'Critical Events']
                st.dataframe(unit_comparison, use_container_width=True)
            
            with col2:
                st.markdown("**Daily Trends**")
                daily_trends = df.groupby(['Date', 'UNIT'])['derate_gap'].mean().reset_index()
                
                fig_trends = go.Figure()
                for unit in df['UNIT'].unique():
                    unit_data = daily_trends[daily_trends['UNIT'] == unit]
                    fig_trends.add_trace(go.Scatter(
                        x=unit_data['Date'],
                        y=unit_data['derate_gap'],
                        mode='lines+markers',
                        name=f'Unit {unit}',
                        line=dict(width=3)
                    ))
                
                fig_trends.update_layout(
                    title='Daily Average Derate Trends',
                    xaxis_title='Date',
                    yaxis_title='Average Derate %',
                    height=400
                )
                st.plotly_chart(fig_trends, use_container_width=True)
        
        # Download section
        st.markdown('<div class="sub-header">💾 Download Reports</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Fleet Analysis"):
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Complete Data",
                    data=csv_buffer.getvalue(),
                    file_name="fleet_derate_analysis.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🚨 Download Problem Cars Report"):
                problem_buffer = io.StringIO()
                problem_data.to_csv(problem_buffer, index=False)
                st.download_button(
                    label="Download Problem Cars",
                    data=problem_buffer.getvalue(),
                    file_name="problem_cars_report.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📈 Download Summary Statistics"):
                summary_buffer = io.StringIO()
                summary_table.to_csv(summary_buffer, index=False)
                st.download_button(
                    label="Download Summary Stats",
                    data=summary_buffer.getvalue(),
                    file_name="fleet_summary_stats.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
