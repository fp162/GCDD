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
    .alert-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #f44336;
    }
    .info-card {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
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
            # Simulate realistic patterns: some cars not running (0%), some with issues
            if np.random.random() < 0.15:  # 15% chance not running
                derate_gap = 0.0
            elif equipment_id in [55908, 56912]:  # Some cars with more issues
                derate_gap = np.random.uniform(0, 40)
            elif equipment_id in [54908, 59912]:  # Some moderate issues
                derate_gap = np.random.uniform(0, 25)
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
        
        # Calculate operational status
        df['Is_Running'] = df['derate_gap'] > 0
        df['Is_Problem'] = df['derate_gap'] > 10
        df['Is_Critical'] = df['derate_gap'] > 20
        df['Is_Active_Derate'] = df['derate_gap'] > 5
        
        # Calculate additional metrics
        df['Power_Available'] = 100 - df['derate_gap']
        
        return df
    except Exception as e:
        st.error(f"Error processing derate data: {str(e)}")
        return None

def calculate_hybrid_metrics(df, unit_id=None, date=None):
    """Calculate hybrid metrics for analysis"""
    # Filter data if needed
    filtered_df = df.copy()
    if unit_id:
        filtered_df = filtered_df[filtered_df['UNIT'] == unit_id]
    if date:
        filtered_df = filtered_df[filtered_df['Date'] == date]
    
    # Calculate metrics by car
    metrics = []
    
    for car_info in filtered_df.groupby(['UNIT', 'CAR', 'CAR_ID']):
        (unit, car_num, car_id), car_data = car_info
        
        # Hybrid metrics
        total_hours = len(car_data)
        operational_hours = (car_data['derate_gap'] > 0).sum()
        problem_hours = (car_data['derate_gap'] > 10).sum()
        critical_hours = (car_data['derate_gap'] > 20).sum()
        
        # Active derate average (only when > 5%)
        active_derate_data = car_data[car_data['derate_gap'] > 5]
        active_avg = active_derate_data['derate_gap'].mean() if len(active_derate_data) > 0 else 0
        
        # Max derate
        max_derate = car_data['derate_gap'].max()
        
        # Operational ratio
        operational_ratio = (operational_hours / total_hours * 100) if total_hours > 0 else 0
        
        metrics.append({
            'UNIT': unit,
            'CAR': car_num,
            'CAR_ID': car_id,
            'Active_Avg_Derate': active_avg,
            'Max_Derate': max_derate,
            'Problem_Hours': problem_hours,
            'Critical_Hours': critical_hours,
            'Operational_Hours': operational_hours,
            'Operational_Ratio': operational_ratio,
            'Total_Hours': total_hours
        })
    
    return pd.DataFrame(metrics)

def create_unit_car_heatmap(df, unit_id, selected_date=None, period='daily'):
    """Create car heatmap with your color scheme"""
    unit_data = df[df['UNIT'] == unit_id].copy()
    
    if selected_date and period == 'daily':
        unit_data = unit_data[unit_data['Date'] == selected_date]
        title_suffix = f" - {selected_date}"
        
        # For daily analysis: adjust hours to start at 5 AM
        unit_data['Adjusted_Hour'] = unit_data['Hour'].apply(lambda x: x - 5 if x >= 5 else x + 19)
        
        # Create heatmap data using actual car IDs
        heatmap_data = unit_data.groupby(['CAR_ID', 'Adjusted_Hour'])['derate_gap'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='CAR_ID', columns='Adjusted_Hour', values='derate_gap')
        heatmap_pivot = heatmap_pivot.fillna(0)
        
        # Create hour labels starting from 5 AM
        hour_labels = []
        for h in range(24):
            actual_hour = (h + 5) % 24
            hour_labels.append(f"{actual_hour:02d}:00")
        
        x_labels = hour_labels
        y_labels = [str(car_id) for car_id in heatmap_pivot.index]
        
    else:
        title_suffix = " - Weekly Overview"
        
        # For weekly analysis: show cars × days
        # Create a combination of Date and CAR_ID for Y-axis
        unit_data['Date_Car'] = unit_data['Date'].astype(str) + ' - ' + unit_data['CAR_ID']
        
        # Adjust hours to start at 5 AM
        unit_data['Adjusted_Hour'] = unit_data['Hour'].apply(lambda x: x - 5 if x >= 5 else x + 19)
        
        # Create heatmap data
        heatmap_data = unit_data.groupby(['Date_Car', 'Adjusted_Hour'])['derate_gap'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='Date_Car', columns='Adjusted_Hour', values='derate_gap')
        heatmap_pivot = heatmap_pivot.fillna(0)
        
        # Create hour labels starting from 5 AM
        hour_labels = []
        for h in range(24):
            actual_hour = (h + 5) % 24
            hour_labels.append(f"{actual_hour:02d}:00")
        
        x_labels = hour_labels
        y_labels = heatmap_pivot.index
    
    # Define your color scale
    colorscale = [
        [0, '#D3D3D3'],      # Light Grey (0%)
        [0.05, '#D3D3D3'],   # Light Grey (0-5%)
        [0.1, '#FFFF00'],    # Yellow (5-10%)
        [0.2, '#FFA500'],    # Amber (10-20%)
        [0.35, '#FF8C00'],   # Dark Amber (20-35%)
        [1, '#8B0000']       # Dark Red (35%+)
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Derate: %{z:.1f}%<extra></extra>',
        colorbar=dict(title="Derate %"),
        zmin=0,
        zmax=40
    ))
    
    fig.update_layout(
        title=f'Unit {unit_id} - Car Performance{title_suffix}',
        xaxis_title="Hour of Day (Starting 5 AM)",
        yaxis_title="Car ID" if period == 'daily' else "Date - Car ID",
        height=400 if period == 'daily' else 800,  # Taller for weekly view
        font=dict(size=12)
    )
    
    return fig

def create_active_derate_chart(df, unit_id, period='daily'):
    """Create active derate average line chart"""
    unit_data = df[df['UNIT'] == unit_id].copy()
    
    if period == 'daily':
        # Group by date and calculate active derate average
        daily_data = []
        for date in sorted(unit_data['Date'].unique()):
            date_data = unit_data[unit_data['Date'] == date]
            active_data = date_data[date_data['derate_gap'] > 5]
            active_avg = active_data['derate_gap'].mean() if len(active_data) > 0 else 0
            daily_data.append({'Date': date, 'Active_Avg_Derate': active_avg})
        
        chart_data = pd.DataFrame(daily_data)
        x_col = 'Date'
        title = f'Unit {unit_id} - Daily Active Derate Average (>5%)'
    else:
        # Group by hour across the week
        hourly_data = unit_data[unit_data['derate_gap'] > 5].groupby('Hour')['derate_gap'].mean().reset_index()
        hourly_data.columns = ['Hour', 'Active_Avg_Derate']
        chart_data = hourly_data
        x_col = 'Hour'
        title = f'Unit {unit_id} - Hourly Active Derate Average (>5%)'
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data[x_col],
        y=chart_data['Active_Avg_Derate'],
        mode='lines+markers',
        name='Active Derate Avg',
        line=dict(width=3, color='#ff6b6b')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title="Active Derate Average (%)",
        height=300
    )
    
    return fig

def create_problem_hours_chart(df, unit_id):
    """Create problem hours identification chart"""
    unit_data = df[df['UNIT'] == unit_id].copy()
    
    # Count problem hours by hour of day
    problem_by_hour = unit_data[unit_data['derate_gap'] > 10].groupby('Hour').size().reset_index(name='Problem_Count')
    critical_by_hour = unit_data[unit_data['derate_gap'] > 20].groupby('Hour').size().reset_index(name='Critical_Count')
    
    # Merge data
    all_hours = pd.DataFrame({'Hour': range(24)})
    problem_data = all_hours.merge(problem_by_hour, on='Hour', how='left').fillna(0)
    problem_data = problem_data.merge(critical_by_hour, on='Hour', how='left').fillna(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=[f"{h:02d}:00" for h in problem_data['Hour']],
        y=problem_data['Problem_Count'],
        name='Problem Hours (>10%)',
        marker_color='#ffa500'
    ))
    
    fig.add_trace(go.Bar(
        x=[f"{h:02d}:00" for h in problem_data['Hour']],
        y=problem_data['Critical_Count'],
        name='Critical Hours (>20%)',
        marker_color='#8b0000'
    ))
    
    fig.update_layout(
        title=f'Unit {unit_id} - Problem Hours Distribution',
        xaxis_title='Hour of Day',
        yaxis_title='Number of Problem Events',
        height=300,
        barmode='group'
    )
    
    return fig

def create_car_performance_bars(df, unit_id):
    """Create car performance bar chart"""
    unit_data = df[df['UNIT'] == unit_id].copy()
    
    # Calculate metrics per car using actual car IDs
    car_metrics = []
    for car_id in sorted(unit_data['CAR_ID'].unique()):
        car_data = unit_data[unit_data['CAR_ID'] == car_id]
        
        # Active derate average
        active_data = car_data[car_data['derate_gap'] > 5]
        active_avg = active_data['derate_gap'].mean() if len(active_data) > 0 else 0
        
        # Max derate
        max_derate = car_data['derate_gap'].max()
        
        # Problem hours
        problem_hours = (car_data['derate_gap'] > 10).sum()
        
        car_metrics.append({
            'CAR_ID': car_id,
            'Active_Avg': active_avg,
            'Max_Derate': max_derate,
            'Problem_Hours': problem_hours
        })
    
    metrics_df = pd.DataFrame(car_metrics)
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Active Derate Average', 'Max Derate', 'Problem Hours'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Active average
    fig.add_trace(
        go.Bar(x=metrics_df['CAR_ID'], 
               y=metrics_df['Active_Avg'],
               name='Active Avg',
               marker_color='#ff6b6b'),
        row=1, col=1
    )
    
    # Max derate
    fig.add_trace(
        go.Bar(x=metrics_df['CAR_ID'], 
               y=metrics_df['Max_Derate'],
               name='Max Derate',
               marker_color='#8b0000'),
        row=1, col=2
    )
    
    # Problem hours
    fig.add_trace(
        go.Bar(x=metrics_df['CAR_ID'], 
               y=metrics_df['Problem_Hours'],
               name='Problem Hours',
               marker_color='#ffa500'),
        row=1, col=3
    )
    
    fig.update_layout(
        title=f'Unit {unit_id} - Car Performance Metrics',
        height=400,
        showlegend=False
    )
    
    return fig

def create_fleet_comparison(df):
    """Create fleet comparison charts"""
    # Calculate metrics for each unit
    unit_metrics = []
    for unit in df['UNIT'].unique():
        unit_data = df[df['UNIT'] == unit]
        
        # Active derate average
        active_data = unit_data[unit_data['derate_gap'] > 5]
        active_avg = active_data['derate_gap'].mean() if len(active_data) > 0 else 0
        
        # Other metrics
        max_derate = unit_data['derate_gap'].max()
        problem_hours = (unit_data['derate_gap'] > 10).sum()
        operational_hours = (unit_data['derate_gap'] > 0).sum()
        total_hours = len(unit_data)
        availability = (operational_hours / total_hours * 100) if total_hours > 0 else 0
        
        unit_metrics.append({
            'Unit': unit,
            'Active_Avg_Derate': active_avg,
            'Max_Derate': max_derate,
            'Problem_Hours': problem_hours,
            'Availability': availability
        })
    
    metrics_df = pd.DataFrame(unit_metrics)
    
    # Create comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Active Derate Average', 'Max Derate', 'Problem Hours', 'Availability %'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#1f77b4', '#ff7f0e']
    
    # Active average
    fig.add_trace(go.Bar(x=metrics_df['Unit'], y=metrics_df['Active_Avg_Derate'], 
                         name='Active Avg', marker_color=colors), row=1, col=1)
    
    # Max derate
    fig.add_trace(go.Bar(x=metrics_df['Unit'], y=metrics_df['Max_Derate'], 
                         name='Max Derate', marker_color=colors), row=1, col=2)
    
    # Problem hours
    fig.add_trace(go.Bar(x=metrics_df['Unit'], y=metrics_df['Problem_Hours'], 
                         name='Problem Hours', marker_color=colors), row=2, col=1)
    
    # Availability
    fig.add_trace(go.Bar(x=metrics_df['Unit'], y=metrics_df['Availability'], 
                         name='Availability', marker_color=colors), row=2, col=2)
    
    fig.update_layout(
        title='Fleet Comparison - Unit Performance',
        height=600,
        showlegend=False
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🚂 Fleet Derate Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Fleet Engineer Dashboard** - Hybrid metrics for real performance analysis")
    
    # Sidebar
    st.sidebar.markdown("## 🔧 Fleet Data")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source:",
        ["Upload Fleet Data", "Use Sample Data"]
    )
    
    df = None
    
    if data_source == "Upload Fleet Data":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV/Excel with derate data",
            type=['csv', 'xlsx', 'xls'],
            help="Columns: bucket, equipment_id, derate_gap"
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
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("👆 Please upload your fleet derate data to get started!")
            
    else:
        df = load_sample_derate_data()
        df = process_derate_data(df)
        st.sidebar.success("✅ Sample data loaded")
    
    if df is not None:
        # Collapsible data overview
        with st.expander("📊 Data Overview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", len(df))
            with col2:
                st.metric("Units", df['UNIT'].nunique())
            with col3:
                st.metric("Cars", df['equipment_id'].nunique())
            with col4:
                active_data = df[df['derate_gap'] > 5]
                fleet_active_avg = active_data['derate_gap'].mean() if len(active_data) > 0 else 0
                st.metric("Fleet Active Avg", f"{fleet_active_avg:.1f}%")
            
            # Fleet structure
            st.markdown("**Fleet Structure:**")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Unit 180108:** 50908, 54908, 55908, 56908, 59908")
            with col2:
                st.markdown("**Unit 180112:** 50912, 54912, 55912, 56912, 59912")
        
        # Main analysis tabs
        tab1, tab2 = st.tabs(["📊 UNIT Analysis", "🚁 Fleet Analysis"])
        
        with tab1:
            st.markdown("### Unit Performance Analysis")
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                selected_unit = st.selectbox("Select Unit:", df['UNIT'].unique())
            
            with col2:
                period = st.selectbox("Analysis Period:", ["Daily Analysis", "Weekly Analysis"])
            
            with col3:
                if period == "Daily Analysis":
                    available_dates = sorted(df['Date'].unique())
                    selected_date = st.selectbox("Select Date:", available_dates)
                else:
                    selected_date = None
            
            # Worst performing car alert
            unit_data = df[df['UNIT'] == selected_unit]
            if selected_date and period == "Daily Analysis":
                unit_data = unit_data[unit_data['Date'] == selected_date]
            
            # Calculate worst car using actual car IDs
            car_active_avg = []
            for car_id in unit_data['CAR_ID'].unique():
                car_data = unit_data[unit_data['CAR_ID'] == car_id]
                active_data = car_data[car_data['derate_gap'] > 5]
                active_avg = active_data['derate_gap'].mean() if len(active_data) > 0 else 0
                car_active_avg.append((car_id, active_avg))
            
            if car_active_avg:
                worst_car = max(car_active_avg, key=lambda x: x[1])
                if worst_car[1] > 15:  # Alert threshold
                    st.markdown('<div class="alert-card">', unsafe_allow_html=True)
                    st.markdown(f"🚨 **Maintenance Alert**: Car {worst_car[0]} has highest active derate average of {worst_car[1]:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Main heatmap
            heatmap_fig = create_unit_car_heatmap(df, selected_unit, selected_date, 
                                                 'daily' if period == "Daily Analysis" else 'weekly')
            if heatmap_fig:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            # Additional insights
            col1, col2 = st.columns(2)
            
            with col1:
                # Active derate trend
                active_chart = create_active_derate_chart(df, selected_unit, 
                                                        'daily' if period == "Daily Analysis" else 'weekly')
                if active_chart:
                    st.plotly_chart(active_chart, use_container_width=True)
            
            with col2:
                # Problem hours
                problem_chart = create_problem_hours_chart(df, selected_unit)
                if problem_chart:
                    st.plotly_chart(problem_chart, use_container_width=True)
            
            # Car performance metrics
            car_bars = create_car_performance_bars(df, selected_unit)
            if car_bars:
                st.plotly_chart(car_bars, use_container_width=True)
            
            # Unit metrics table
            st.markdown("### Unit Metrics Summary")
            unit_metrics = calculate_hybrid_metrics(df, selected_unit, selected_date)
            if not unit_metrics.empty:
                # Round numeric columns
                numeric_cols = ['Active_Avg_Derate', 'Max_Derate', 'Operational_Ratio']
                for col in numeric_cols:
                    if col in unit_metrics.columns:
                        unit_metrics[col] = unit_metrics[col].round(2)
                
                st.dataframe(unit_metrics, use_container_width=True)
        
        with tab2:
            st.markdown("### Fleet Analysis")
            
            # Fleet comparison
            fleet_comparison_fig = create_fleet_comparison(df)
            if fleet_comparison_fig:
                st.plotly_chart(fleet_comparison_fig, use_container_width=True)
            
            # Fleet overview heatmap (all cars)
            st.markdown("### Fleet Overview - All Cars")
            fleet_heatmap_data = df.groupby(['UNIT', 'CAR', 'Hour'])['derate_gap'].mean().reset_index()
            fleet_heatmap_data['Car_Label'] = fleet_heatmap_data['UNIT'] + '-C' + fleet_heatmap_data['CAR'].astype(str)
            
            fleet_pivot = fleet_heatmap_data.pivot(index='Car_Label', columns='Hour', values='derate_gap')
            fleet_pivot = fleet_pivot.fillna(0)
            
            # Same color scale
            colorscale = [
                [0, '#D3D3D3'],      # Light Grey
                [0.05, '#D3D3D3'],   # Light Grey
                [0.1, '#FFFF00'],    # Yellow
                [0.2, '#FFA500'],    # Amber
                [0.35, '#FF8C00'],   # Dark Amber
                [1, '#8B0000']       # Dark Red
            ]
            
            fleet_fig = go.Figure(data=go.Heatmap(
                z=fleet_pivot.values,
                x=[f"{h:02d}:00" for h in fleet_pivot.columns],
                y=fleet_pivot.index,
                colorscale=colorscale,
                hoverongaps=False,
                hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Derate: %{z:.1f}%<extra></extra>',
                colorbar=dict(title="Derate %"),
                zmin=0,
                zmax=40
            ))
            
            fleet_fig.update_layout(
                title='Fleet Overview - All Cars Performance',
                xaxis_title="Hour of Day",
                yaxis_title="Car (Unit-CarNum)",
                height=600
            )
            
            st.plotly_chart(fleet_fig, use_container_width=True)
            
            # Problem car ranking
            st.markdown("### Problem Car Ranking")
            fleet_metrics = calculate_hybrid_metrics(df)
            if not fleet_metrics.empty:
                # Calculate problem score
                fleet_metrics['Problem_Score'] = (
                    fleet_metrics['Active_Avg_Derate'] * 0.4 +
                    fleet_metrics['Max_Derate'] * 0.3 +
                    (fleet_metrics['Problem_Hours'] / fleet_metrics['Total_Hours'] * 100) * 0.3
                )
                
                # Sort by problem score
                fleet_metrics = fleet_metrics.sort_values('Problem_Score', ascending=False)
                
                # Round numeric columns
                numeric_cols = ['Active_Avg_Derate', 'Max_Derate', 'Operational_Ratio', 'Problem_Score']
                for col in numeric_cols:
                    if col in fleet_metrics.columns:
                        fleet_metrics[col] = fleet_metrics[col].round(2)
                
                st.dataframe(fleet_metrics, use_container_width=True)
                
                # Top 3 problem cars
                top_problems = fleet_metrics.head(3)
                if len(top_problems) > 0:
                    st.markdown('<div class="alert-card">', unsafe_allow_html=True)
                    st.markdown("**🚨 Top 3 Problem Cars:**")
                    for _, row in top_problems.iterrows():
                        st.markdown(f"• **Car {row['CAR']} (Unit {row['UNIT']})**: Problem Score {row['Problem_Score']:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Download section
        st.markdown("---")
        st.markdown("### 💾 Download Reports")
        
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
            if st.button("🚨 Download Problem Cars"):
                if 'fleet_metrics' in locals():
                    problem_buffer = io.StringIO()
                    fleet_metrics.to_csv(problem_buffer, index=False)
                    st.download_button(
                        label="Download Problem Cars Report",
                        data=problem_buffer.getvalue(),
                        file_name="problem_cars_report.csv",
                        mime="text/csv"
                    )
        
        with col3:
            if st.button("📈 Download Hybrid Metrics"):
                metrics_buffer = io.StringIO()
                hybrid_metrics = calculate_hybrid_metrics(df)
                hybrid_metrics.to_csv(metrics_buffer, index=False)
                st.download_button(
                    label="Download Hybrid Metrics",
                    data=metrics_buffer.getvalue(),
                    file_name="hybrid_metrics_report.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
