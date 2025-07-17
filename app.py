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
        operational_hours = (car_data['derate_gap'] >= 0).sum()  # ALL hours including 0% derate
        warning_hours = (car_data['derate_gap'] > 10).sum()  # Renamed from problem_hours
        high_derate_hours = (car_data['derate_gap'] > 20).sum()  # Renamed from critical_hours
        
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
            'Warning_Hours': warning_hours,  # Renamed from Problem_Hours
            'High_Derate_Hours': high_derate_hours,  # Renamed from Critical_Hours
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
        
        # For daily analysis: 5 AM to midnight (19 hours)
        unit_data_filtered = unit_data[unit_data['Hour'] >= 5]
        
        # Create heatmap data using actual car IDs
        heatmap_data = unit_data_filtered.groupby(['CAR_ID', 'Hour'])['derate_gap'].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index='CAR_ID', columns='Hour', values='derate_gap')
        heatmap_pivot = heatmap_pivot.fillna(0)
        
        # Create hour labels from 5 AM to midnight
        available_hours = sorted(heatmap_pivot.columns)
        hour_labels = [f"{h:02d}:00" for h in available_hours]
        
        # Create individual traces for each car (for legend functionality)
        fig = go.Figure()
        
        car_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # Different colors for each car
        
        for i, car_id in enumerate(sorted(heatmap_pivot.index)):
            car_data = heatmap_pivot.loc[car_id].values
            
            # Apply color mapping based on derate values
            colors = []
            for val in car_data:
                if val == 0:
                    colors.append('#D3D3D3')  # Light Grey
                elif val <= 10:
                    colors.append('#FFFF00')  # Yellow
                elif val <= 20:
                    colors.append('#FFA500')  # Amber
                elif val <= 35:
                    colors.append('#FF8C00')  # Dark Amber
                else:
                    colors.append('#8B0000')  # Dark Red
            
            fig.add_trace(go.Bar(
                x=hour_labels,
                y=[1] * len(hour_labels),  # Same height for all
                name=f"Car {car_id}",
                marker_color=colors,
                yaxis=f'y{i+1}',
                hovertemplate=f'<b>Car {car_id}</b><br>Hour: %{{x}}<br>Derate: %{{customdata:.1f}}%<extra></extra>',
                customdata=car_data,
                visible=True
            ))
        
        # Update layout with multiple y-axes for each car
        layout_updates = {
            'title': f'Unit {unit_id} - Car Performance{title_suffix}',
            'xaxis_title': "Hour of Day (5 AM to Midnight)",
            'height': 500,
            'font': dict(size=12),
            'showlegend': True,
            'legend': dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        }
        
        # Create subplots for each car
        for i, car_id in enumerate(sorted(heatmap_pivot.index)):
            layout_updates[f'yaxis{i+1}'] = dict(
                title=f"Car {car_id}",
                domain=[i/len(heatmap_pivot.index), (i+1)/len(heatmap_pivot.index)],
                showticklabels=False
            )
        
        fig.update_layout(**layout_updates)
        
    else:
        # Weekly analysis: Create 7 separate heatmaps
        title_suffix = " - Weekly Overview"
        
        # Get unique dates
        dates = sorted(unit_data['Date'].unique())
        
        # Create subplot with 7 days
        fig = make_subplots(
            rows=7, cols=1,
            subplot_titles=[f"{date} ({pd.to_datetime(date).strftime('%A')})" for date in dates],
            vertical_spacing=0.02,
            specs=[[{"secondary_y": False}] for _ in range(7)]
        )
        
        # Color scale for values
        def get_color(val):
            if val == 0:
                return '#D3D3D3'  # Light Grey
            elif val <= 10:
                return '#FFFF00'  # Yellow
            elif val <= 20:
                return '#FFA500'  # Amber
            elif val <= 35:
                return '#FF8C00'  # Dark Amber
            else:
                return '#8B0000'  # Dark Red
        
        # Add heatmap for each day
        for day_idx, date in enumerate(dates):
            day_data = unit_data[unit_data['Date'] == date]
            day_filtered = day_data[day_data['Hour'] >= 5]  # 5 AM to midnight
            
            # Create heatmap data for this day
            day_heatmap = day_filtered.groupby(['CAR_ID', 'Hour'])['derate_gap'].mean().reset_index()
            day_pivot = day_heatmap.pivot(index='CAR_ID', columns='Hour', values='derate_gap')
            day_pivot = day_pivot.fillna(0)
            
            if not day_pivot.empty:
                available_hours = sorted(day_pivot.columns)
                hour_labels = [f"{h:02d}:00" for h in available_hours]
                
                # Add each car as a separate trace for this day
                for car_idx, car_id in enumerate(sorted(day_pivot.index)):
                    car_data = day_pivot.loc[car_id].values
                    colors = [get_color(val) for val in car_data]
                    
                    fig.add_trace(
                        go.Bar(
                            x=hour_labels,
                            y=[1] * len(hour_labels),
                            name=f"Car {car_id}" if day_idx == 0 else f"Car {car_id}",
                            marker_color=colors,
                            hovertemplate=f'<b>Car {car_id}</b><br>Hour: %{{x}}<br>Derate: %{{customdata:.1f}}%<extra></extra>',
                            customdata=car_data,
                            showlegend=(day_idx == 0),  # Only show legend for first day
                            legendgroup=f"car_{car_id}",
                            visible=True
                        ),
                        row=day_idx+1, col=1
                    )
        
        fig.update_layout(
            title=f'Unit {unit_id} - Car Performance{title_suffix}',
            height=1200,  # Taller for 7 days
            font=dict(size=10),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update x-axes for all subplots
        for i in range(7):
            fig.update_xaxes(title_text="Hour of Day (5 AM to Midnight)" if i == 6 else "", row=i+1, col=1)
            fig.update_yaxes(showticklabels=False, row=i+1, col=1)
    
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
        operational_hours = (unit_data['derate_gap'] >= 0).sum()  # Fixed: includes 0% derate
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
            
            # Fleet overview with daily tabs (moved to first position)
            st.markdown("### Fleet Overview - Daily Performance")
            
            # Get available dates
            available_dates = sorted(df['Date'].unique())
            
            # Create tabs for each date
            date_tabs = st.tabs([date.strftime('%d-%m') for date in available_dates])
            
            for i, date in enumerate(available_dates):
                with date_tabs[i]:
                    # Filter data for this specific date and 5 AM to midnight
                    day_data = df[(df['Date'] == date) & (df['Hour'] >= 5)]
                    
                    if not day_data.empty:
                        # Create ordered car list: Unit 180108 first, then Unit 180112
                        unit_180108_cars = [50908, 54908, 55908, 56908, 59908]
                        unit_180112_cars = [50912, 54912, 55912, 56912, 59912]
                        ordered_cars = unit_180108_cars + unit_180112_cars
                        
                        # Create heatmap data using integer car IDs
                        fleet_heatmap_data = day_data.groupby(['equipment_id', 'Hour'])['derate_gap'].mean().reset_index()
                        fleet_pivot = fleet_heatmap_data.pivot(index='equipment_id', columns='Hour', values='derate_gap')
                        fleet_pivot = fleet_pivot.fillna(0)
                        
                        # Reorder rows according to our specified order
                        available_cars = [car for car in ordered_cars if car in fleet_pivot.index]
                        fleet_pivot = fleet_pivot.reindex(available_cars)
                        
                        # Create hour labels from 5 AM to midnight
                        available_hours = sorted(fleet_pivot.columns)
                        hour_labels = [f"{h:02d}:00" for h in available_hours]
                        
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
                            x=hour_labels,
                            y=[str(int(car)) for car in fleet_pivot.index],  # Convert to int to remove commas
                            colorscale=colorscale,
                            hoverongaps=False,
                            hovertemplate='<b>Car %{y}</b><br>Hour: %{x}<br>Derate: %{z:.1f}%<extra></extra>',
                            colorbar=dict(title="Derate %"),
                            zmin=0,
                            zmax=40
                        ))
                        
                        fleet_fig.update_layout(
                            title=f'Fleet Performance - {date.strftime("%d-%m")}',
                            xaxis_title="Hour of Day (5 AM to Midnight)",
                            yaxis_title="Car ID",
                            height=600,
                            yaxis=dict(
                                tickmode='array',
                                tickvals=list(range(len(fleet_pivot.index))),
                                ticktext=[str(int(car)) for car in fleet_pivot.index],
                                type='category'
                            )
                        )
                        
                        st.plotly_chart(fleet_fig, use_container_width=True)
                    else:
                        st.info(f"No data available for {date.strftime('%d-%m')}")
            
            # Problem car ranking
            st.markdown("### Problem Car Ranking")
            fleet_metrics = calculate_hybrid_metrics(df)
            if not fleet_metrics.empty:
                # Calculate problem score
                fleet_metrics['Problem_Score'] = (
                    fleet_metrics['Active_Avg_Derate'] * 0.4 +
                    fleet_metrics['Max_Derate'] * 0.3 +
                    (fleet_metrics['Warning_Hours'] / fleet_metrics['Total_Hours'] * 100) * 0.3
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
                        st.markdown(f"• **Car {int(row['CAR_ID'])} (Unit {row['UNIT']})**: Problem Score {row['Problem_Score']:.1f}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Weekly Volume Bar Chart
            st.markdown("### Weekly Derate Volume Analysis")
            
            # Filter data for 5 AM to midnight across all days
            volume_data = df[df['Hour'] >= 5].copy()
            
            if not volume_data.empty:
                # Create ordered car list
                unit_180108_cars = [50908, 54908, 55908, 56908, 59908]
                unit_180112_cars = [50912, 54912, 55912, 56912, 59912]
                ordered_cars = unit_180108_cars + unit_180112_cars
                
                # Create time labels (Day-Hour combinations)
                dates = sorted(volume_data['Date'].unique())
                hours = list(range(5, 24))  # 5 AM to 11 PM
                
                time_labels = []
                for date in dates:
                    for hour in hours:
                        time_labels.append(f"{date.strftime('%d-%m')}_{hour:02d}:00")
                
                # Prepare data for 3D bar chart
                x_data = []  # Time indices
                y_data = []  # Car indices
                z_data = []  # Derate values
                colors = []  # Bar colors
                
                car_positions = {car: i for i, car in enumerate(ordered_cars)}
                
                for time_idx, (date, hour) in enumerate([(d, h) for d in dates for h in hours]):
                    for car in ordered_cars:
                        if car in volume_data['equipment_id'].values:
                            # Get derate value for this car at this time
                            car_data = volume_data[
                                (volume_data['Date'] == date) & 
                                (volume_data['Hour'] == hour) & 
                                (volume_data['equipment_id'] == car)
                            ]
                            
                            if not car_data.empty:
                                derate_val = car_data['derate_gap'].iloc[0]
                            else:
                                derate_val = 0
                            
                            x_data.append(time_idx)
                            y_data.append(car_positions[car])
                            z_data.append(derate_val)
                            
                            # Assign color based on derate value
                            if derate_val == 0:
                                colors.append('#D3D3D3')  # Light Grey
                            elif derate_val <= 10:
                                colors.append('#FFFF00')  # Yellow
                            elif derate_val <= 20:
                                colors.append('#FFA500')  # Amber
                            elif derate_val <= 35:
                                colors.append('#FF8C00')  # Dark Amber
                            else:
                                colors.append('#8B0000')  # Dark Red
                
                # Create 3D scatter plot with sized markers to simulate volume bars
                volume_fig = go.Figure(data=go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=z_data,
                    mode='markers',
                    marker=dict(
                        size=[max(3, z/2) for z in z_data],  # Size based on derate value
                        color=colors,
                        opacity=0.8,
                        symbol='square'
                    ),
                    hovertemplate='<b>Car: %{customdata[0]}</b><br>' +
                                  'Time: %{customdata[1]}<br>' +
                                  'Derate: %{z:.1f}%<br>' +
                                  '<extra></extra>',
                    customdata=[[ordered_cars[y_data[i]], time_labels[x_data[i]]] for i in range(len(x_data))]
                ))
                
                # Update layout for 3D chart
                volume_fig.update_layout(
                    title='Weekly Derate Volume - All Cars Across 7 Days',
                    scene=dict(
                        xaxis_title="Time (Day-Hour)",
                        yaxis_title="Car ID",
                        zaxis_title="Derate %",
                        xaxis=dict(
                            tickmode='array',
                            tickvals=list(range(0, len(time_labels), 19)),  # Show every day
                            ticktext=[dates[i].strftime('%d-%m') for i in range(len(dates))]
                        ),
                        yaxis=dict(
                            tickmode='array',
                            tickvals=list(range(len(ordered_cars))),
                            ticktext=[str(car) for car in ordered_cars]
                        ),
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.2)
                        )
                    ),
                    height=700,
                    font=dict(size=12)
                )
                
                st.plotly_chart(volume_fig, use_container_width=True)
            else:
                st.info("No data available for volume analysis")
        
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
