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
    page_title="Train Derate Analysis Dashboard",
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
    .train-info {
        background-color: #e1f5fe;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #0277bd;
    }
</style>
""", unsafe_allow_html=True)

def map_equipment_to_units(equipment_id):
    """Map equipment_id to UNIT and CAR based on your data structure"""
    # Your equipment IDs: 5090, 5091, 5490, 5491, 5590, 5591, 5690, 5691, 5990, 5991
    equipment_mapping = {
        5090: {'UNIT': 1, 'CAR': 1},
        5091: {'UNIT': 1, 'CAR': 2},
        5490: {'UNIT': 1, 'CAR': 3},
        5491: {'UNIT': 1, 'CAR': 4},
        5590: {'UNIT': 1, 'CAR': 5},
        5591: {'UNIT': 2, 'CAR': 1},
        5690: {'UNIT': 2, 'CAR': 2},
        5691: {'UNIT': 2, 'CAR': 3},
        5990: {'UNIT': 2, 'CAR': 4},
        5991: {'UNIT': 2, 'CAR': 5}
    }
    return equipment_mapping.get(equipment_id, {'UNIT': 0, 'CAR': 0})

def load_sample_derate_data():
    """Generate sample derate data matching your format"""
    np.random.seed(42)
    
    # Equipment IDs from your data
    equipment_ids = [5090, 5091, 5490, 5491, 5590, 5591, 5690, 5691, 5990, 5991]
    
    # Create hourly buckets for a week
    dates = pd.date_range('2025-07-07', '2025-07-13', freq='H')
    
    data = []
    for bucket in dates:
        for equipment_id in equipment_ids:
            # Generate realistic derate values
            derate_gap = np.random.uniform(0, 25)  # 0-25% derate
            
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
        
        # Convert equipment_id to numeric
        df['equipment_id'] = pd.to_numeric(df['equipment_id'], errors='coerce')
        
        # Convert derate_gap to numeric
        df['derate_gap'] = pd.to_numeric(df['derate_gap'], errors='coerce')
        
        # Map equipment_id to UNIT and CAR
        unit_car_mapping = df['equipment_id'].apply(map_equipment_to_units)
        df['UNIT'] = [mapping['UNIT'] for mapping in unit_car_mapping]
        df['CAR'] = [mapping['CAR'] for mapping in unit_car_mapping]
        
        # Calculate additional metrics
        df['Power_Available'] = 100 - df['derate_gap']
        df['Status'] = df['derate_gap'].apply(
            lambda x: 'Normal' if x < 10 else 'Degraded' if x < 20 else 'Critical'
        )
        
        # Create equipment label
        df['Equipment_Label'] = 'U' + df['UNIT'].astype(str) + 'C' + df['CAR'].astype(str) + ' (' + df['equipment_id'].astype(str) + ')'
        
        return df
    except Exception as e:
        st.error(f"Error processing derate data: {str(e)}")
        return None

def create_unit_daily_heatmap(df, unit_num):
    """Create heatmap for a specific unit showing daily hourly derate performance"""
    unit_data = df[df['UNIT'] == unit_num].copy()
    
    if unit_data.empty:
        return None
    
    # Aggregate by date and hour (average derate across all cars in the unit)
    hourly_data = unit_data.groupby(['Date', 'Hour'])['derate_gap'].mean().reset_index()
    
    # Pivot to create heatmap format
    heatmap_data = hourly_data.pivot(index='Date', columns='Hour', values='derate_gap')
    
    # Fill missing values with 0
    heatmap_data = heatmap_data.fillna(0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h:02d}:00" for h in heatmap_data.columns],
        y=[str(date) for date in heatmap_data.index],
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='<b>Date: %{y}</b><br>' +
                      '<b>Hour: %{x}</b><br>' +
                      'Avg Derate: %{z:.1f}%<br>' +
                      '<extra></extra>',
        colorbar=dict(title="Derate %")
    ))
    
    fig.update_layout(
        title=f'Unit {unit_num} - Average Derate % by Hour and Date',
        xaxis_title="Hour of Day",
        yaxis_title="Date",
        height=400,
        font=dict(size=12)
    )
    
    return fig

def create_equipment_heatmap(df, selected_date=None):
    """Create heatmap showing all equipment derate by hour"""
    if selected_date:
        df_filtered = df[df['Date'] == selected_date].copy()
        title_suffix = f" - {selected_date}"
    else:
        df_filtered = df.copy()
        title_suffix = " - All Dates Average"
    
    # Aggregate by equipment and hour
    if selected_date:
        equipment_hourly = df_filtered.groupby(['Equipment_Label', 'Hour'])['derate_gap'].mean().reset_index()
    else:
        equipment_hourly = df_filtered.groupby(['Equipment_Label', 'Hour'])['derate_gap'].mean().reset_index()
    
    # Pivot for heatmap
    heatmap_data = equipment_hourly.pivot(index='Equipment_Label', columns='Hour', values='derate_gap')
    heatmap_data = heatmap_data.fillna(0)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=[f"{h:02d}:00" for h in heatmap_data.columns],
        y=heatmap_data.index,
        colorscale='RdYlBu_r',
        hoverongaps=False,
        hovertemplate='<b>Equipment: %{y}</b><br>' +
                      '<b>Hour: %{x}</b><br>' +
                      'Derate: %{z:.1f}%<br>' +
                      '<extra></extra>',
        colorbar=dict(title="Derate %")
    ))
    
    fig.update_layout(
        title=f'All Equipment - Derate % by Hour{title_suffix}',
        xaxis_title="Hour of Day",
        yaxis_title="Equipment (Unit-Car-ID)",
        height=600,
        font=dict(size=12)
    )
    
    return fig

def create_daily_summary_chart(df):
    """Create daily summary chart showing derate trends"""
    daily_summary = df.groupby(['Date', 'UNIT']).agg({
        'derate_gap': ['mean', 'max', 'min', 'std'],
        'equipment_id': 'count'
    }).reset_index()
    
    # Flatten column names
    daily_summary.columns = ['Date', 'UNIT', 'Avg_Derate', 'Max_Derate', 'Min_Derate', 'Std_Derate', 'Records']
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Average Daily Derate by Unit', 'Maximum Daily Derate by Unit', 
                       'Derate Standard Deviation', 'Daily Record Count'),
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # Add traces for each unit
    for unit in sorted(df['UNIT'].unique()):
        unit_data = daily_summary[daily_summary['UNIT'] == unit]
        
        # Average derate
        fig.add_trace(
            go.Scatter(
                x=unit_data['Date'],
                y=unit_data['Avg_Derate'],
                mode='lines+markers',
                name=f'Unit {unit}',
                line=dict(width=3),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Max derate
        fig.add_trace(
            go.Scatter(
                x=unit_data['Date'],
                y=unit_data['Max_Derate'],
                mode='lines+markers',
                name=f'Unit {unit}',
                line=dict(dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        # Standard deviation
        fig.add_trace(
            go.Scatter(
                x=unit_data['Date'],
                y=unit_data['Std_Derate'],
                mode='lines+markers',
                name=f'Unit {unit}',
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Record count
        fig.add_trace(
            go.Scatter(
                x=unit_data['Date'],
                y=unit_data['Records'],
                mode='lines+markers',
                name=f'Unit {unit}',
                showlegend=False
            ),
            row=2, col=2
        )
    
    fig.update_layout(height=700, title_text="Daily Derate Analysis Summary")
    
    # Update axes labels
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Avg Derate %", row=1, col=1)
    fig.update_yaxes(title_text="Max Derate %", row=1, col=2)
    fig.update_yaxes(title_text="Std Derate %", row=2, col=1)
    fig.update_yaxes(title_text="Records", row=2, col=2)
    
    return fig

def create_equipment_comparison_chart(df):
    """Create chart comparing equipment performance"""
    equipment_summary = df.groupby(['equipment_id', 'UNIT', 'CAR']).agg({
        'derate_gap': ['mean', 'max', 'count']
    }).reset_index()
    
    # Flatten columns
    equipment_summary.columns = ['equipment_id', 'UNIT', 'CAR', 'Avg_Derate', 'Max_Derate', 'Records']
    equipment_summary['Equipment_Label'] = ('U' + equipment_summary['UNIT'].astype(str) + 
                                           'C' + equipment_summary['CAR'].astype(str) + 
                                           ' (' + equipment_summary['equipment_id'].astype(str) + ')')
    
    # Create bar chart
    fig = go.Figure()
    
    # Add bars for each unit
    for unit in sorted(df['UNIT'].unique()):
        unit_data = equipment_summary[equipment_summary['UNIT'] == unit]
        
        fig.add_trace(go.Bar(
            x=unit_data['Equipment_Label'],
            y=unit_data['Avg_Derate'],
            name=f'Unit {unit}',
            text=unit_data['Avg_Derate'].round(1),
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Average Derate % by Equipment',
        xaxis_title='Equipment',
        yaxis_title='Average Derate %',
        height=500,
        xaxis_tickangle=-45
    )
    
    return fig

def main():
    # Header
    st.markdown('<div class="main-header">🚂 Train Derate Analysis Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Analyze train derate performance across units and equipment with interactive heatmaps")
    
    # Sidebar
    st.sidebar.markdown("## 🚂 Derate Analysis Options")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload Your Derate Data", "Use Sample Data"]
    )
    
    df = None
    
    if data_source == "Upload Your Derate Data":
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
                    st.sidebar.markdown(f"**Equipment:** {df['equipment_id'].nunique()}")
                    st.sidebar.markdown(f"**Date Range:** {df['Date'].min()} to {df['Date'].max()}")
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        else:
            st.info("👆 Please upload your derate data file to get started!")
            
    else:
        df = load_sample_derate_data()
        df = process_derate_data(df)
        st.sidebar.markdown("### 🎯 Sample Data Loaded")
        st.sidebar.info("Using sample derate data for 2 units with 5 cars each")
    
    if df is not None:
        # Data overview
        st.markdown('<div class="sub-header">📊 Data Overview</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Records", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Units (Trains)", df['UNIT'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Equipment", df['equipment_id'].nunique())
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Date Range", f"{df['Date'].nunique()} days")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Equipment mapping display
        st.markdown('<div class="train-info">', unsafe_allow_html=True)
        st.markdown("**Equipment Mapping:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Unit 1:**")
            unit1_equipment = df[df['UNIT'] == 1]['equipment_id'].unique()
            for eq in sorted(unit1_equipment):
                car_num = df[df['equipment_id'] == eq]['CAR'].iloc[0]
                st.markdown(f"• Car {car_num}: Equipment {eq}")
        
        with col2:
            st.markdown("**Unit 2:**")
            unit2_equipment = df[df['UNIT'] == 2]['equipment_id'].unique()
            for eq in sorted(unit2_equipment):
                car_num = df[df['equipment_id'] == eq]['CAR'].iloc[0]
                st.markdown(f"• Car {car_num}: Equipment {eq}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data preview
        st.markdown('<div class="sub-header">📋 Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df.head(20), use_container_width=True)
        
        # Analysis options
        st.sidebar.markdown("### 🎨 Analysis Options")
        
        # Date selection
        available_dates = sorted(df['Date'].unique())
        selected_date = st.sidebar.selectbox(
            "Select specific date for detailed analysis:",
            ["All Dates"] + [str(date) for date in available_dates]
        )
        
        if selected_date != "All Dates":
            selected_date = pd.to_datetime(selected_date).date()
        else:
            selected_date = None
        
        # Visualizations
        st.markdown('<div class="sub-header">🔥 Derate Heatmap Analysis</div>', unsafe_allow_html=True)
        
        # Unit-specific heatmaps
        col1, col2 = st.columns(2)
        
        with col1:
            if 1 in df['UNIT'].values:
                fig_unit1 = create_unit_daily_heatmap(df, 1)
                if fig_unit1:
                    st.plotly_chart(fig_unit1, use_container_width=True)
        
        with col2:
            if 2 in df['UNIT'].values:
                fig_unit2 = create_unit_daily_heatmap(df, 2)
                if fig_unit2:
                    st.plotly_chart(fig_unit2, use_container_width=True)
        
        # Equipment comparison heatmap
        st.markdown('<div class="sub-header">🔧 Equipment Comparison</div>', unsafe_allow_html=True)
        fig_equipment = create_equipment_heatmap(df, selected_date)
        if fig_equipment:
            st.plotly_chart(fig_equipment, use_container_width=True)
        
        # Equipment performance comparison
        fig_equipment_bars = create_equipment_comparison_chart(df)
        if fig_equipment_bars:
            st.plotly_chart(fig_equipment_bars, use_container_width=True)
        
        # Daily summary
        st.markdown('<div class="sub-header">📈 Daily Summary Analysis</div>', unsafe_allow_html=True)
        fig_daily = create_daily_summary_chart(df)
        if fig_daily:
            st.plotly_chart(fig_daily, use_container_width=True)
        
        # Statistics summary
        st.markdown('<div class="sub-header">📊 Statistical Summary</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Overall Derate Statistics**")
            stats_df = df.groupby('UNIT')['derate_gap'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
            stats_df.columns = ['Average', 'Std Dev', 'Minimum', 'Maximum', 'Records']
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.markdown("**Equipment Performance Summary**")
            equipment_stats = df.groupby(['equipment_id', 'UNIT'])['derate_gap'].agg(['mean', 'max']).round(2)
            equipment_stats.columns = ['Avg Derate', 'Max Derate']
            st.dataframe(equipment_stats, use_container_width=True)
        
        # Download options
        st.markdown('<div class="sub-header">💾 Download Options</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Download Processed Data"):
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Processed CSV",
                    data=csv_buffer.getvalue(),
                    file_name="derate_analysis_processed.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Download Summary Stats"):
                stats_buffer = io.StringIO()
                df.groupby(['UNIT', 'equipment_id'])['derate_gap'].describe().to_csv(stats_buffer)
                st.download_button(
                    label="Download Statistics CSV",
                    data=stats_buffer.getvalue(),
                    file_name="derate_summary_stats.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📋 Download Equipment Mapping"):
                mapping_df = df[['equipment_id', 'UNIT', 'CAR', 'Equipment_Label']].drop_duplicates()
                mapping_buffer = io.StringIO()
                mapping_df.to_csv(mapping_buffer, index=False)
                st.download_button(
                    label="Download Equipment Mapping",
                    data=mapping_buffer.getvalue(),
                    file_name="equipment_mapping.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
