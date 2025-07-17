import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import io

# Set page config
st.set_page_config(
    page_title="Data Heatmap Visualizer",
    page_icon="🔥",
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
</style>
""", unsafe_allow_html=True)

def load_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Create sample datasets
    datasets = {
        "Sales Performance": pd.DataFrame({
            'Product_A': np.random.randint(10, 100, 12),
            'Product_B': np.random.randint(15, 85, 12),
            'Product_C': np.random.randint(5, 95, 12),
            'Product_D': np.random.randint(20, 80, 12),
            'Product_E': np.random.randint(25, 75, 12)
        }, index=[f'Month_{i+1}' for i in range(12)]),
        
        "Website Traffic": pd.DataFrame({
            'Homepage': np.random.randint(1000, 5000, 7),
            'Products': np.random.randint(800, 4000, 7),
            'About': np.random.randint(200, 1500, 7),
            'Contact': np.random.randint(100, 800, 7),
            'Blog': np.random.randint(300, 2000, 7)
        }, index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']),
        
        "Employee Performance": pd.DataFrame({
            'Communication': np.random.randint(60, 100, 8),
            'Technical_Skills': np.random.randint(65, 95, 8),
            'Leadership': np.random.randint(50, 90, 8),
            'Problem_Solving': np.random.randint(70, 100, 8),
            'Teamwork': np.random.randint(75, 100, 8)
        }, index=[f'Employee_{i+1}' for i in range(8)])
    }
    
    return datasets

def create_plotly_heatmap(df, title="Interactive Heatmap"):
    """Create an interactive heatmap using Plotly"""
    fig = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns,
        y=df.index,
        colorscale='Viridis',
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>' +
                      '<b>%{x}</b><br>' +
                      'Value: %{z}<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Categories",
        yaxis_title="Items",
        width=800,
        height=600,
        font=dict(size=12)
    )
    
    return fig

def create_seaborn_heatmap(df, title="Static Heatmap"):
    """Create a static heatmap using Seaborn"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df, annot=True, cmap='viridis', fmt='.1f', cbar_kws={'label': 'Value'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Categories', fontsize=14)
    plt.ylabel('Items', fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    return plt.gcf()

def process_uploaded_file(uploaded_file):
    """Process uploaded file and return DataFrame"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, index_col=0)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file, index_col=0)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
        
        # Convert to numeric where possible
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def main():
    # Header
    st.markdown('<div class="main-header">🔥 Data Heatmap Visualizer</div>', unsafe_allow_html=True)
    st.markdown("Upload your data and create beautiful interactive heatmaps!")
    
    # Sidebar
    st.sidebar.markdown("## 📊 Data Options")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload File", "Use Sample Data"]
    )
    
    df = None
    
    if data_source == "Upload File":
        st.sidebar.markdown("### 📁 File Upload")
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file with numeric data. First column should be row labels."
        )
        
        if uploaded_file is not None:
            df = process_uploaded_file(uploaded_file)
        else:
            st.info("👆 Please upload a file to get started!")
            
    else:
        st.sidebar.markdown("### 🎯 Sample Datasets")
        sample_datasets = load_sample_data()
        selected_sample = st.sidebar.selectbox(
            "Choose a sample dataset:",
            list(sample_datasets.keys())
        )
        df = sample_datasets[selected_sample]
    
    if df is not None:
        # Display data info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Rows", len(df))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Columns", len(df.columns))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Columns", len(numeric_cols))
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data preview
        st.markdown('<div class="sub-header">📋 Data Preview</div>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        
        # Visualization options
        st.sidebar.markdown("### 🎨 Visualization Options")
        
        # Filter numeric columns for heatmap
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.error("No numeric columns found in the data. Please upload data with numeric values.")
            return
        
        # Column selection
        if len(numeric_df.columns) > 1:
            selected_columns = st.sidebar.multiselect(
                "Select columns to visualize:",
                numeric_df.columns.tolist(),
                default=numeric_df.columns.tolist()
            )
            
            if selected_columns:
                numeric_df = numeric_df[selected_columns]
        
        # Heatmap type
        heatmap_type = st.sidebar.selectbox(
            "Heatmap Type:",
            ["Interactive (Plotly)", "Static (Seaborn)", "Both"]
        )
        
        # Color scheme
        color_scheme = st.sidebar.selectbox(
            "Color Scheme:",
            ["Viridis", "Plasma", "Inferno", "Magma", "Cividis", "Blues", "Reds", "Greens"]
        )
        
        # Normalization
        normalize = st.sidebar.checkbox("Normalize data", help="Scale values to 0-1 range")
        
        if normalize:
            numeric_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
        
        # Create visualizations
        st.markdown('<div class="sub-header">🔥 Heatmap Visualization</div>', unsafe_allow_html=True)
        
        if heatmap_type in ["Interactive (Plotly)", "Both"]:
            fig = create_plotly_heatmap(numeric_df, f"Interactive Heatmap - {data_source}")
            fig.update_traces(colorscale=color_scheme.lower())
            st.plotly_chart(fig, use_container_width=True)
        
        if heatmap_type in ["Static (Seaborn)", "Both"]:
            fig = create_seaborn_heatmap(numeric_df, f"Static Heatmap - {data_source}")
            st.pyplot(fig)
        
        # Statistics
        st.markdown('<div class="sub-header">📊 Data Statistics</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Summary Statistics**")
            st.dataframe(numeric_df.describe(), use_container_width=True)
        
        with col2:
            st.markdown("**Correlation Matrix**")
            if len(numeric_df.columns) > 1:
                corr_matrix = numeric_df.corr()
                fig_corr = create_plotly_heatmap(corr_matrix, "Correlation Matrix")
                fig_corr.update_traces(colorscale='RdBu', zmid=0)
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation matrix.")
        
        # Download options
        st.markdown('<div class="sub-header">💾 Download Options</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download Processed Data"):
                csv_buffer = io.StringIO()
                numeric_df.to_csv(csv_buffer)
                st.download_button(
                    label="Download CSV",
                    data=csv_buffer.getvalue(),
                    file_name="processed_data.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Download Statistics"):
                stats_buffer = io.StringIO()
                numeric_df.describe().to_csv(stats_buffer)
                st.download_button(
                    label="Download Statistics CSV",
                    data=stats_buffer.getvalue(),
                    file_name="data_statistics.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
