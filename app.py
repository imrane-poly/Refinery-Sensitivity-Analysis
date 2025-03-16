import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
from datetime import timedelta
import calendar
import os
import re

# Set page configuration
st.set_page_config(
    page_title="Refinery Capacity & Maintenance Impact Tool",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve UI
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stTabs [data-baseweb="tab-list"] {gap: 10px;}
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px;
        padding: 10px 16px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4a73c0;
        color: white;
    }
    .metric-card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .metric-value {font-size: 24px; font-weight: bold;}
    .metric-label {font-size: 14px; color: #666;}
    div[data-testid="stHorizontalBlock"] {gap: 10px;}
</style>
""", unsafe_allow_html=True)

# Caching data loading functions to improve performance
@st.cache_data(ttl=3600)
def load_maintenance_data():
    """Load and process refinery maintenance data"""
    try:
        df = pd.read_csv('RefineryMaintenance.2025-02-19T09-55.csv')
        
        # Convert date columns to datetime
        df['StartDate'] = pd.to_datetime(df['StartDate'])
        df['EndDate'] = pd.to_datetime(df['EndDate'])
        
        # Calculate duration in days
        df['Duration'] = (df['EndDate'] - df['StartDate']).dt.days
        
        # Drop rows with invalid dates or capacities
        df = df.dropna(subset=['StartDate', 'EndDate', 'CapacityOffline'])
        
        # Add additional columns for analysis
        df['Year'] = df['StartDate'].dt.year
        df['Month'] = df['StartDate'].dt.month
        df['MonthYear'] = df['StartDate'].dt.strftime('%b-%Y')
        
        # Calculate capacity-days (capacity * duration)
        df['CapacityDays'] = df['CapacityOffline'] * df['Duration']
        
        return df
    except Exception as e:
        st.error(f"Error loading maintenance data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_capacity_changes_data():
    """Load and process refinery capacity changes data"""
    try:
        df = pd.read_csv('RefineryCapacityChanges.2025-02-19T09-55.csv')
        
        # Convert date columns to datetime
        df['EstimatedCompletion'] = pd.to_datetime(df['EstimatedCompletion'])
        
        # Add additional columns for analysis
        df['Year'] = df['EstimatedCompletion'].dt.year
        df['Month'] = df['EstimatedCompletion'].dt.month
        df['MonthYear'] = df['EstimatedCompletion'].dt.strftime('%b-%Y')
        
        return df
    except Exception as e:
        st.error(f"Error loading capacity changes data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_regional_balance_data():
    """Load and process regional balance data"""
    try:
        df = pd.read_csv('Regional_Balance.2025-02-19T09-55.csv')
        
        # Convert date columns to datetime
        df['ReferenceDate'] = pd.to_datetime(df['ReferenceDate'])
        
        # Add additional columns for analysis
        df['Year'] = df['ReferenceDate'].dt.year
        df['Month'] = df['ReferenceDate'].dt.month
        df['MonthYear'] = df['ReferenceDate'].dt.strftime('%b-%Y')
        
        # Filter for refinery runs
        refinery_runs = df[df['FlowBreakdown'] == 'REFINOBS'].copy()
        
        return refinery_runs
    except Exception as e:
        st.error(f"Error loading regional balance data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_terminals_data():
    """Load and process terminals data"""
    try:
        # For the sample data, we'll load the representative sample
        df = pd.read_csv('Terminals.2025-02-19T09-55.csv')
        
        # Convert date columns to datetime
        df['ReferenceDate'] = pd.to_datetime(df['ReferenceDate'])
        
        # Add additional columns for analysis
        df['Year'] = df['ReferenceDate'].dt.year
        df['Month'] = df['ReferenceDate'].dt.month
        df['MonthYear'] = df['ReferenceDate'].dt.strftime('%b-%Y')
        
        return df
    except Exception as e:
        st.error(f"Error loading terminals data: {e}")
        return pd.DataFrame()

# Function to create date range for forecasting
def create_forecast_dates(start_date, periods=12):
    """Create forecast date range starting from a given date"""
    date_range = pd.date_range(start=start_date, periods=periods, freq='MS')
    return date_range

# Function to forecast maintenance impact
def forecast_maintenance_impact(maintenance_df, regions, start_date, periods=12):
    """
    Forecast maintenance impact on refinery runs
    
    Parameters:
    -----------
    maintenance_df : DataFrame
        Maintenance data
    regions : list
        List of regions to include
    start_date : datetime
        Start date for forecast
    periods : int
        Number of periods to forecast
    
    Returns:
    --------
    DataFrame
        Forecasted impact by region and date
    """
    # Create date range for forecast
    forecast_dates = create_forecast_dates(start_date, periods)
    
    # Initialize forecast dataframe
    forecast = pd.DataFrame(index=forecast_dates)
    
    # Filter maintenance events in the forecast period
    future_maintenance = maintenance_df[
        (maintenance_df['StartDate'] >= start_date) | 
        (maintenance_df['EndDate'] >= start_date)
    ]
    
    # Group by country and aggregate capacity offline
    country_impact = {}
    
    for date in forecast_dates:
        month_end = date + pd.offsets.MonthEnd(0)
        
        # Filter maintenance events active in this month
        month_maintenance = future_maintenance[
            ((future_maintenance['StartDate'] <= month_end) & 
             (future_maintenance['EndDate'] >= date))
        ]
        
        # Aggregate by country
        for country in month_maintenance['Country'].unique():
            country_data = month_maintenance[month_maintenance['Country'] == country]
            capacity_offline = country_data['CapacityOffline'].sum()
            
            if country not in country_impact:
                country_impact[country] = []
                
            country_impact[country].append({
                'Date': date,
                'CapacityOffline': capacity_offline
            })
    
    # Convert to DataFrame
    result = pd.DataFrame()
    for country, impacts in country_impact.items():
        if impacts:
            country_df = pd.DataFrame(impacts)
            country_df['Country'] = country
            result = pd.concat([result, country_df])
    
    # If we have regional mapping, we could map countries to regions here
    
    return result

# Function to analyze seasonal patterns
def analyze_seasonal_patterns(maintenance_df, capacity_df=None, years=5):
    """
    Analyze seasonal patterns in maintenance events
    
    Parameters:
    -----------
    maintenance_df : DataFrame
        Maintenance data
    capacity_df : DataFrame, optional
        Capacity changes data
    years : int
        Number of years of historical data to analyze
    
    Returns:
    --------
    DataFrame
        Seasonal patterns by month
    """
    # Get current year and start year for analysis
    current_year = datetime.datetime.now().year
    start_year = current_year - years
    
    # Filter for the analysis period
    historical = maintenance_df[maintenance_df['Year'] >= start_year]
    
    # Group by month and aggregate
    monthly_patterns = historical.groupby('Month').agg({
        'CapacityOffline': 'sum',
        'Country': 'count',
        'Duration': 'mean'
    }).reset_index()
    
    # Rename columns for clarity
    monthly_patterns = monthly_patterns.rename(columns={
        'Country': 'EventCount',
        'CapacityOffline': 'TotalCapacityOffline',
        'Duration': 'AverageDuration'
    })
    
    # Add month names
    monthly_patterns['MonthName'] = monthly_patterns['Month'].apply(
        lambda x: calendar.month_abbr[x]
    )
    
    # Sort by month
    monthly_patterns = monthly_patterns.sort_values('Month')
    
    return monthly_patterns

# Main app layout
def main():
    # Title
    st.title("🏭 Refinery Capacity & Maintenance Impact Tool")

    # Sidebar for filters and controls
    with st.sidebar:
        st.header("Filters & Controls")
        
        st.subheader("Date Range")
        today = datetime.datetime.now()
        start_date = st.date_input(
            "Start Date",
            value=today - timedelta(days=365),
            max_value=today
        )
        end_date = st.date_input(
            "End Date",
            value=today + timedelta(days=365),
            min_value=start_date
        )
        
        st.subheader("Regions")
        # These would be populated from actual data
        regions = ["OECD Americas", "OECD Europe", "OECD Asia", "China", 
                   "Other Asia", "Middle East", "Latin America", "Africa", "FSU"]
        selected_regions = st.multiselect(
            "Select Regions",
            options=regions,
            default=["OECD Americas", "OECD Europe", "China"]
        )
        
        st.subheader("Options")
        show_historical = st.checkbox("Show Historical Data", value=True)
        show_forecast = st.checkbox("Show Forecast", value=True)
        forecast_months = st.slider("Forecast Period (Months)", 1, 24, 12)
        
        st.markdown("---")
        st.caption("Data last updated: Feb 19, 2025")

    # Load data
    with st.spinner("Loading data..."):
        maintenance_df = load_maintenance_data()
        capacity_changes_df = load_capacity_changes_data()
        refinery_runs_df = load_regional_balance_data()
        terminals_df = load_terminals_data()
        
        # Check if data loaded successfully
        data_loaded = (
            not maintenance_df.empty and 
            not capacity_changes_df.empty and 
            not refinery_runs_df.empty
        )
        
        if not data_loaded:
            st.error("Failed to load one or more required datasets.")
            return

    # Convert selected dates to datetime for filtering
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    
    # Filter data based on date range
    maintenance_filtered = maintenance_df[
        ((maintenance_df['StartDate'] >= start_datetime) & (maintenance_df['StartDate'] <= end_datetime)) |
        ((maintenance_df['EndDate'] >= start_datetime) & (maintenance_df['EndDate'] <= end_datetime))
    ]
    
    capacity_changes_filtered = capacity_changes_df[
        (capacity_changes_df['EstimatedCompletion'] >= start_datetime) & 
        (capacity_changes_df['EstimatedCompletion'] <= end_datetime)
    ]
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard Overview", 
        "🔧 Maintenance Analysis", 
        "⚡ Capacity Changes", 
        "🔄 Run Rate Impact",
        "📈 Forecasts & Scenarios"
    ])
    
    # Tab 1: Dashboard Overview
    with tab1:
        st.header("Refinery Sector Overview")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Current metrics
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Active Maintenance Events</div>", unsafe_allow_html=True)
            current_events = len(maintenance_filtered[
                (maintenance_filtered['StartDate'] <= pd.to_datetime('today')) &
                (maintenance_filtered['EndDate'] >= pd.to_datetime('today'))
            ])
            st.markdown(f"<div class='metric-value'>{current_events}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Current Capacity Offline (KBD)</div>", unsafe_allow_html=True)
            current_capacity_offline = maintenance_filtered[
                (maintenance_filtered['StartDate'] <= pd.to_datetime('today')) &
                (maintenance_filtered['EndDate'] >= pd.to_datetime('today'))
            ]['CapacityOffline'].sum()
            st.markdown(f"<div class='metric-value'>{current_capacity_offline:,.0f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col3:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Upcoming Capacity Changes (KBD)</div>", unsafe_allow_html=True)
            upcoming_capacity_change = capacity_changes_filtered['NetChange'].sum()
            color = "green" if upcoming_capacity_change >= 0 else "red"
            st.markdown(f"<div class='metric-value' style='color:{color}'>{upcoming_capacity_change:+,.0f}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col4:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("<div class='metric-label'>Planned Events Next 90 Days</div>", unsafe_allow_html=True)
            upcoming_events = len(maintenance_filtered[
                (maintenance_filtered['StartDate'] > pd.to_datetime('today')) &
                (maintenance_filtered['StartDate'] <= pd.to_datetime('today') + pd.DateOffset(days=90))
            ])
            st.markdown(f"<div class='metric-value'>{upcoming_events}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Global map and charts
        st.subheader("Global Refinery Sector Status")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create a map of maintenance events
            if not maintenance_filtered.empty:
                # Create a copy for aggregation
                map_data = maintenance_filtered.copy()
                
                # Aggregate by country
                country_agg = map_data.groupby('Country').agg({
                    'CapacityOffline': 'sum',
                    'Refinery': 'count'
                }).reset_index()
                
                country_agg.rename(columns={'Refinery': 'EventCount'}, inplace=True)
                
                # Create the map
                fig = px.choropleth(
                    country_agg,
                    locations='Country',
                    locationmode='country names',
                    color='CapacityOffline',
                    hover_name='Country',
                    hover_data=['EventCount', 'CapacityOffline'],
                    color_continuous_scale='Reds',
                    title='Refinery Capacity Offline by Country (KBD)',
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No maintenance events in the selected date range.")
        
        with col2:
            # Project type breakdown
            if not capacity_changes_filtered.empty:
                project_counts = capacity_changes_filtered['ProjectType'].value_counts().reset_index()
                project_counts.columns = ['ProjectType', 'Count']
                
                colors = {
                    'New': '#2ECC71',
                    'Expansion': '#3498DB',
                    'Closure': '#E74C3C',
                    'Reduction': '#F39C12'
                }
                
                fig = px.pie(
                    project_counts,
                    values='Count',
                    names='ProjectType',
                    title='Capacity Changes by Project Type',
                    color='ProjectType',
                    color_discrete_map=colors
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No capacity changes in the selected date range.")
        
        # Capacity offline by month
        st.subheader("Maintenance Outlook")
        
        if not maintenance_filtered.empty:
            # Create date bins for the next 12 months
            today = pd.to_datetime('today')
            next_year = today + pd.DateOffset(months=12)
            date_bins = pd.date_range(today, next_year, freq='MS')
            
            # Initialize data structure for capacity offline by month
            monthly_capacity = {date.strftime('%b-%Y'): 0 for date in date_bins}
            
            # Calculate capacity offline for each month
            for _, event in maintenance_filtered.iterrows():
                for date in date_bins:
                    month_start = date
                    month_end = date + pd.offsets.MonthEnd(0)
                    
                    # Check if maintenance event is active in this month
                    if event['StartDate'] <= month_end and event['EndDate'] >= month_start:
                        # Calculate days in month that event is active
                        days_in_month = (min(month_end, event['EndDate']) - 
                                         max(month_start, event['StartDate'])).days + 1
                        days_in_month = max(0, days_in_month)
                        
                        # Calculate proportion of month
                        month_days = (month_end - month_start).days + 1
                        proportion = days_in_month / month_days
                        
                        # Add proportional capacity offline
                        monthly_capacity[date.strftime('%b-%Y')] += event['CapacityOffline'] * proportion
            
            # Convert to dataframe
            monthly_df = pd.DataFrame([
                {'Month': month, 'CapacityOffline': capacity}
                for month, capacity in monthly_capacity.items()
            ])
            
            # Create bar chart
            fig = px.bar(
                monthly_df,
                x='Month',
                y='CapacityOffline',
                title='Projected Refinery Capacity Offline (KBD)',
                labels={'CapacityOffline': 'Capacity Offline (KBD)', 'Month': ''},
                color_discrete_sequence=['#F39C12']
            )
            
            fig.update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': monthly_df['Month'].tolist()},
                height=400,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No maintenance events in the selected date range.")
    
    # Tab 2: Maintenance Analysis
    with tab2:
        st.header("Refinery Maintenance Analysis")
        
        # Filters specific to maintenance
        col1, col2, col3 = st.columns(3)
        with col1:
            outage_types = maintenance_filtered['OutageType'].unique().tolist()
            selected_outage_types = st.multiselect(
                "Outage Types",
                options=outage_types,
                default=outage_types
            )
        
        with col2:
            countries = maintenance_filtered['Country'].unique().tolist()
            selected_countries = st.multiselect(
                "Countries",
                options=countries,
                default=[]
            )
        
        with col3:
            min_capacity = int(maintenance_filtered['CapacityOffline'].min())
            max_capacity = int(maintenance_filtered['CapacityOffline'].max())
            capacity_range = st.slider(
                "Capacity Offline Range (KBD)",
                min_capacity,
                max_capacity,
                (min_capacity, max_capacity)
            )
        
        # Apply additional filters
        maintenance_view = maintenance_filtered.copy()
        
        if selected_outage_types:
            maintenance_view = maintenance_view[maintenance_view['OutageType'].isin(selected_outage_types)]
        
        if selected_countries:
            maintenance_view = maintenance_view[maintenance_view['Country'].isin(selected_countries)]
        
        maintenance_view = maintenance_view[
            (maintenance_view['CapacityOffline'] >= capacity_range[0]) &
            (maintenance_view['CapacityOffline'] <= capacity_range[1])
        ]
        
        # Display maintenance data
        if not maintenance_view.empty:
            # Timeline view
            st.subheader("Maintenance Events Timeline")
            
            # Create timeline data
            timeline_data = []
            for _, event in maintenance_view.iterrows():
                timeline_data.append({
                    'Task': f"{event['Country']} - {event['Refinery']}",
                    'Start': event['StartDate'],
                    'Finish': event['EndDate'],
                    'Capacity': event['CapacityOffline'],
                    'OutageType': event['OutageType']
                })
            
            # Convert to dataframe
            timeline_df = pd.DataFrame(timeline_data)
            
            # Sort by start date
            timeline_df = timeline_df.sort_values('Start')
            
            # Limit to top events by capacity for readability
            top_events = timeline_df.nlargest(20, 'Capacity')
            
            # Create Gantt chart
            fig = px.timeline(
                top_events,
                x_start='Start',
                x_end='Finish',
                y='Task',
                color='OutageType',
                color_discrete_map={'Actual': '#3498DB', 'Implied': '#F39C12'},
                hover_data=['Capacity'],
                title='Top 20 Maintenance Events by Capacity'
            )
            
            fig.update_layout(
                height=600,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            fig.update_yaxes(autorange="reversed")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Maintenance statistics
            st.subheader("Maintenance Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Capacity offline by outage type
                outage_type_agg = maintenance_view.groupby('OutageType').agg({
                    'CapacityOffline': 'sum',
                    'Refinery': 'count'
                }).reset_index()
                
                outage_type_agg.rename(columns={'Refinery': 'EventCount'}, inplace=True)
                
                fig = px.bar(
                    outage_type_agg,
                    x='OutageType',
                    y='CapacityOffline',
                    text='EventCount',
                    title='Capacity Offline by Outage Type',
                    labels={
                        'OutageType': 'Outage Type',
                        'CapacityOffline': 'Capacity Offline (KBD)',
                        'EventCount': 'Event Count'
                    },
                    color='OutageType',
                    color_discrete_map={'Actual': '#3498DB', 'Implied': '#F39C12'}
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Maintenance events by country (top 10)
                country_agg = maintenance_view.groupby('Country').agg({
                    'CapacityOffline': 'sum',
                    'Refinery': 'count'
                }).reset_index()
                
                country_agg.rename(columns={'Refinery': 'EventCount'}, inplace=True)
                
                # Sort by capacity offline
                top_countries = country_agg.nlargest(10, 'CapacityOffline')
                
                fig = px.bar(
                    top_countries,
                    x='Country',
                    y='CapacityOffline',
                    text='EventCount',
                    title='Top 10 Countries by Capacity Offline',
                    labels={
                        'Country': 'Country',
                        'CapacityOffline': 'Capacity Offline (KBD)',
                        'EventCount': 'Event Count'
                    },
                    color='CapacityOffline',
                    color_continuous_scale='Reds'
                )
                
                fig.update_layout(
                    height=400,
                    margin=dict(l=10, r=10, t=50, b=10),
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Seasonal patterns
            st.subheader("Maintenance Seasonal Patterns")
            
            # Analyze seasonal patterns
            seasonal_patterns = analyze_seasonal_patterns(maintenance_df)
            
            # Create line chart
            fig = px.line(
                seasonal_patterns,
                x='MonthName',
                y='TotalCapacityOffline',
                title='Historical Maintenance Capacity Offline by Month',
                labels={
                    'MonthName': 'Month',
                    'TotalCapacityOffline': 'Total Capacity Offline (KBD)'
                },
                markers=True,
                color_discrete_sequence=['#E74C3C']
            )
            
            # Add event count as bar chart
            fig.add_bar(
                x=seasonal_patterns['MonthName'],
                y=seasonal_patterns['EventCount'],
                name='Event Count',
                yaxis='y2'
            )
            
            # Update layout for dual y-axis
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis={'categoryorder': 'array', 'categoryarray': seasonal_patterns['MonthName'].tolist()},
                yaxis2={
                    'title': 'Event Count',
                    'overlaying': 'y',
                    'side': 'right'
                },
                legend={'orientation': 'h', 'y': -0.15}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed data table
            st.subheader("Maintenance Events Detail")
            
            # Select columns to display
            display_cols = [
                'Country', 'Refinery', 'OutageType', 'StartDate', 'EndDate', 
                'Duration', 'CapacityOffline'
            ]
            
            # Convert dates to string for display
            table_view = maintenance_view[display_cols].copy()
            table_view['StartDate'] = table_view['StartDate'].dt.strftime('%Y-%m-%d')
            table_view['EndDate'] = table_view['EndDate'].dt.strftime('%Y-%m-%d')
            
            # Sort by start date
            table_view = table_view.sort_values('StartDate', ascending=False)
            
            # Display table
            st.dataframe(
                table_view,
                column_config={
                    'Country': st.column_config.TextColumn('Country'),
                    'Refinery': st.column_config.TextColumn('Refinery'),
                    'OutageType': st.column_config.TextColumn('Type'),
                    'StartDate': st.column_config.TextColumn('Start Date'),
                    'EndDate': st.column_config.TextColumn('End Date'),
                    'Duration': st.column_config.NumberColumn('Duration (Days)'),
                    'CapacityOffline': st.column_config.NumberColumn(
                        'Capacity Offline (KBD)',
                        format="%.0f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No maintenance events matching the selected filters.")
    
    # Tab 3: Capacity Changes
    with tab3:
        st.header("Refinery Capacity Changes Analysis")
        
        # Filters specific to capacity changes
        col1, col2, col3 = st.columns(3)
        with col1:
            project_types = capacity_changes_filtered['ProjectType'].unique().tolist()
            selected_project_types = st.multiselect(
                "Project Types",
                options=project_types,
                default=project_types
            )
        
        with col2:
            regions = capacity_changes_filtered['Region'].unique().tolist()
            selected_capacity_regions = st.multiselect(
                "Regions",
                options=regions,
                default=[]
            )
        
        with col3:
            min_change = int(capacity_changes_filtered['NetChange'].min())
            max_change = int(capacity_changes_filtered['NetChange'].max())
            change_range = st.slider(
                "Net Change Range (KBD)",
                min_change,
                max_change,
                (min_change, max_change)
            )
        
        # Apply additional filters
        capacity_view = capacity_changes_filtered.copy()
        
        if selected_project_types:
            capacity_view = capacity_view[capacity_view['ProjectType'].isin(selected_project_types)]
        
        if selected_capacity_regions:
            capacity_view = capacity_view[capacity_view['Region'].isin(selected_capacity_regions)]
        
        capacity_view = capacity_view[
            (capacity_view['NetChange'] >= change_range[0]) &
            (capacity_view['NetChange'] <= change_range[1])
        ]
        
        # Display capacity changes data
        if not capacity_view.empty:
            # Capacity changes by region
            st.subheader("Capacity Changes by Region")
            
            # Aggregate by region
            region_agg = capacity_view.groupby('Region').agg({
                'NetChange': 'sum',
                'Refinery': 'count'
            }).reset_index()
            
            region_agg.rename(columns={'Refinery': 'ProjectCount'}, inplace=True)
            
            # Sort by net change
            region_agg = region_agg.sort_values('NetChange', ascending=False)
            
            # Create waterfall chart using a bar chart
            fig = go.Figure()
            
            # Add bars for each region
            for i, row in region_agg.iterrows():
                color = '#2ECC71' if row['NetChange'] > 0 else '#E74C3C'
                fig.add_trace(go.Bar(
                    name=row['Region'],
                    x=[row['Region']],
                    y=[row['NetChange']],
                    text=[f"{row['ProjectCount']} projects"],
                    marker_color=color
                ))
            
            # Add net total
            total_change = region_agg['NetChange'].sum()
            color = '#2ECC71' if total_change > 0 else '#E74C3C'
            fig.add_trace(go.Bar(
                name='Total',
                x=['Total'],
                y=[total_change],
                marker_color='#3498DB'
            ))
            
            fig.update_layout(
                title='Net Capacity Changes by Region (KBD)',
                yaxis_title='Net Change (KBD)',
                height=500,
                margin=dict(l=10, r=10, t=50, b=10),
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Capacity changes timeline
            st.subheader("Capacity Changes Timeline")
            
            # Create timeline data
            timeline_data = []
            for _, project in capacity_view.iterrows():
                size = abs(project['NetChange'])
                color = '#2ECC71' if project['NetChange'] > 0 else '#E74C3C'
                
                timeline_data.append({
                    'Project': f"{project['Country']} - {project['Refinery']}",
                    'Date': project['EstimatedCompletion'],
                    'NetChange': project['NetChange'],
                    'Size': size,
                    'Color': color,
                    'ProjectType': project['ProjectType']
                })
            
            # Convert to dataframe
            timeline_df = pd.DataFrame(timeline_data)
            
            # Sort by date
            timeline_df = timeline_df.sort_values('Date')
            
            # Create scatter plot for timeline
            fig = px.scatter(
                timeline_df,
                x='Date',
                y='Project',
                size='Size',
                color='ProjectType',
                color_discrete_map={
                    'New': '#2ECC71',
                    'Expansion': '#3498DB',
                    'Closure': '#E74C3C',
                    'Reduction': '#F39C12'
                },
                hover_data=['NetChange'],
                title='Capacity Changes Timeline'
            )
            
            fig.update_layout(
                height=600,
                margin=dict(l=10, r=10, t=50, b=10),
                yaxis={'categoryorder': 'total ascending'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Capacity changes by company
            st.subheader("Capacity Changes by Company")
            
            # Aggregate by company
            company_agg = capacity_view.groupby('Company').agg({
                'NetChange': 'sum',
                'Refinery': 'count'
            }).reset_index()
            
            company_agg.rename(columns={'Refinery': 'ProjectCount'}, inplace=True)
            
            # Sort by absolute net change
            company_agg['AbsNetChange'] = company_agg['NetChange'].abs()
            top_companies = company_agg.nlargest(10, 'AbsNetChange')
            
            # Create horizontal bar chart
            fig = px.bar(
                top_companies,
                y='Company',
                x='NetChange',
                text='ProjectCount',
                title='Top 10 Companies by Capacity Change',
                labels={
                    'Company': 'Company',
                    'NetChange': 'Net Change (KBD)',
                    'ProjectCount': 'Project Count'
                },
                color='NetChange',
                color_continuous_scale=['#E74C3C', '#FFFFFF', '#2ECC71'],
                color_continuous_midpoint=0
            )
            
            fig.update_layout(
                height=500,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed data table
            st.subheader("Capacity Changes Detail")
            
            # Select columns to display
            display_cols = [
                'Country', 'Region', 'Refinery', 'Company', 'ProjectType', 
                'EstimatedCompletion', 'NetChange'
            ]
            
            # Convert dates to string for display
            table_view = capacity_view[display_cols].copy()
            table_view['EstimatedCompletion'] = table_view['EstimatedCompletion'].dt.strftime('%Y-%m-%d')
            
            # Sort by estimated completion date
            table_view = table_view.sort_values('EstimatedCompletion')
            
            # Display table
            st.dataframe(
                table_view,
                column_config={
                    'Country': st.column_config.TextColumn('Country'),
                    'Region': st.column_config.TextColumn('Region'),
                    'Refinery': st.column_config.TextColumn('Refinery'),
                    'Company': st.column_config.TextColumn('Company'),
                    'ProjectType': st.column_config.TextColumn('Project Type'),
                    'EstimatedCompletion': st.column_config.TextColumn('Est. Completion'),
                    'NetChange': st.column_config.NumberColumn(
                        'Net Change (KBD)',
                        format="%.1f"
                    )
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No capacity changes matching the selected filters.")
    
    # Tab 4: Run Rate Impact
    with tab4:
        st.header("Refinery Run Rate Impact Analysis")
        
        # Analyze refinery runs data
        if not refinery_runs_df.empty:
            # Filter for REFINOBS (refinery observed runs)
            refinery_runs = refinery_runs_df[refinery_runs_df['FlowBreakdown'] == 'REFINOBS'].copy()
            
            # Group by region and date
            runs_by_region_date = refinery_runs.groupby(['GroupName', 'ReferenceDate']).agg({
                'ObservedValue': 'mean'  # Taking mean in case of duplicates
            }).reset_index()
            
            # Get list of regions
            run_regions = runs_by_region_date['GroupName'].unique().tolist()
            
            # Region selection
            selected_run_regions = st.multiselect(
                "Select Regions for Run Rate Analysis",
                options=run_regions,
                default=run_regions[:3] if len(run_regions) >= 3 else run_regions
            )
            
            if selected_run_regions:
                # Filter for selected regions
                selected_runs = runs_by_region_date[
                    runs_by_region_date['GroupName'].isin(selected_run_regions)
                ]
                
                # Plot historical refinery runs
                st.subheader("Historical Refinery Runs by Region")
                
                # Create line chart
                fig = px.line(
                    selected_runs,
                    x='ReferenceDate',
                    y='ObservedValue',
                    color='GroupName',
                    title='Historical Refinery Runs by Region',
                    labels={
                        'ReferenceDate': 'Date',
                        'ObservedValue': 'Refinery Runs (KBD)',
                        'GroupName': 'Region'
                    },
                )
                
                fig.update_layout(
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend={'orientation': 'h', 'y': -0.15}
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Maintenance impact on run rates
                st.subheader("Maintenance Impact on Run Rates")
                
                # Select reference region
                reference_region = st.selectbox(
                    "Select Reference Region",
                    options=selected_run_regions,
                    index=0
                )
                
                # Get maintenance data for the reference region
                # Note: In a real app, we'd need a mapping of countries to regions
                # For the demo, we'll simulate this by aggregating maintenance by year-month
                
                # Create monthly maintenance aggregation
                monthly_maintenance = maintenance_df.groupby([
                    maintenance_df['StartDate'].dt.to_period('M')
                ]).agg({
                    'CapacityOffline': 'sum'
                }).reset_index()
                
                # Convert period to datetime for plotting
                monthly_maintenance['Date'] = monthly_maintenance['StartDate'].dt.to_timestamp()
                
                # Get runs data for reference region
                reference_runs = runs_by_region_date[runs_by_region_date['GroupName'] == reference_region]
                
                # Create dual-axis chart
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add runs line
                fig.add_trace(
                    go.Scatter(
                        x=reference_runs['ReferenceDate'],
                        y=reference_runs['ObservedValue'],
                        name=f"{reference_region} Runs",
                        line=dict(color='#3498DB')
                    ),
                    secondary_y=False
                )
                
                # Add maintenance line
                fig.add_trace(
                    go.Scatter(
                        x=monthly_maintenance['Date'],
                        y=monthly_maintenance['CapacityOffline'],
                        name='Capacity Offline',
                        line=dict(color='#E74C3C')
                    ),
                    secondary_y=True
                )
                
                # Update layout
                fig.update_layout(
                    title=f"Refinery Runs vs. Maintenance for {reference_region}",
                    height=500,
                    margin=dict(l=10, r=10, t=50, b=10),
                    legend={'orientation': 'h', 'y': -0.15}
                )
                
                fig.update_yaxes(
                    title_text="Refinery Runs (KBD)",
                    secondary_y=False
                )
                
                fig.update_yaxes(
                    title_text="Capacity Offline (KBD)",
                    secondary_y=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Run rate impact simulator
                st.subheader("Run Rate Impact Simulator")
                
                # Simulation parameters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    utilization_factor = st.slider(
                        "Utilization Factor for Available Capacity",
                        min_value=0.5,
                        max_value=1.0,
                        value=0.85,
                        step=0.01
                    )
                
                with col2:
                    maintenance_impact = st.slider(
                        "Maintenance Impact Factor",
                        min_value=0.5,
                        max_value=1.5,
                        value=1.0,
                        step=0.1,
                        help="Factor to adjust the impact of maintenance on run rates"
                    )
                
                with col3:
                    forecast_months = st.slider(
                        "Forecast Months",
                        min_value=1,
                        max_value=12,
                        value=6
                    )
                
                # Get the most recent run rate for the reference region
                if not reference_runs.empty:
                    recent_runs = reference_runs.sort_values('ReferenceDate', ascending=False).iloc[0]
                    recent_run_rate = recent_runs['ObservedValue']
                    recent_date = recent_runs['ReferenceDate']
                    
                    # Get upcoming maintenance for simulation period
                    forecast_end = recent_date + pd.DateOffset(months=forecast_months)
                    upcoming_maintenance = maintenance_df[
                        (maintenance_df['StartDate'] >= recent_date) &
                        (maintenance_df['StartDate'] <= forecast_end)
                    ]
                    
                    # Get upcoming capacity changes
                    upcoming_capacity = capacity_changes_df[
                        (capacity_changes_df['EstimatedCompletion'] >= recent_date) &
                        (capacity_changes_df['EstimatedCompletion'] <= forecast_end)
                    ]
                    
                    # Create forecast dates
                    forecast_dates = pd.date_range(
                        start=recent_date,
                        periods=forecast_months + 1,
                        freq='MS'
                    )
                    
                    # Initialize forecast data
                    forecast_data = []
                    
                    # Base run rate (most recent)
                    current_run_rate = recent_run_rate
                    
                    # Simulate each month
                    for i, date in enumerate(forecast_dates):
                        month_end = date + pd.offsets.MonthEnd(0)
                        
                        # Calculate maintenance impact for this month
                        month_maintenance = maintenance_df[
                            ((maintenance_df['StartDate'] <= month_end) & 
                             (maintenance_df['EndDate'] >= date))
                        ]
                        
                        maintenance_offline = month_maintenance['CapacityOffline'].sum() * maintenance_impact
                        
                        # Calculate capacity changes up to this month
                        month_capacity_changes = capacity_changes_df[
                            (capacity_changes_df['EstimatedCompletion'] >= recent_date) &
                            (capacity_changes_df['EstimatedCompletion'] <= month_end)
                        ]
                        
                        capacity_change = month_capacity_changes['NetChange'].sum() * utilization_factor
                        
                        # Calculate forecast run rate
                        if i == 0:
                            # First month is actual
                            forecast_run_rate = current_run_rate
                        else:
                            # Adjust for maintenance and capacity changes
                            forecast_run_rate = current_run_rate - maintenance_offline + capacity_change
                        
                        # Store forecast data
                        forecast_data.append({
                            'Date': date,
                            'RunRate': forecast_run_rate,
                            'MaintenanceOffline': maintenance_offline,
                            'CapacityChange': capacity_change
                        })
                    
                    # Convert to dataframe
                    forecast_df = pd.DataFrame(forecast_data)
                    
                    # Create forecast chart
                    fig = go.Figure()
                    
                    # Add actual run rate
                    fig.add_trace(go.Scatter(
                        x=reference_runs['ReferenceDate'],
                        y=reference_runs['ObservedValue'],
                        name='Historical Runs',
                        line=dict(color='#3498DB')
                    ))
                    
                    # Add forecast run rate
                    fig.add_trace(go.Scatter(
                        x=forecast_df['Date'],
                        y=forecast_df['RunRate'],
                        name='Forecast Runs',
                        line=dict(color='#2ECC71', dash='dash')
                    ))
                    
                    # Add maintenance impact
                    fig.add_trace(go.Bar(
                        x=forecast_df['Date'],
                        y=forecast_df['MaintenanceOffline'],
                        name='Maintenance Offline',
                        marker_color='#E74C3C',
                        opacity=0.7
                    ))
                    
                    # Add capacity changes
                    fig.add_trace(go.Bar(
                        x=forecast_df['Date'],
                        y=forecast_df['CapacityChange'],
                        name='Capacity Change',
                        marker_color='#F39C12',
                        opacity=0.7
                    ))
                    
                    # Update layout
                    fig.update_layout(
                        title=f"Run Rate Forecast for {reference_region}",
                        xaxis_title='Date',
                        yaxis_title='Refinery Runs (KBD)',
                        height=500,
                        margin=dict(l=10, r=10, t=50, b=10),
                        legend={'orientation': 'h', 'y': -0.15}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast summary
                    st.subheader("Forecast Summary")
                    
                    # Calculate key metrics
                    avg_forecast = forecast_df['RunRate'].mean()
                    max_forecast = forecast_df['RunRate'].max()
                    min_forecast = forecast_df['RunRate'].min()
                    total_maintenance = forecast_df['MaintenanceOffline'].sum()
                    total_capacity_change = forecast_df['CapacityChange'].sum()
                    
                    # Display summary
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric(
                            label="Avg Forecast Run Rate",
                            value=f"{avg_forecast:.0f} KBD",
                            delta=f"{avg_forecast - recent_run_rate:.0f} KBD"
                        )
                    
                    with col2:
                        st.metric(
                            label="Max Forecast Run Rate",
                            value=f"{max_forecast:.0f} KBD"
                        )
                    
                    with col3:
                        st.metric(
                            label="Min Forecast Run Rate",
                            value=f"{min_forecast:.0f} KBD"
                        )
                    
                    with col4:
                        st.metric(
                            label="Total Maintenance Impact",
                            value=f"{total_maintenance:.0f} KBD"
                        )
                    
                    with col5:
                        st.metric(
                            label="Total Capacity Change",
                            value=f"{total_capacity_change:.0f} KBD"
                        )
                else:
                    st.warning("No runs data available for the selected region.")
            else:
                st.info("Please select at least one region.")
        else:
            st.warning("No refinery runs data available.")
    
    # Tab 5: Forecasts & Scenarios
    with tab5:
        st.header("Refinery Forecasts & Scenario Analysis")
        
        # Scenario builder
        st.subheader("Scenario Builder")
        
        col1, col2 = st.columns(2)
        
        with col1:
            scenario_option = st.selectbox(
                "Scenario Type",
                options=["Base Case", "High Maintenance", "Low Maintenance", "Major Outage", "New Capacity"]
            )
        
        with col2:
            scenario_region = st.selectbox(
                "Region",
                options=regions,
                index=0
            )
        
        # Define scenario parameters based on selection
        if scenario_option == "Base Case":
            maintenance_factor = 1.0
            capacity_factor = 1.0
            utilization_factor = 0.85
            scenario_description = "Base case with expected maintenance and capacity changes"
        elif scenario_option == "High Maintenance":
            maintenance_factor = 1.5
            capacity_factor = 1.0
            utilization_factor = 0.80
            scenario_description = "Higher than expected maintenance events (+50%)"
        elif scenario_option == "Low Maintenance":
            maintenance_factor = 0.7
            capacity_factor = 1.0
            utilization_factor = 0.90
            scenario_description = "Lower than expected maintenance events (-30%)"
        elif scenario_option == "Major Outage":
            maintenance_factor = 2.0
            capacity_factor = 0.9
            utilization_factor = 0.75
            scenario_description = "Major unplanned outage scenario with significant impact"
        else:  # New Capacity
            maintenance_factor = 1.0
            capacity_factor = 1.5
            utilization_factor = 0.85
            scenario_description = "Additional new capacity coming online"
        
        st.info(scenario_description)
        
        # Create forecast date range
        today = pd.to_datetime('today')
        forecast_end = today + pd.DateOffset(months=12)
        forecast_dates = pd.date_range(today, forecast_end, freq='MS')
        
        # Create dummy forecast data for demonstration
        # In a real app, this would use the actual data and scenario parameters
        base_run_rate = 10000  # Example starting point
        
        base_scenario = []
        alt_scenario = []
        
        for i, date in enumerate(forecast_dates):
            # Base case
            base_value = base_run_rate + (i * 50) * np.sin(i/2)
            
            # Alternative scenario based on selected parameters
            alt_value = (base_run_rate + (i * 50) * np.sin(i/2)) * (1 - 0.1 * maintenance_factor + 0.05 * capacity_factor)
            
            base_scenario.append({
                'Date': date,
                'RunRate': base_value,
                'Scenario': 'Base Case'
            })
            
            alt_scenario.append({
                'Date': date,
                'RunRate': alt_value,
                'Scenario': scenario_option
            })
        
        # Combine scenarios
        scenarios_df = pd.DataFrame(base_scenario + alt_scenario)
        
        # Create scenario comparison chart
        fig = px.line(
            scenarios_df,
            x='Date',
            y='RunRate',
            color='Scenario',
            title=f'Scenario Comparison for {scenario_region}',
            labels={
                'Date': 'Date',
                'RunRate': 'Refinery Runs (KBD)',
                'Scenario': 'Scenario'
            },
            color_discrete_map={
                'Base Case': '#3498DB',
                scenario_option: '#E74C3C'
            }
        )
        
        fig.update_layout(
            height=500,
            margin=dict(l=10, r=10, t=50, b=10),
            legend={'orientation': 'h', 'y': -0.15}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Scenario impact analysis
        st.subheader("Scenario Impact Analysis")
        
        # Calculate impact metrics
        avg_base = np.mean([item['RunRate'] for item in base_scenario])
        avg_alt = np.mean([item['RunRate'] for item in alt_scenario])
        
        diff_pct = ((avg_alt - avg_base) / avg_base) * 100
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Base Case Avg Run Rate",
                value=f"{avg_base:.0f} KBD"
            )
        
        with col2:
            st.metric(
                label=f"{scenario_option} Avg Run Rate",
                value=f"{avg_alt:.0f} KBD",
                delta=f"{diff_pct:.1f}%"
            )
        
        with col3:
            impact_value = abs(avg_alt - avg_base) * 30  # Monthly to daily
            st.metric(
                label="Monthly Volume Impact",
                value=f"{impact_value:.0f} KB"
            )
        
        # Scenario comparison table
        monthly_comparison = []
        
        for i, date in enumerate(forecast_dates):
            base_value = base_scenario[i]['RunRate']
            alt_value = alt_scenario[i]['RunRate']
            diff = alt_value - base_value
            diff_pct = (diff / base_value) * 100 if base_value else 0
            
            monthly_comparison.append({
                'Month': date.strftime('%b %Y'),
                'Base Case': base_value,
                scenario_option: alt_value,
                'Difference': diff,
                'Difference %': diff_pct
            })
        
        monthly_df = pd.DataFrame(monthly_comparison)
        
        st.subheader("Monthly Comparison")
        st.dataframe(
            monthly_df,
            column_config={
                'Month': st.column_config.TextColumn('Month'),
                'Base Case': st.column_config.NumberColumn(
                    'Base Case (KBD)',
                    format="%.0f"
                ),
                scenario_option: st.column_config.NumberColumn(
                    f'{scenario_option} (KBD)',
                    format="%.0f"
                ),
                'Difference': st.column_config.NumberColumn(
                    'Difference (KBD)',
                    format="%.0f"
                ),
                'Difference %': st.column_config.NumberColumn(
                    'Difference %',
                    format="%.1f%%"
                )
            },
            hide_index=True,
            use_container_width=True
        )
        
        # Market impact analysis
        st.subheader("Market Impact Analysis")
        
        # Dummy data for market impact simulation
        impact_categories = [
            "Crude Demand Impact",
            "Product Supply Impact",
            "Trade Flow Impact",
            "Price Impact",
            "Storage Impact"
        ]
        
        impact_base = [0, 0, 0, 0, 0]
        impact_alt = [
            diff_pct * 1.2,  # Crude demand impact
            diff_pct * 0.9,  # Product supply impact
            diff_pct * 0.7,  # Trade flow impact
            diff_pct * 0.5,  # Price impact
            diff_pct * 0.3   # Storage impact
        ]
        
        # Create impact comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=impact_categories,
            y=impact_base,
            name='Base Case',
            marker_color='#3498DB'
        ))
        
        fig.add_trace(go.Bar(
            x=impact_categories,
            y=impact_alt,
            name=scenario_option,
            marker_color='#E74C3C'
        ))
        
        fig.update_layout(
            title=f'Market Impact Analysis for {scenario_option}',
            xaxis_title='Impact Category',
            yaxis_title='Impact (%)',
            height=400,
            margin=dict(l=10, r=10, t=50, b=10),
            legend={'orientation': 'h', 'y': -0.15}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendation
        st.subheader("Scenario Recommendations")
        
        if diff_pct < -5:
            recommendation = f"""
            ### Significant Negative Impact
            
            The {scenario_option} scenario shows a significant negative impact of {diff_pct:.1f}% on refinery run rates.
            
            **Recommended Actions:**
            - Monitor for potential product supply constraints
            - Evaluate alternative sourcing options
            - Consider adjusting trade positions to account for reduced runs
            - Watch for potential price impacts in affected regions
            """
        elif diff_pct > 5:
            recommendation = f"""
            ### Significant Positive Impact
            
            The {scenario_option} scenario shows a significant positive impact of {diff_pct:.1f}% on refinery run rates.
            
            **Recommended Actions:**
            - Evaluate potential for increased crude demand
            - Monitor for product oversupply conditions
            - Consider adjusting trade positions to account for increased runs
            - Watch for potential price impacts in affected regions
            """
        else:
            recommendation = f"""
            ### Moderate Impact
            
            The {scenario_option} scenario shows a moderate impact of {diff_pct:.1f}% on refinery run rates.
            
            **Recommended Actions:**
            - Continue monitoring actual maintenance and capacity changes
            - Make minor adjustments to trading positions if needed
            - No major market impacts expected
            """
        
        st.markdown(recommendation)

# Run the app
if __name__ == '__main__':
    main()