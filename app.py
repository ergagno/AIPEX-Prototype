import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time
import io
import folium
from streamlit_folium import folium_static

# Custom CSS with layout adjustments (background removed)
st.markdown(
    f"""
    <style>
    .main .block-container {{
        background: rgba(236, 240, 241, 0.9); /* Semi-transparent overlay for readability */
        color: #2c3e50;
        padding: 20px;
    }}
    .sidebar .sidebar-content {{
        background-color: #2c3e50; /* Dark blue sidebar */
        color: #ecf0f1; /* Light text */
    }}
    .sidebar .sidebar-content .st-expander {{
        background-color: #34495e; /* Slightly lighter blue */
        color: #ecf0f1;
    }}
    .sidebar .sidebar-content .st-expander button {{
        color: #ecf0f1;
    }}
    h1 {{
        color: #2980b9; /* Professional blue for title */
        font-family: Arial, sans-serif;
        display: inline; /* Align title with logo */
        margin-right: 10px; /* Space between title and logo */
    }}
    .chart-container {{
        margin-bottom: 20px; /* Consistent spacing below charts */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the Excel file
@st.cache_data
def load_data():
    return pd.read_excel("AIPEX PROJECT LIST -WORKINGv13.xlsx")  # Your file name

# Loading state
with st.spinner("Loading data..."):
    time.sleep(1)  # Simulate loading time
    data = load_data()

# Ensure numeric columns are properly formatted
base_numeric_columns = ["StandardCost", "ProjectedCost", "Actual Cost", "Project Duration (days)", "Project Readiness Ranking"]
for col in base_numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Define expected monthly columns
monthly_columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
# Check which monthly columns actually exist
available_monthly_columns = [col for col in monthly_columns if col in data.columns]
if not available_monthly_columns:
    st.warning("No monthly columns (e.g., 'January', 'February', etc.) found in the dataset. Please verify column names in your Excel file.")
else:
    for col in available_monthly_columns:
        data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Title and Logo
col1, col2 = st.columns([3, 4])  # Adjust column widths as needed
with col1:
    st.image("logo.png", width=300)  # Display logo, adjust width as needed
with col2:
    st.title("AIPEX")
st.write("A strategic tool for managing and visualizing oil and gas project data.")

# Initialize session state for assessment parameters with default values
if 'base_assessment' not in st.session_state:
    st.session_state.base_assessment = 90  # Start high, as higher is better
if 'driver_safety_high_deduction' not in st.session_state:
    st.session_state.driver_safety_high_deduction = 30  # Deduction for high safety concerns
if 'other_driver_high_deduction' not in st.session_state:
    st.session_state.other_driver_high_deduction = 20  # Deduction for other high concerns
if 'driver_medium_deduction' not in st.session_state:
    st.session_state.driver_medium_deduction = 10  # Deduction for medium concerns
if 'cost_overrun_deduction' not in st.session_state:
    st.session_state.cost_overrun_deduction = 15  # Deduction for cost overrun
if 'priority_threshold' not in st.session_state:
    st.session_state.priority_threshold = 7  # Same threshold
if 'priority_deduction' not in st.session_state:
    st.session_state.priority_deduction = 10  # Deduction for high priority
if 'readiness_threshold' not in st.session_state:
    st.session_state.readiness_threshold = 50  # Same threshold
if 'readiness_bonus' not in st.session_state:
    st.session_state.readiness_bonus = 20  # Bonus for high readiness
if 'alert_threshold' not in st.session_state:
    st.session_state.alert_threshold = 50  # Threshold for low scores (now low is bad)

# Initialize session state for probability parameters
if 'base_probability' not in st.session_state:
    st.session_state.base_probability = 0.9  # Starting probability (90%)
if 'cost_overrun_sensitivity' not in st.session_state:
    st.session_state.cost_overrun_sensitivity = 1.0  # Multiplier for cost overrun impact
if 'budget_surplus_boost' not in st.session_state:
    st.session_state.budget_surplus_boost = 1.0  # Multiplier for surplus effect
if 'min_probability_cap' not in st.session_state:
    st.session_state.min_probability_cap = 0.1  # Minimum probability cap (10%)
if 'assessment_score_weight' not in st.session_state:
    st.session_state.assessment_score_weight = 1.0  # Weight for assessment score impact

# Initialize session state for popup visibility and table state
if 'show_assessment_popup' not in st.session_state:
    st.session_state.show_assessment_popup = False
if 'show_probability_popup' not in st.session_state:
    st.session_state.show_probability_popup = False
if 'show_table' not in st.session_state:
    st.session_state.show_table = False  # Start with table collapsed

# --- Filtering Section ---
with st.sidebar.expander("Project Filters", expanded=True):
    # Filter 1: Planned Year
    year_options = sorted(data["Planned Year"].unique().tolist())
    selected_years = st.multiselect("Select Planned Years", options=year_options, default=[])

    # Filter 2: Program Classification
    filtered_data = data.copy()
    if selected_years:
        filtered_data = filtered_data[filtered_data["Planned Year"].isin(selected_years)]
    classification_column = "PROGRAM CLASSIFICATION"
    if classification_column in filtered_data.columns:
        classification_options = sorted(filtered_data[classification_column].dropna().unique().tolist())
    else:
        classification_options = []
        st.write(f"Warning: Column '{classification_column}' not found.")
    selected_classifications = st.multiselect("Select Program Classifications", options=classification_options, default=[])

    # Filter 3: Program Names
    if selected_classifications and classification_column in filtered_data.columns:
        filtered_data = filtered_data[filtered_data[classification_column].isin(selected_classifications)]
    program_options = sorted(filtered_data["Program NAME"].dropna().unique().tolist())
    selected_programs = st.multiselect("Select Program Names", options=program_options, default=[])

    # Filter 4: Predicted Repair Type (Single Selection)
    if selected_programs:
        filtered_data = filtered_data[filtered_data["Program NAME"].isin(selected_programs)]
    predicted_repair_type_options = sorted(filtered_data["Predicted Repair Type"].dropna().unique().tolist()) if "Predicted Repair Type" in filtered_data.columns else []
    if not predicted_repair_type_options:
        st.warning("No 'Predicted Repair Type' data available in the dataset.")
    selected_predicted_repair_type = st.selectbox("Select Predicted Repair Type", options=["Choose an option"] + predicted_repair_type_options, index=0)

    # Reset Button
    if st.button("Reset Filters"):
        selected_years = []
        selected_classifications = []
        selected_programs = []
        selected_predicted_repair_type = "Choose an option"
        st.rerun()

# Apply filters
filtered_data = data.copy()
if selected_years:
    filtered_data = filtered_data[filtered_data["Planned Year"].isin(selected_years)]
if selected_classifications and "PROGRAM CLASSIFICATION" in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["PROGRAM CLASSIFICATION"].isin(selected_classifications)]
if selected_programs:
    filtered_data = filtered_data[filtered_data["Program NAME"].isin(selected_programs)]
if selected_predicted_repair_type != "Choose an option" and "Predicted Repair Type" in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["Predicted Repair Type"] == selected_predicted_repair_type]

# Display filtered data with collapse/expand button
st.subheader("Filtered Project Data")
if st.button("Show/Hide Table"):
    st.session_state.show_table = not st.session_state.show_table
if st.session_state.show_table:
    st.dataframe(filtered_data)

# Key Metrics
st.subheader("Key Metrics")
col_metrics1, col_metrics2 = st.columns([1, 1])
with col_metrics1:
    total_projected_cost = filtered_data["ProjectedCost"].sum()  # Full dollar amount
    st.metric(label="Total Projected Cost", value=f"${total_projected_cost:,.0f}")
with col_metrics2:
    readiness_percentage = filtered_data["Project Readiness Ranking"].mean() if not filtered_data.empty else 0
    st.metric(label="Average Project Readiness", value=f"{readiness_percentage:.1f}")

# Additional Metrics under Key Metrics
col_add_metrics1, col_add_metrics2 = st.columns([1, 1])
with col_add_metrics1:
    total_projects = len(filtered_data)
    fully_scoped_projects = len(filtered_data[filtered_data["Fully Scoped"] == "Yes"])
    percent_fully_scoped = (fully_scoped_projects / total_projects * 100) if total_projects > 0 else 0
    st.metric(label="% Portfolio Fully Scoped", value=f"{percent_fully_scoped:.1f}%")
with col_add_metrics2:
    total_projected_cost = filtered_data["ProjectedCost"].sum()
    total_standard_cost = filtered_data["StandardCost"].sum()
    total_actual_cost = filtered_data["Actual Cost"].sum()
    cost_table_data = pd.DataFrame({
        "Cost Type": ["Total Projected Cost", "Total Standard Cost", "Total Actual Cost"],
        "Amount": [f"${total_projected_cost:,.0f}", f"${total_standard_cost:,.0f}", f"${total_actual_cost:,.0f}"]
    })
    st.table(cost_table_data)

# --- Charts Section ---
st.subheader("Visualizations")

# Chart 1: Program Count by Scope
st.subheader("Program Count by Scope")
if len(selected_programs) == 1:
    chart_data = filtered_data[filtered_data["Program NAME"] == selected_programs[0]]
    if "Predicted Repair Type" in chart_data.columns and not chart_data["Predicted Repair Type"].isna().all():
        priority_chart = px.histogram(
            chart_data, 
            x="FEATURE TYPE", 
            color="Fully Scoped", 
            facet_col="Predicted Repair Type" if selected_predicted_repair_type != "Choose an option" else None,
            title="",
            barmode="group"
        )
    else:
        priority_chart = px.histogram(
            chart_data, 
            x="FEATURE TYPE", 
            color="Fully Scoped", 
            title="",
            barmode="group"
        )
    # Display unique Predicted Repair Types under the chart if a single program is selected
    if len(selected_programs) == 1 and selected_predicted_repair_type != "Choose an option" and "Predicted Repair Type" in chart_data.columns:
        unique_repair_types = chart_data["Predicted Repair Type"].dropna().unique().tolist()
        st.write("**Predicted Repair Types in this Program:**", ", ".join(unique_repair_types))
else:
    if len(selected_years) > 1:
        chart_data = filtered_data.groupby(['Program NAME', 'Fully Scoped', 'Planned Year', 'Predicted Repair Type']).size().reset_index(name='Count')
        if "Predicted Repair Type" in chart_data.columns and not chart_data["Predicted Repair Type"].isna().all():
            priority_chart = px.bar(
                chart_data, 
                x="Program NAME", 
                y="Count", 
                color="Fully Scoped", 
                facet_col="Planned Year", 
                facet_row="Predicted Repair Type" if selected_predicted_repair_type != "Choose an option" else None,
                title="",
                barmode="group",
                category_orders={"Planned Year": sorted(selected_years)}
            )
        else:
            priority_chart = px.bar(
                chart_data, 
                x="Program NAME", 
                y="Count", 
                color="Fully Scoped", 
                facet_col="Planned Year", 
                title="",
                barmode="group",
                category_orders={"Planned Year": sorted(selected_years)}
            )
    else:
        chart_data = filtered_data.groupby(['Program NAME', 'Fully Scoped', 'Predicted Repair Type']).size().reset_index(name='Count')
        if "Predicted Repair Type" in chart_data.columns and not chart_data["Predicted Repair Type"].isna().all():
            priority_chart = px.histogram(
                chart_data, 
                x="Program NAME", 
                y="Count",
                color="Fully Scoped", 
                facet_col="Predicted Repair Type" if selected_predicted_repair_type != "Choose an option" else None,
                title="",
                barmode="group"
            )
        else:
            priority_chart = px.histogram(
                chart_data, 
                x="Program NAME", 
                y="Count",
                color="Fully Scoped", 
                title="",
                barmode="group"
            )
priority_chart.update_layout(
    yaxis_title="Count", 
    yaxis_title_font_color="#2c3e50",
    xaxis_title="Program Name" if len(selected_programs) != 1 else "Feature Type",
    xaxis_title_font_color="#2c3e50",
    height=400,
    width=600,
    paper_bgcolor="rgba(255, 255, 255, 0.5)",
    plot_bgcolor="rgba(255, 255, 255, 0.5)",
    legend_title="Fully Scoped"
)
priority_chart.update_traces(textposition="auto")
st.plotly_chart(priority_chart, use_container_width=True)

# Chart 2: Total Cost by Program
st.subheader("Total Cost by Program")
if len(selected_programs) == 1:
    chart_data = filtered_data[filtered_data["Program NAME"] == selected_programs[0]]
    for col in ["StandardCost", "ProjectedCost", "Actual Cost"]:
        chart_data[col] = pd.to_numeric(chart_data[col], errors='coerce').fillna(0)
    if "Predicted Repair Type" in chart_data.columns and not chart_data["Predicted Repair Type"].isna().all():
        cost_data = chart_data[["FEATURE TYPE", "StandardCost", "ProjectedCost", "Actual Cost", "Predicted Repair Type"]].melt(
            id_vars=["FEATURE TYPE", "Predicted Repair Type"], 
            value_vars=["StandardCost", "ProjectedCost", "Actual Cost"], 
            var_name="Cost Type", 
            value_name="Total Cost"
        )
        cost_data = cost_data.groupby(["FEATURE TYPE", "Cost Type", "Predicted Repair Type"])["Total Cost"].sum().reset_index()
        cost_data = cost_data[cost_data["Total Cost"] > 0]
        # Calculate total cost per FEATURE TYPE for conditional text positioning
        total_cost_per_feature = cost_data.groupby("FEATURE TYPE")["Total Cost"].sum().reset_index()
        cost_chart = px.bar(
            cost_data, 
            x="FEATURE TYPE", 
            y="Total Cost", 
            color="Cost Type", 
            facet_col="Predicted Repair Type" if selected_predicted_repair_type != "Choose an option" else None,
            title="",
            barmode="group", 
            text=cost_data["Total Cost"].apply(lambda x: f'${x:,.0f}' if x > 0 else "")
        )
    else:
        cost_data = chart_data[["FEATURE TYPE", "StandardCost", "ProjectedCost", "Actual Cost"]].melt(
            id_vars=["FEATURE TYPE"], 
            value_vars=["StandardCost", "ProjectedCost", "Actual Cost"], 
            var_name="Cost Type", 
            value_name="Total Cost"
        )
        cost_data = cost_data.groupby(["FEATURE TYPE", "Cost Type"])["Total Cost"].sum().reset_index()
        cost_data = cost_data[cost_data["Total Cost"] > 0]
        # Calculate total cost per FEATURE TYPE for conditional text positioning
        total_cost_per_feature = cost_data.groupby("FEATURE TYPE")["Total Cost"].sum().reset_index()
        cost_chart = px.bar(
            cost_data, 
            x="FEATURE TYPE", 
            y="Total Cost", 
            color="Cost Type", 
            title="",
            barmode="group", 
            text=cost_data["Total Cost"].apply(lambda x: f'${x:,.0f}' if x > 0 else "")
        )
else:
    if len(selected_years) > 1:
        for col in ["StandardCost", "ProjectedCost", "Actual Cost"]:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce').fillna(0)
        if "Predicted Repair Type" in filtered_data.columns and not filtered_data["Predicted Repair Type"].isna().all():
            cost_data = filtered_data[["Program NAME", "StandardCost", "ProjectedCost", "Actual Cost", "Planned Year", "Predicted Repair Type"]].melt(
                id_vars=["Program NAME", "Planned Year", "Predicted Repair Type"], 
                value_vars=["StandardCost", "ProjectedCost", "Actual Cost"], 
                var_name="Cost Type", 
                value_name="Total Cost"
            )
            cost_data = cost_data.groupby(["Program NAME", "Cost Type", "Planned Year", "Predicted Repair Type"])["Total Cost"].sum().reset_index()
            cost_data = cost_data[cost_data["Total Cost"] > 0]
            # Calculate total cost per Program NAME for conditional text positioning
            total_cost_per_program = cost_data.groupby("Program NAME")["Total Cost"].sum().reset_index()
            cost_chart = px.bar(
                cost_data, 
                x="Program NAME", 
                y="Total Cost", 
                color="Cost Type", 
                facet_col="Planned Year", 
                facet_row="Predicted Repair Type" if selected_predicted_repair_type != "Choose an option" else None,
                title="",
                barmode="group",
                text=cost_data["Total Cost"].apply(lambda x: f'${x:,.0f}' if x > 0 else "")
            )
        else:
            cost_data = filtered_data[["Program NAME", "StandardCost", "ProjectedCost", "Actual Cost", "Planned Year"]].melt(
                id_vars=["Program NAME", "Planned Year"], 
                value_vars=["StandardCost", "ProjectedCost", "Actual Cost"], 
                var_name="Cost Type", 
                value_name="Total Cost"
            )
            cost_data = cost_data.groupby(["Program NAME", "Cost Type", "Planned Year"])["Total Cost"].sum().reset_index()
            cost_data = cost_data[cost_data["Total Cost"] > 0]
            # Calculate total cost per Program NAME for conditional text positioning
            total_cost_per_program = cost_data.groupby("Program NAME")["Total Cost"].sum().reset_index()
            cost_chart = px.bar(
                cost_data, 
                x="Program NAME", 
                y="Total Cost", 
                color="Cost Type", 
                facet_col="Planned Year", 
                title="",
                barmode="group",
                text=cost_data["Total Cost"].apply(lambda x: f'${x:,.0f}' if x > 0 else "")
            )
    else:
        for col in ["StandardCost", "ProjectedCost", "Actual Cost"]:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce').fillna(0)
        if "Predicted Repair Type" in filtered_data.columns and not filtered_data["Predicted Repair Type"].isna().all():
            cost_data = filtered_data[["Program NAME", "StandardCost", "ProjectedCost", "Actual Cost", "Predicted Repair Type"]].melt(
                id_vars=["Program NAME", "Predicted Repair Type"], 
                value_vars=["StandardCost", "ProjectedCost", "Actual Cost"], 
                var_name="Cost Type", 
                value_name="Total Cost"
            )
            cost_data = cost_data.groupby(["Program NAME", "Cost Type", "Predicted Repair Type"])["Total Cost"].sum().reset_index()
            cost_data = cost_data[cost_data["Total Cost"] > 0]
            # Calculate total cost per Program NAME for conditional text positioning
            total_cost_per_program = cost_data.groupby("Program NAME")["Total Cost"].sum().reset_index()
            cost_chart = px.bar(
                cost_data, 
                x="Program NAME", 
                y="Total Cost", 
                color="Cost Type", 
                facet_col="Predicted Repair Type" if selected_predicted_repair_type != "Choose an option" else None,
                title="",
                barmode="group", 
                text=cost_data["Total Cost"].apply(lambda x: f'${x:,.0f}' if x > 0 else "")
            )
        else:
            cost_data = filtered_data[["Program NAME", "StandardCost", "ProjectedCost", "Actual Cost"]].melt(
                id_vars=["Program NAME"], 
                value_vars=["StandardCost", "ProjectedCost", "Actual Cost"], 
                var_name="Cost Type", 
                value_name="Total Cost"
            )
            cost_data = cost_data.groupby(["Program NAME", "Cost Type"])["Total Cost"].sum().reset_index()
            cost_data = cost_data[cost_data["Total Cost"] > 0]
            # Calculate total cost per Program NAME for conditional text positioning
            total_cost_per_program = cost_data.groupby("Program NAME")["Total Cost"].sum().reset_index()
            cost_chart = px.bar(
                cost_data, 
                x="Program NAME", 
                y="Total Cost", 
                color="Cost Type", 
                title="",
                barmode="group", 
                text=cost_data["Total Cost"].apply(lambda x: f'${x:,.0f}' if x > 0 else "")
            )
# Merge with total cost data for conditional text positioning
if len(selected_programs) == 1:
    cost_chart.data = [trace for trace in cost_chart.data if trace.x[0] in total_cost_per_feature["FEATURE TYPE"].values]
    for trace in cost_chart.data:
        feature = trace.x[0]
        total_cost = total_cost_per_feature[total_cost_per_feature["FEATURE TYPE"] == feature]["Total Cost"].iloc[0]
        if total_cost < 50000000:  # Threshold for short bars
            trace.textposition = "outside"
        else:
            trace.textposition = "inside"
else:
    cost_chart.data = [trace for trace in cost_chart.data if trace.x[0] in total_cost_per_program["Program NAME"].values]
    for trace in cost_chart.data:
        program = trace.x[0]
        total_cost = total_cost_per_program[total_cost_per_program["Program NAME"] == program]["Total Cost"].iloc[0]
        if total_cost < 50000000:  # Threshold for short bars
            trace.textposition = "outside"
        else:
            trace.textposition = "inside"
cost_chart.update_layout(
    yaxis_title="Total Cost ($)",
    yaxis_title_font_color="#2c3e50",
    xaxis_title="Feature Type" if len(selected_programs) == 1 else "Program Name",
    xaxis_title_font_color="#2c3e50",
    yaxis_tickformat="$,.0f",
    yaxis_range=[0, cost_data["Total Cost"].max() * 1.2],
    height=400,
    width=600,
    showlegend=True,
    paper_bgcolor="rgba(255, 255, 255, 0.5)",
    plot_bgcolor="rgba(255, 255, 255, 0.5)",
    bargap=0.2  # Add small gap between bars for better readability
)
cost_chart.update_traces(
    textangle=-90,  # Rotate text to be vertical
    textfont=dict(size=10)  # Adjust font size for readability
)
st.plotly_chart(cost_chart, use_container_width=True)

# Chart 3: Spend Profile
st.subheader("Spend Profile")
if selected_years:
    if len(selected_years) > 1:
        selected_year = st.sidebar.selectbox("Select Single Planned Year for Detailed View (optional)", options=selected_years + [None], index=len(selected_years))
    else:
        selected_year = selected_years[0]
else:
    selected_year = year_options[0] if year_options else None

if selected_year:
    spend_data = filtered_data[filtered_data["Planned Year"] == selected_year].copy()
else:
    spend_data = filtered_data.copy()

monthly_columns = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

spend_profile_data = []
if not spend_data.empty and any(col in spend_data.columns for col in monthly_columns):
    for year in selected_years if selected_years else [selected_year] if selected_year else []:
        year_data = filtered_data[filtered_data["Planned Year"] == year].copy()
        year_spend = {}
        for month in monthly_columns:
            if month in year_data.columns:
                year_spend[month] = year_data[month].sum()
        for month, short_month in zip(monthly_columns, months):
            spend_profile_data.append({'Year': year, 'Month': short_month, 'Spend Amount': year_spend.get(month, 0)})
    
    spend_profile_df = pd.DataFrame(spend_profile_data)

    if spend_profile_df['Spend Amount'].sum() > 0:
        spend_chart = px.line(spend_profile_df,
                            x="Month",
                            y="Spend Amount",
                            color="Year",
                            title="",
                            labels={"Spend Amount": "Spend Amount ($)", "Year": "Planned Year"})
        spend_chart.update_traces(mode="lines+markers")
        spend_chart.update_layout(
            yaxis_title="Spend Amount ($)",
            yaxis_title_font_color="#2c3e50",
            xaxis_title="Month",
            xaxis_title_font_color="#2c3e50",
            xaxis={'tickmode': 'array', 'tickvals': months},
            yaxis_tickformat="$,.0f",
            yaxis_range=[0, spend_profile_df["Spend Amount"].max() * 1.1 if spend_profile_df["Spend Amount"].max() > 0 else 1000],
            height=400,
            width=600,
            paper_bgcolor="rgba(255, 255, 255, 0.5)",
            plot_bgcolor="rgba(255, 255, 255, 0.5)",
            legend=dict(title="Years", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(spend_chart, use_container_width=True)
    else:
        st.warning(f"No spend data found in monthly columns for the selected year(s). Please ensure the columns contain non-zero values.")
else:
    st.warning(f"No valid monthly spend data available for the selected year(s). Please check if all monthly columns exist and contain data.")

# Chart 4: Driver Distribution by Rating
st.subheader("Driver Distribution by Rating")
driver_columns = ["Driver Safety", "Driver Operational", "Driver Regulatory"]
driver_data = filtered_data[driver_columns].melt(var_name="Driver Type", value_name="Rating")

# Rename "Driver Operational" to "Reliability"
driver_data["Driver Type"] = driver_data["Driver Type"].replace("Driver Operational", "Driver Reliability")

driver_data = driver_data.groupby(["Driver Type", "Rating"]).size().reset_index(name="Count")
driver_data = driver_data[driver_data["Rating"].notna()]

driver_chart = px.bar(driver_data, 
                      x="Driver Type", 
                      y="Count", 
                      color="Rating", 
                      title="",
                      text=driver_data["Count"].astype(str),
                      barmode="stack")
driver_chart.update_layout(
    yaxis_title="Number of Projects",
    yaxis_title_font_color="#2c3e50",
    xaxis_title="Driver Type",
    xaxis_title_font_color="#2c3e50",
    height=400,
    width=600,
    showlegend=True,
    paper_bgcolor="rgba(255, 255, 255, 0.5)",
    plot_bgcolor="rgba(255, 255, 255, 0.5)"
)
driver_chart.update_traces(textposition="auto")
st.plotly_chart(driver_chart, use_container_width=True)

# Chart 5: Project Readiness Distribution
st.subheader("Project Readiness Distribution")
if "Project Readiness Ranking" not in filtered_data.columns or filtered_data["Project Readiness Ranking"].isna().all() or (filtered_data["Project Readiness Ranking"] == 0).all():
    st.warning("No valid Project Readiness Ranking data available.")
else:
    readiness_counts = filtered_data["Project Readiness Ranking"].value_counts().reset_index()
    readiness_counts.columns = ["Project Readiness Ranking", "Count"]
    
    readiness_chart = px.bar(readiness_counts, 
                             x="Project Readiness Ranking", 
                             y="Count", 
                             color="Project Readiness Ranking",
                             color_continuous_scale=px.colors.sequential.Viridis,
                             text=readiness_counts["Count"].astype(str),
                             title="")
    readiness_chart.update_layout(
        yaxis_title="Number of Projects",
        yaxis_title_font_color="#2c3e50",
        xaxis_title="Project Readiness Ranking",
        xaxis_title_font_color="#2c3e50",
        height=400,
        width=600,
        paper_bgcolor="rgba(255, 255, 255, 0.5)",
        plot_bgcolor="rgba(255, 255, 255, 0.5)"
    )
    readiness_chart.update_traces(textposition="auto")
    st.plotly_chart(readiness_chart, use_container_width=True)

# Chart 6: Priority Ranking Distribution
st.subheader("Priority Ranking Distribution")
priority_data = filtered_data["Priority Ranking "].value_counts().reset_index()
priority_data.columns = ["Priority Ranking", "Count"]
priority_chart = px.bar(priority_data, 
                        x="Priority Ranking", 
                        y="Count", 
                        title="",
                        text=priority_data["Count"].astype(str),
                        color="Priority Ranking",
                        color_continuous_scale=px.colors.sequential.Viridis)
priority_chart.update_layout(
    yaxis_title="Number of Projects",
    yaxis_title_font_color="#2c3e50",
    xaxis_title="Priority Ranking",
    xaxis_title_font_color="#2c3e50",
    height=400,
    width=600,
    showlegend=False,
    paper_bgcolor="rgba(255, 255, 255, 0.5)",
    plot_bgcolor="rgba(255, 255, 255, 0.5)"
)
priority_chart.update_traces(textposition="auto")
st.plotly_chart(priority_chart, use_container_width=True)

# --- Project Completion Assessment Section ---
st.subheader("Project Completion Assessment (Powered by AI)")

# Button to trigger the popup (moved to this section)
if st.button("Configure Assessment Parameters"):
    st.session_state.show_assessment_popup = True

# Display the popup if triggered
if st.session_state.show_assessment_popup:
    with st.form("assessment_parameters_form"):
        st.subheader("Assessment Analyzer Parameters")
        
        # Category: Base Assessment
        st.subheader("Base Assessment")
        base_assessment = st.number_input("Base Assessment Score", min_value=0, max_value=100, value=st.session_state.base_assessment, step=1)
        st.write("The starting score for all projects (0-100). Higher values mean better completion likelihood.")
        
        # Category: Driver Deductions
        st.subheader("Driver Deductions")
        driver_safety_high_deduction = st.number_input("Driver Safety 'High' Deduction", min_value=0, max_value=100, value=st.session_state.driver_safety_high_deduction, step=1)
        st.write("Points deducted if Driver Safety is rated 'High', indicating a significant concern.")
        other_driver_high_deduction = st.number_input("Other Driver 'High' Deduction", min_value=0, max_value=100, value=st.session_state.other_driver_high_deduction, step=1)
        st.write("Points deducted if other drivers (Regulatory, Compliance) are rated 'High'.")
        driver_medium_deduction = st.number_input("Driver 'Medium' Deduction", min_value=0, max_value=100, value=st.session_state.driver_medium_deduction, step=1)
        st.write("Points deducted if any driver is rated 'Medium'.")
        
        # Category: Cost and Priority Deductions
        st.subheader("Cost and Priority Deductions")
        cost_overrun_deduction = st.number_input("Cost Overrun Deduction (>10%)", min_value=0, max_value=100, value=st.session_state.cost_overrun_deduction, step=1)
        st.write("Points deducted if Projected Cost exceeds Standard Cost by more than 10%.")
        priority_threshold = st.number_input("Priority Ranking Threshold", min_value=1, max_value=10, value=st.session_state.priority_threshold, step=1)
        st.write("Threshold above which a Priority Ranking triggers a deduction.")
        priority_deduction = st.number_input("Priority Ranking Deduction", min_value=0, max_value=100, value=st.session_state.priority_deduction, step=1)
        st.write("Points deducted if Priority Ranking exceeds the threshold.")
        
        # Category: Readiness Bonuses
        st.subheader("Readiness Bonuses")
        readiness_threshold = st.number_input("Project Readiness Threshold (%)", min_value=0, max_value=100, value=st.session_state.readiness_threshold, step=1)
        st.write("Threshold above which Project Readiness Ranking earns a bonus.")
        readiness_bonus = st.number_input("Project Readiness Bonus", min_value=0, max_value=100, value=st.session_state.readiness_bonus, step=1)
        st.write("Points added if Project Readiness Ranking meets or exceeds the threshold.")
        
        # Category: Alert Threshold
        st.subheader("Alert Threshold")
        alert_threshold = st.number_input("Alert Assessment Threshold", min_value=0, max_value=100, value=st.session_state.alert_threshold, step=1)
        st.write("Threshold below which an alert is triggered for low assessment scores (low scores indicate poor completion likelihood).")

        # Submit and Close buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Save Parameters"):
                # Update session state with new values
                st.session_state.base_assessment = base_assessment
                st.session_state.driver_safety_high_deduction = driver_safety_high_deduction
                st.session_state.other_driver_high_deduction = other_driver_high_deduction
                st.session_state.driver_medium_deduction = driver_medium_deduction
                st.session_state.cost_overrun_deduction = cost_overrun_deduction
                st.session_state.priority_threshold = priority_threshold
                st.session_state.priority_deduction = priority_deduction
                st.session_state.readiness_threshold = readiness_threshold
                st.session_state.readiness_bonus = readiness_bonus
                st.session_state.alert_threshold = alert_threshold
                st.session_state.show_assessment_popup = False
                st.success("Parameters saved successfully!")
        with col2:
            if st.form_submit_button("Close"):
                st.session_state.show_assessment_popup = False

# Retrieve parameters from session state for use in calculations
base_assessment = st.session_state.base_assessment
driver_safety_high_deduction = st.session_state.driver_safety_high_deduction
other_driver_high_deduction = st.session_state.other_driver_high_deduction
driver_medium_deduction = st.session_state.driver_medium_deduction
cost_overrun_deduction = st.session_state.cost_overrun_deduction
priority_threshold = st.session_state.priority_threshold
priority_deduction = st.session_state.priority_deduction
readiness_threshold = st.session_state.readiness_threshold
readiness_bonus = st.session_state.readiness_bonus
alert_threshold = st.session_state.alert_threshold

# Compute assessment scores for each program (higher is better)
def compute_assessment_score(row, base_assessment, driver_safety_high_deduction, other_driver_high_deduction, driver_medium_deduction,
                             cost_overrun_deduction, priority_threshold, priority_deduction, readiness_threshold, readiness_bonus):
    assessment_score = base_assessment  # Start with a high base score

    driver_columns = ["Driver Safety", "Driver Regulatory", "Driver Compliance"]
    for driver in driver_columns:
        if row[driver] == "High":
            if driver == "Driver Safety":
                assessment_score -= driver_safety_high_deduction  # Deduct for high safety concerns
            else:
                assessment_score -= other_driver_high_deduction  # Deduct for other high concerns
        elif row[driver] == "Medium":
            assessment_score -= driver_medium_deduction  # Deduct for medium concerns

    if row["StandardCost"] > 0 and row["ProjectedCost"] / row["StandardCost"] > 1.1:
        assessment_score -= cost_overrun_deduction  # Deduct for cost overrun

    if row["Priority Ranking "] > priority_threshold:
        assessment_score -= priority_deduction  # Deduct for high priority

    if row["Project Readiness Ranking"] >= readiness_threshold:
        assessment_score += readiness_bonus  # Add bonus for high readiness

    return max(min(assessment_score, 100), 0)  # Cap between 0 and 100

# Apply assessment scoring to filtered data with user-defined parameters
filtered_data["Assessment Score"] = filtered_data.apply(
    lambda row: compute_assessment_score(row, base_assessment, driver_safety_high_deduction, other_driver_high_deduction, driver_medium_deduction,
                                         cost_overrun_deduction, priority_threshold, priority_deduction, readiness_threshold, readiness_bonus),
    axis=1
)

# Aggregate assessment scores by Program NAME
assessment_data = filtered_data.groupby("Program NAME")["Assessment Score"].mean().reset_index()

# Display average assessment score as a metric
average_assessment = assessment_data["Assessment Score"].mean() if not assessment_data.empty else 0
st.metric(label="Average Assessment Score", value=f"{average_assessment:.1f}/100", delta=None)

# Add warning if any program falls below user-defined assessment threshold
if not assessment_data.empty and assessment_data["Assessment Score"].min() < alert_threshold:
    st.warning(f"Low Assessment Detected: Review programs with assessment scores below {alert_threshold}!")

# Plot assessment scores as a bar chart (higher is better, so use a reversed color scale)
assessment_chart = px.bar(assessment_data, 
                          x="Program NAME", 
                          y="Assessment Score", 
                          title="",
                          text=assessment_data["Assessment Score"].apply(lambda x: f'{x:.1f}'),
                          color="Assessment Score",
                          color_continuous_scale=px.colors.sequential.Greens)  # Green for high scores (good)
assessment_chart.update_layout(
    yaxis_title="Assessment Score (0-100)",
    yaxis_title_font_color="#2c3e50",
    xaxis_title="Program Name",
    xaxis_title_font_color="#2c3e50",
    yaxis_range=[0, 100],
    height=400,
    width=600,
    showlegend=False,
    paper_bgcolor="rgba(255, 255, 255, 0.5)",
    plot_bgcolor="rgba(255, 255, 255, 0.5)"
)
assessment_chart.update_traces(textposition="auto")
st.plotly_chart(assessment_chart, use_container_width=True)

# --- Budget Certainty Assessment Tool Section ---
if year_options:  # Show the section if there are years in the data
    st.subheader("Budget Certainty Assessment Tool")

    # Button to trigger the probability parameters popup
    if st.button("Configure Probability Parameters"):
        st.session_state.show_probability_popup = True

    # Display the popup if triggered
    if st.session_state.show_probability_popup:
        with st.form("probability_parameters_form"):
            st.subheader("Budget Probability Parameters")
            
            # Category: Base Probability
            st.subheader("Base Probability")
            base_probability = st.number_input(
                "Base Probability (%)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.base_probability * 100,
                step=1.0
            )
            st.write("The starting probability (0-100%) before adjustments for cost overrun and assessment score. Higher values increase the overall probability.")
            
            # Category: Cost Overrun Sensitivity
            st.subheader("Cost Overrun Sensitivity")
            cost_overrun_sensitivity = st.number_input(
                "Cost Overrun Sensitivity Factor",
                min_value=0.0,
                max_value=10.0,
                value=st.session_state.cost_overrun_sensitivity,
                step=0.1
            )
            st.write("Controls how much a cost overrun (Projected Cost / Standard Cost) reduces the probability. A value of 1 means the cost ratio directly scales the probability; higher values amplify the impact of cost overruns.")
            
            # Category: Budget Surplus Boost
            st.subheader("Budget Surplus Boost")
            budget_surplus_boost = st.number_input(
                "Budget Surplus Boost Factor",
                min_value=0.0,
                max_value=5.0,
                value=st.session_state.budget_surplus_boost,
                step=0.1
            )
            st.write("Amplifies the effect of a budget surplus (budget beyond projected cost) on the overall probability. A value of 1 means the surplus scales linearly; higher values increase the probability more with larger surpluses.")
            
            # Category: Minimum Probability Cap
            st.subheader("Minimum Probability Cap")
            min_probability_cap = st.number_input(
                "Minimum Probability Cap (%)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.min_probability_cap * 100,
                step=1.0
            )
            st.write("Sets the minimum probability (0-100%) for any individual project, preventing the probability from dropping too low. This ensures a baseline confidence level.")
            
            # Category: Assessment Score Weight
            st.subheader("Assessment Score Weight")
            assessment_score_weight = st.number_input(
                "Assessment Score Weight Factor",
                min_value=0.0,
                max_value=5.0,
                value=st.session_state.assessment_score_weight,
                step=0.1
            )
            st.write("Adjusts the influence of the Assessment Score (0-100) on the probability. A value of 1 means the score scales the probability directly (Score / 100); higher values increase the impact of a high assessment score.")
            
            # Category: Assessment Score Impact (Informational)
            st.subheader("Assessment Score Impact (Set in Project Completion Assessment)")
            st.write("The Assessment Score (0-100) from the Project Completion Assessment section is used to scale the probability. A higher score increases the probability (Assessment Score / 100, weighted by the Assessment Score Weight). Adjust this in the 'Configure Assessment Parameters' section above.")

            # Submit and Close buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Save Parameters"):
                    # Update session state with new values
                    st.session_state.base_probability = base_probability / 100  # Convert percentage to decimal
                    st.session_state.cost_overrun_sensitivity = cost_overrun_sensitivity
                    st.session_state.budget_surplus_boost = budget_surplus_boost
                    st.session_state.min_probability_cap = min_probability_cap / 100  # Convert percentage to decimal
                    st.session_state.assessment_score_weight = assessment_score_weight
                    st.session_state.show_probability_popup = False
                    st.success("Probability parameters saved successfully!")
            with col2:
                if st.form_submit_button("Close"):
                    st.session_state.show_probability_popup = False

    # Retrieve probability parameters
    base_probability = st.session_state.base_probability
    cost_overrun_sensitivity = st.session_state.cost_overrun_sensitivity
    budget_surplus_boost = st.session_state.budget_surplus_boost
    min_probability_cap = st.session_state.min_probability_cap
    assessment_score_weight = st.session_state.assessment_score_weight

    analysis_year = st.selectbox("Select Year for Budget Analysis", options=year_options, index=0 if year_options else None)
    
    # Use text_input to allow comma-formatted budget input
    default_budget = 40000000  # Default value as before
    budget_input = st.text_input(
        "Enter Annual Budget ($)", 
        value=f"{default_budget:,}",  # Display with commas
        help="Enter the annual budget in dollars. Use commas for thousands (e.g., 40,000,000)."
    )
    
    # Parse the input: Remove commas and convert to integer
    try:
        # Remove commas and convert to integer
        annual_budget = int(budget_input.replace(',', '').replace('$', '').strip())
        # Ensure the budget is not negative
        if annual_budget < 0:
            st.error("Annual budget cannot be negative. Please enter a value greater than or equal to 0.")
            annual_budget = 0
    except ValueError:
        st.error("Please enter a valid number for the annual budget (e.g., 40,000,000).")
        annual_budget = default_budget  # Fallback to default if input is invalid
    
    # Update the displayed value with commas for user feedback
    st.write(f"Annual Budget Entered: ${annual_budget:,}")
    
    budget_data = filtered_data[filtered_data["Planned Year"] == analysis_year].copy()
    
    total_projected_cost_year = budget_data["ProjectedCost"].sum()
    total_standard_cost_year = budget_data["StandardCost"].sum()
    
    def calculate_budget_probability(row, base_probability, cost_overrun_sensitivity, budget_surplus_boost, min_probability_cap, assessment_score_weight):
        cost_ratio = row["ProjectedCost"] / row["StandardCost"] if row["StandardCost"] > 0 else 1.0
        assessment_score = row["Assessment Score"]
        assessment_impact = (assessment_score / 100) ** assessment_score_weight  # Weighted assessment impact
        cost_impact = (1 / cost_ratio) ** cost_overrun_sensitivity  # Adjust cost impact with sensitivity
        probability = base_probability * cost_impact * assessment_impact
        return max(min_probability_cap, min(1, probability))  # Apply minimum cap and maximum of 1
    
    budget_data["Assessment Score"] = budget_data.apply(
        lambda row: compute_assessment_score(row, base_assessment, driver_safety_high_deduction, other_driver_high_deduction, driver_medium_deduction,
                                             cost_overrun_deduction, priority_threshold, priority_deduction, readiness_threshold, readiness_bonus),
        axis=1
    )
    budget_data["Budget Probability"] = budget_data.apply(
        lambda row: calculate_budget_probability(row, base_probability, cost_overrun_sensitivity, budget_surplus_boost, min_probability_cap, assessment_score_weight),
        axis=1
    )
    
    # Budget factor logic: Scale with surplus, boosted by budget_surplus_boost
    budget_factor = min(1.5, 1 + budget_surplus_boost * (annual_budget - total_projected_cost_year) / (2 * total_projected_cost_year)) if total_projected_cost_year > 0 else 1
    base_overall_probability = budget_data["Budget Probability"].mean() if not budget_data.empty else 0
    overall_probability = base_overall_probability * budget_factor
    overall_probability = max(0, min(1, overall_probability))
    
    def propose_action(row):
        cost_ratio = row["ProjectedCost"] / row["StandardCost"] if row["StandardCost"] > 0 else 1.0
        if row["Assessment Score"] < 30 and row["Budget Probability"] < 0.5:  # Low score is bad
            return "Pause"
        elif row["Assessment Score"] > 70 and row["Budget Probability"] > 0.8:  # High score is good
            return "Accelerate"
        elif row["Assessment Score"] < 50 and row["Budget Probability"] < 0.7:
            return "Slow Down"
        elif cost_ratio > 1.2 and row["Budget Probability"] < 0.6:
            return "Defer to Future Years"
        else:
            return "Continue"
    
    budget_data["Proposed Action"] = budget_data.apply(propose_action, axis=1)
    
    st.write(f"### Budget Analysis for {analysis_year}")
    st.write(f"Total Projected Cost: ${total_projected_cost_year:,.0f}")
    st.write(f"Total Standard Cost: ${total_standard_cost_year:,.0f}")
    st.write(f"Remaining Budget: ${max(0, annual_budget - total_projected_cost_year):,.0f}")
    
    st.metric(label="Overall Probability to Hit Budget", value=f"{overall_probability:.0%}")
    
    st.subheader("Project Recommendations")
    st.dataframe(budget_data[["Program NAME", "ProjectedCost", "StandardCost", "Assessment Score", "Budget Probability", "Proposed Action"]].style.format({
        "ProjectedCost": "${:,.0f}",
        "StandardCost": "${:,.0f}",
        "Assessment Score": "{:.1f}",
        "Budget Probability": "{:.0%}"
    }).set_properties(**{
        'background-color': '#f9f9f9',
        'border-color': '#dddddd',
        'font-family': 'Arial, sans-serif',
        'text-align': 'center'
    }).set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#2980b9'), ('color', 'white'), ('font-weight', 'bold')]
    }]))

# --- Project Locations Map ---
st.subheader("Project Locations Map")
# Count projects per location
location_counts = filtered_data["Location"].value_counts().reset_index()
location_counts.columns = ["Location", "Project_Count"]
total_projects = len(filtered_data)

# Predefined coordinates for Canadian cities (approximate)
canada_cities = {
    "Toronto": [43.651070, -79.347015],
    "Vancouver": [49.282729, -123.120738],
    "Montreal": [45.501689, -73.567256],
    "Quebec City": [46.813878, -71.207981],
    "Calgary": [51.044733, -114.071883],
    "Ottawa": [45.421106, -75.690306],
    "Edmonton": [53.546125, -113.493823],
    "Winnipeg": [49.895136, -97.138374],
    "Halifax": [44.648618, -63.585948],
    # Add more cities as needed based on your data
}

# Create a map centered on Canada
map_center = [60.000000, -95.000000]  # Approximate center of Canada
m = folium.Map(location=map_center, zoom_start=4, height=400, width=600)

# Add markers for each location with project count
for index, row in location_counts.iterrows():
    location = row["Location"]
    count = row["Project_Count"]
    if location in canada_cities:
        lat, lon = canada_cities[location]
        folium.CircleMarker(
            location=[lat, lon],
            radius=count / 10,  # Adjust radius based on count (scale factor)
            popup=f"{location}: {count} projects",
            fill=True,
            color="blue",
            fill_color="blue",
            fill_opacity=0.6
        ).add_to(m)

# Display the map in Streamlit
folium_static(m)

# --- Footer ---
st.write("### Summary")
st.write(f"Total Projects: {total_projects}")

# --- Run Instructions ---
# Save as app.py and run with: streamlit run app.py
# Ensure requirements.txt includes: streamlit, pandas, plotly, numpy, openpyxl, folium
