import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import time
import base64

# Custom CSS with background image
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

bg_image = "background.jpg"  # Ensure the image is saved as 'background.jpg'
bg_base64 = get_base64(bg_image)
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url(data:image/jpeg;base64,{bg_base64});
        background-size: cover;
    }}
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
    }}
    h2 {{
        color: #34495e;
        font-family: Arial, sans-serif;
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
for col in ["StandardCost", "ProjectedCost", "Actual Cost", "Project Duration (days)", "Project Readiness Percentage"]:
    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(0)

# Title and Description
st.title("AIPEX by Enabled Performance")
st.write("A professional tool for managing and visualizing oil and gas project data.")

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

    # Filter 4: Custom Filter
    if selected_programs:
        filtered_data = filtered_data[filtered_data["Program NAME"].isin(selected_programs)]

    filter_column = st.selectbox("Select Column to Filter", options=filtered_data.columns)
    filter_value_options = sorted(filtered_data[filter_column].dropna().unique().tolist())
    selected_filter_values = st.multiselect(f"Select {filter_column} Values", options=filter_value_options, default=[])

    # Filter 5: Priority Ranking (Max value, include all <= selected)
    # Compute min and max for Priority Ranking
    if not filtered_data["Priority Ranking "].empty:
        min_priority = int(filtered_data["Priority Ranking "].min())
        max_priority = int(filtered_data["Priority Ranking "].max())
        # Ensure min_value < max_value for the slider
        if min_priority == max_priority:
            min_priority = max(1, min_priority - 1)  # Ensure min is at least 1
            max_priority = min_priority + 2  # Add a small range to make the slider functional
        default_value = max_priority
    else:
        min_priority = 1
        max_priority = 10
        default_value = 10

    priority_max = st.slider("Max Priority Ranking (inclusive)", 
                             min_value=min_priority, 
                             max_value=max_priority, 
                             value=default_value)

    # Reset Button
    if st.button("Reset Filters"):
        selected_years = []
        selected_classifications = []
        selected_programs = []
        selected_filter_values = []
        st.experimental_rerun()

# --- Risk Analyzer Parameters Section ---
with st.sidebar.expander("Risk Analyzer Parameters", expanded=True):
    st.write("Adjust the weights and thresholds for risk scoring:")
    
    base_risk = st.number_input("Base Risk Score", min_value=0, max_value=100, value=10, step=1)
    
    driver_safety_high_weight = st.number_input("Driver Safety 'High' Weight", min_value=0, max_value=100, value=30, step=1)
    other_driver_high_weight = st.number_input("Other Driver 'High' Weight", min_value=0, max_value=100, value=20, step=1)
    driver_medium_weight = st.number_input("Driver 'Medium' Weight", min_value=0, max_value=100, value=10, step=1)
    
    cost_overrun_penalty = st.number_input("Cost Overrun Penalty (>10%)", min_value=0, max_value=100, value=15, step=1)
    
    priority_threshold = st.number_input("Priority Ranking Threshold", min_value=1, max_value=10, value=7, step=1)
    priority_penalty = st.number_input("Priority Ranking Penalty", min_value=0, max_value=100, value=10, step=1)
    
    readiness_threshold = st.number_input("Project Readiness Threshold (%)", min_value=0, max_value=100, value=50, step=1)
    readiness_penalty = st.number_input("Project Readiness Penalty", min_value=0, max_value=100, value=20, step=1)
    
    alert_threshold = st.number_input("Alert Risk Threshold", min_value=0, max_value=100, value=50, step=1)

# Apply filters
filtered_data = data.copy()
if selected_years:
    filtered_data = filtered_data[filtered_data["Planned Year"].isin(selected_years)]
if selected_classifications and classification_column in filtered_data.columns:
    filtered_data = filtered_data[filtered_data[classification_column].isin(selected_classifications)]
if selected_programs:
    filtered_data = filtered_data[filtered_data["Program NAME"].isin(selected_programs)]
if selected_filter_values:
    filtered_data = filtered_data[filtered_data[filter_column].isin(selected_filter_values)]
if "Priority Ranking " in filtered_data.columns:
    filtered_data = filtered_data[filtered_data["Priority Ranking "] <= priority_max]

# Display filtered data
st.subheader("Filtered Project Data")
st.dataframe(filtered_data)

# Key Metrics
st.subheader("Key Metrics")
total_projected_cost = filtered_data["ProjectedCost"].sum()  # Full dollar amount
readiness_percentage = filtered_data["Project Readiness Percentage"].mean() if not filtered_data.empty else 0
st.metric(label="Total Projected Cost", value=f"${total_projected_cost:,.0f}")
st.metric(label="Average Project Readiness", value=f"{readiness_percentage:.1f}%")

# --- Charts Section ---
st.subheader("Visualizations")

# Create 3x2 grid using columns
# Row 1
col1, col2 = st.columns(2)
# Row 2
col3, col4 = st.columns(2)
# Row 3
col5, col6 = st.columns(2)

# Bar chart 1: Count of Programs (Row 1, Col 1)
with col1:
    st.subheader("Program Count by Scope")
    priority_chart = px.histogram(filtered_data, 
                                  x="Program NAME", 
                                  color="Fully Scoped", 
                                  title="",
                                  barmode="group")  # Group bars by color
    priority_chart.update_layout(
        yaxis_title="Count", 
        height=400,
        paper_bgcolor="rgba(255, 255, 255, 0.5)",  # Fixed transparent white
        plot_bgcolor="rgba(255, 255, 255, 0.5)"   # Fixed transparent white
    )
    st.plotly_chart(priority_chart)

# Bar chart 2: Total Cost by Program (Row 1, Col 2) - Linear Scale with Full Dollar Values
with col2:
    st.subheader("Total Cost by Program")
    # Ensure numeric values for cost columns
    for col in ["StandardCost", "ProjectedCost", "Actual Cost"]:
        filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce').fillna(0)

    # Melt the data for plotting
    cost_data = filtered_data[["Program NAME", "StandardCost", "ProjectedCost", "Actual Cost"]].melt(
        id_vars=["Program NAME"], 
        value_vars=["StandardCost", "ProjectedCost", "Actual Cost"], 
        var_name="Cost Type", 
        value_name="Total Cost"
    )
    
    # Group by Program NAME and Cost Type, then sum
    cost_data = cost_data.groupby(["Program NAME", "Cost Type"])["Total Cost"].sum().reset_index()

    # Filter out zero values
    cost_data = cost_data[cost_data["Total Cost"] > 0]

    cost_chart = px.bar(cost_data, 
                        x="Program NAME", 
                        y="Total Cost", 
                        color="Cost Type", 
                        title="",
                        barmode="group", 
                        text=cost_data["Total Cost"].apply(lambda x: f'${x:,.0f}'))  # Full dollar amount with thousand separators
    cost_chart.update_layout(
        yaxis_title="Total Cost ($)",
        yaxis_tickformat="$,.0f",  # Full dollar format with thousand separators
        yaxis_range=[0, cost_data["Total Cost"].max() * 1.2],  # Linear range up to 120% of max
        height=400,
        showlegend=True,
        paper_bgcolor="rgba(255, 255, 255, 0.5)",  # Fixed transparent white
        plot_bgcolor="rgba(255, 255, 255, 0.5)"   # Fixed transparent white
    )
    cost_chart.update_traces(textposition="auto")
    st.plotly_chart(cost_chart)

# Bar chart 3: Spend Profile (Row 2, Col 1) - Full Dollar Values
with col3:
    st.subheader("Spend Profile")
    # Select a single Planned Year for the spend profile
    selected_year = selected_years[0] if selected_years else year_options[0] if year_options else None
    if selected_years and len(selected_years) > 1:
        selected_year = st.sidebar.selectbox("Select Single Planned Year for Spend Profile", options=selected_years)
    spend_data = filtered_data[filtered_data["Planned Year"] == selected_year].copy() if selected_year else filtered_data.iloc[:0].copy()
    
    # Prepare spend profile data
    spend_profile_data = []
    for index, row in spend_data.iterrows():
        if row["Project Duration (days)"] > 0 and not pd.isna(row["Projected Project Start Date"]) and row["ProjectedCost"] > 0:
            start_date = pd.to_datetime(row["Projected Project Start Date"])
            duration = int(row["Project Duration (days)"])
            projected_cost = row["ProjectedCost"]
            
            time_points = np.arange(0, duration + 1)
            peak_day = duration * 0.75  # Peak at 3/4 of duration
            sigma = duration / 6  # Standard deviation
            distribution = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((time_points - peak_day) / sigma) ** 2)
            distribution = distribution / distribution.sum() # Normalize
            spend_amount = projected_cost * distribution
            
            dates = [start_date + timedelta(days=int(t)) for t in time_points]
            for date, amount in zip(dates, spend_amount):
                spend_profile_data.append({
                    "Date": date,
                    "Spend Amount": amount,
                    "Program": row["Program NAME"]
                })
    
    spend_profile_df = pd.DataFrame(spend_profile_data)
    daily_spend = spend_profile_df.groupby("Date")["Spend Amount"].sum().reset_index()
    
    spend_chart = px.line(daily_spend, 
                          x="Date", 
                          y="Spend Amount", 
                          title="",
                          labels={"Spend Amount": "Spend Amount ($)"},  # Full dollar label
                         )
    spend_chart.update_traces(mode="lines+markers")
    spend_chart.update_layout(
        yaxis_tickformat="$,.0f",  # Full dollar format with thousand separators
        yaxis_range=[0, daily_spend["Spend Amount"].max() * 1.1],  # Range based on raw dollars
        height=400,
        paper_bgcolor="rgba(255, 255, 255, 0.5)",  # Fixed transparent white
        plot_bgcolor="rgba(255, 255, 255, 0.5)"   # Fixed transparent white
    )
    st.plotly_chart(spend_chart)

# Bar chart 4: Driver Distribution by Rating (Row 2, Col 2) - Stacked Bar
with col4:
    st.subheader("Driver Distribution by Rating")
    # Prepare data for driver columns
    driver_columns = ["Driver Safety", "Driver Operational", "Driver Environment", "Driver Regulatory", "Driver Compliance"]
    driver_data = filtered_data[driver_columns].melt(var_name="Driver Type", value_name="Rating")
    
    # Group by Driver Type and Rating to count occurrences of Low, Medium, High
    driver_data = driver_data.groupby(["Driver Type", "Rating"]).size().reset_index(name="Count")
    # Filter out missing ratings (e.g., NaN)
    driver_data = driver_data[driver_data["Rating"].notna()]
    
    # Create a stacked bar chart
    driver_chart = px.bar(driver_data, 
                          x="Driver Type", 
                          y="Count", 
                          color="Rating", 
                          title="",
                          text=driver_data["Count"].astype(str),  # Show count on bars
                          barmode="stack")
    driver_chart.update_layout(
        yaxis_title="Number of Projects",
        height=400,
        showlegend=True,
        paper_bgcolor="rgba(255, 255, 255, 0.5)",  # Fixed transparent white
        plot_bgcolor="rgba(255, 255, 255, 0.5)"   # Fixed transparent white
    )
    driver_chart.update_traces(textposition="auto")
    st.plotly_chart(driver_chart)

# Pie chart: Budget Type Distribution (Row 3, Col 1) - Expense/Capital
with col5:
    st.subheader("Budget Type Distribution")
    category_chart = px.pie(filtered_data, 
                            names="Financial Budget Type", 
                            title="")
    category_chart.update_layout(
        height=400,
        paper_bgcolor="rgba(255, 255, 255, 0.5)",  # Fixed transparent white
        plot_bgcolor="rgba(255, 255, 255, 0.5)"   # Fixed transparent white
    )
    st.plotly_chart(category_chart)

# Bar chart 6: Priority Ranking Distribution (Row 3, Col 2)
with col6:
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
        height=400,
        showlegend=False,
        paper_bgcolor="rgba(255, 255, 255, 0.5)",  # Fixed transparent white
        plot_bgcolor="rgba(255, 255, 255, 0.5)"   # Fixed transparent white
    )
    priority_chart.update_traces(textposition="auto")
    st.plotly_chart(priority_chart)

# --- Predictive Risk Analyzer Section ---
st.subheader("Predictive Risk Analyzer (Powered by AI)")

# Compute risk scores for each program
def compute_risk_score(row, base_risk, driver_safety_high_weight, other_driver_high_weight, driver_medium_weight,
                       cost_overrun_penalty, priority_threshold, priority_penalty, readiness_threshold, readiness_penalty):
    risk_score = base_risk  # Use user-defined base risk score

    # Driver Ratings (Safety, Regulatory, Compliance)
    driver_columns = ["Driver Safety", "Driver Regulatory", "Driver Compliance"]
    for driver in driver_columns:
        if row[driver] == "High":
            if driver == "Driver Safety":
                risk_score += driver_safety_high_weight  # User-defined weight for Driver Safety
            else:
                risk_score += other_driver_high_weight  # User-defined weight for other drivers
        elif row[driver] == "Medium":
            risk_score += driver_medium_weight  # User-defined weight for "Medium"

    # Cost Overrun Potential
    if row["StandardCost"] > 0 and row["ProjectedCost"] / row["StandardCost"] > 1.1:  # More than 10% overrun
        risk_score += cost_overrun_penalty  # User-defined penalty

    # Priority Ranking
    if row["Priority Ranking "] > priority_threshold:  # User-defined threshold
        risk_score += priority_penalty  # User-defined penalty

    # Project Readiness Percentage
    if row["Project Readiness Percentage"] < readiness_threshold:  # User-defined threshold
        risk_score += readiness_penalty  # User-defined penalty

    # Cap the score at 100
    return min(risk_score, 100)

# Apply risk scoring to filtered data with user-defined parameters
filtered_data["Risk Score"] = filtered_data.apply(
    lambda row: compute_risk_score(row, base_risk, driver_safety_high_weight, other_driver_high_weight, driver_medium_weight,
                                   cost_overrun_penalty, priority_threshold, priority_penalty, readiness_threshold, readiness_penalty),
    axis=1
)

# Aggregate risk scores by Program NAME
risk_data = filtered_data.groupby("Program NAME")["Risk Score"].mean().reset_index()

# Display average risk score as a metric
average_risk = risk_data["Risk Score"].mean() if not risk_data.empty else 0
st.metric(label="Average Risk Score", value=f"{average_risk:.1f}/100", delta=None)

# Add warning if any program exceeds user-defined risk threshold
if not risk_data.empty and risk_data["Risk Score"].max() > alert_threshold:
    st.warning(f"High Risk Detected: Review programs with risk scores above {alert_threshold}!")

# Plot risk scores as a bar chart
risk_chart = px.bar(risk_data, 
                    x="Program NAME", 
                    y="Risk Score", 
                    title="",
                    text=risk_data["Risk Score"].apply(lambda x: f'{x:.1f}'),  # Show risk score on bars
                    color="Risk Score",
                    color_continuous_scale=px.colors.sequential.Reds)  # Gradient from low (light) to high (dark red)
risk_chart.update_layout(
    yaxis_title="Risk Score (0-100)",
    yaxis_range=[0, 100],
    height=400,
    showlegend=False,
    paper_bgcolor="rgba(255, 255, 255, 0.5)",  # Fixed transparent white
    plot_bgcolor="rgba(255, 255, 255, 0.5)"   # Fixed transparent white
)
risk_chart.update_traces(textposition="auto")
st.plotly_chart(risk_chart)

# --- Budget Certainty Tool Button ---
if "show_budget_tool" not in st.session_state:
    st.session_state.show_budget_tool = False

if st.button("Assess Budget Certainty"):
    st.session_state.show_budget_tool = not st.session_state.show_budget_tool

# --- Budget Certainty Tool Section ---
if st.session_state.show_budget_tool:
    st.subheader("Budget Certainty Assessment Tool")
    
    # Select year for analysis
    analysis_year = st.selectbox("Select Year for Budget Analysis", options=year_options, index=0 if year_options else None)
    
    # Set annual budget (placeholder, adjustable by user)
    annual_budget = st.number_input("Enter Annual Budget ($)", min_value=0, value=40000000, step=1000000)
    
    # Filter data for the selected year
    budget_data = filtered_data[filtered_data["Planned Year"] == analysis_year].copy()
    
    # Compute total projected cost for the year
    total_projected_cost_year = budget_data["ProjectedCost"].sum()
    total_standard_cost_year = budget_data["StandardCost"].sum()
    
    # Calculate probability to hit budget for each project
    def calculate_budget_probability(row):
        base_probability = 0.9  # Base 90% chance if on budget
        cost_ratio = row["ProjectedCost"] / row["StandardCost"] if row["StandardCost"] > 0 else 1.0
        risk_score = row["Risk Score"]  # Access risk score for this row
        risk_impact = 1 - (risk_score / 100)  # Higher risk reduces probability
        probability = base_probability * (1 / cost_ratio) * risk_impact
        return max(0, min(1, probability))  # Cap between 0 and 1
    
    # Apply risk scoring and probability calculation
    budget_data["Risk Score"] = budget_data.apply(
        lambda row: compute_risk_score(row, base_risk, driver_safety_high_weight, other_driver_high_weight, driver_medium_weight,
                                       cost_overrun_penalty, priority_threshold, priority_penalty, readiness_threshold, readiness_penalty),
        axis=1
    )
    budget_data["Budget Probability"] = budget_data.apply(calculate_budget_probability, axis=1)
    
    # Adjust overall probability based on annual budget
    budget_ratio = annual_budget / total_projected_cost_year if total_projected_cost_year > 0 else 1.0
    base_overall_probability = budget_data["Budget Probability"].mean() if not budget_data.empty else 0
    # If total projected cost exceeds budget, reduce probability; if under, increase it slightly
    if total_projected_cost_year > annual_budget:
        budget_factor = budget_ratio  # Reduce probability if over budget
    else:
        budget_factor = 1 + (1 - budget_ratio) * 0.1  # Slightly increase probability if under budget (up to 10% boost)
    
    overall_probability = base_overall_probability * budget_factor
    overall_probability = max(0, min(1, overall_probability))  # Cap between 0 and 1
    
    # Propose actions based on risk and probability
    def propose_action(row):
        if row["Risk Score"] > 70 and row["Budget Probability"] < 0.5:
            return "Pause"
        elif row["Risk Score"] < 30 and row["Budget Probability"] > 0.8:
            return "Accelerate"
        else:
            return "Continue"
    
    budget_data["Proposed Action"] = budget_data.apply(propose_action, axis=1)
    
    # Display analysis
    st.write(f"### Budget Analysis for {analysis_year}")
    st.write(f"Total Projected Cost: ${total_projected_cost_year:,.0f}")
    st.write(f"Total Standard Cost: ${total_standard_cost_year:,.0f}")
    st.write(f"Remaining Budget: ${max(0, annual_budget - total_projected_cost_year):,.0f}")
    
    st.metric(label="Overall Probability to Hit Budget", value=f"{overall_probability:.0%}")
    
    st.subheader("Project Recommendations")
    st.dataframe(budget_data[["Program NAME", "ProjectedCost", "StandardCost", "Risk Score", "Budget Probability", "Proposed Action"]].style.format({
        "ProjectedCost": "${:,.0f}",
        "StandardCost": "${:,.0f}",
        "Risk Score": "{:.1f}",
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

# --- Prioritization Section ---
st.subheader("Prioritized Projects")

# Sort options
sort_column = st.selectbox("Sort by", options=data.columns)
sort_order = st.radio("Order", options=["Ascending", "Descending"])
ascending = True if sort_order == "Ascending" else False

# Sort the data
sorted_data = filtered_data.sort_values(by=sort_column, ascending=ascending)

# Optional: Add a priority score
if "Priority Ranking " in sorted_data.columns and "Required Work Completion Date" in sorted_data.columns:
    sorted_data["Days_Until_Due"] = (pd.to_datetime(sorted_data["Required Work Completion Date"]) - pd.Timestamp.now()).dt.days
    sorted_data["Urgency"] = sorted_data["Priority Ranking "] * (1 / (sorted_data["Days_Until_Due"] + 1))
    sorted_data = sorted_data.sort_values(by="Urgency", ascending=False)

# Display sorted/prioritized data with styling
st.dataframe(sorted_data[["Program NAME", "Priority Ranking ", "Required Work Completion Date", "Fully Scoped", "Urgency", "Risk Score"]].style.set_properties(**{
    'background-color': '#f9f9f9',
    'border-color': '#dddddd',
    'font-family': 'Arial, sans-serif',
    'text-align': 'center'
}).set_table_styles([{
    'selector': 'th',
    'props': [('background-color', '#2980b9'), ('color', 'white'), ('font-weight', 'bold')]
}]))

# --- Run Instructions ---
# Save as app.py and run with: streamlit run app.py