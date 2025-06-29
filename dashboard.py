import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder


import streamlit as st
from PIL import Image

# Password configuration
CORRECT_PASSWORD = "MSBA382" 

def login():
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        with st.sidebar:
            st.markdown("### 🔒 Please enter the password")
            password = st.text_input("Password", type="password")
            if st.button("Login"):
                if password == CORRECT_PASSWORD:
                    st.session_state["authenticated"] = True
                else:
                    st.error("Incorrect password")

        # Center layout
        st.markdown("<h1 style='text-align:center; color:#333;'>Colon & Rectum Cancer Analytics Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center; color:#555;'>Presented by Joanna El Kazzi</h3>", unsafe_allow_html=True)

        # Show image in center (update path if needed)
        image = Image.open("colon_image.jpeg")  # 👈 Ensure this file is in the same folder as dashboard.py
        st.image(image, use_container_width=True)

        st.stop()

# 🔑 Run the login check before anything else
login()

# Page config
# Page config
st.set_page_config(page_title="Colon & Rectum Cancer Analytics Dashboard", layout="wide")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .filter-container {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h4 style='text-align:center; margin-top: 2rem;'>MSBA382 Streamlit Project</h4>", unsafe_allow_html=True)
st.markdown('<h1 class="main-header">Colon & Rectum Cancer Global Analytics Dashboard</h1>', unsafe_allow_html=True)


uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if not uploaded_file:
    st.sidebar.info("Upload a CSV file to activate the dashboard.")
    st.stop()

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()
    return df

df = load_data(uploaded_file)

st.sidebar.markdown("## Dashboard Filters")

with st.sidebar:
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)

    selected_measure = st.selectbox("Select Burden Type", ['All'] + sorted(df['measure_name'].unique()))
    selected_location = st.selectbox("Select Location", ['All'] + sorted(df['location_name'].unique()))
    selected_sex = st.selectbox("Select Gender", ['All'] + sorted(df['sex_name'].unique()))
    selected_age = st.multiselect("Select Age Groups", ['All'] + sorted(df['age_name'].unique()), default=['All'])
    selected_years = st.slider("Select Year Range", min_value=min(df['year']), max_value=max(df['year']), value=(min(df['year']), max(df['year'])))
    selected_metric = st.selectbox("Select Metric", sorted(df['metric_name'].unique()))

    st.markdown('</div>', unsafe_allow_html=True)

# Filter logic
filtered_df = df.copy()

if selected_location != 'All':
    filtered_df = filtered_df[filtered_df['location_name'] == selected_location]
if selected_sex != 'All':
    filtered_df = filtered_df[filtered_df['sex_name'] == selected_sex]
if 'All' not in selected_age:
    filtered_df = filtered_df[filtered_df['age_name'].isin(selected_age)]
filtered_df = filtered_df[(filtered_df['year'] >= selected_years[0]) & (filtered_df['year'] <= selected_years[1])]
filtered_df = filtered_df[filtered_df['metric_name'] == selected_metric]
if selected_measure != 'All':
    filtered_df = filtered_df[filtered_df['measure_name'] == selected_measure]

if filtered_df.empty:
    st.warning("No data found for the selected filters. Please try a different combination.")
    st.stop()

# KPIs
st.markdown("## 📈 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Burden by Type", f"{filtered_df['val'].sum():,.0f}")
with col2:
    st.metric("Average Rate", f"{filtered_df['val'].mean():.2f}")
with col3:
    st.metric("Locations Covered", filtered_df['location_name'].nunique())
with col4:
    st.metric("Years of Data", filtered_df['year'].nunique())

# Tabs
st.markdown("## Interactive Analytics Tabs")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Geographic", "Demographics", "Trends", "Comparative", "Predictive"])

with tab1:
    geo_df = filtered_df.groupby('location_name')['val'].agg(['sum', 'mean']).reset_index()
    geo_df.columns = ['location_name', 'total_burden', 'average_rate']
    
    # Bar Chart
    fig_geo = px.bar(
        geo_df.sort_values('total_burden', ascending=False).head(20),
        x='location_name', y='total_burden', color='average_rate',
        title="Locations by Burden"
    )
    fig_geo.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_geo, use_container_width=True)

    # Regional  Map
    st.markdown("### Regional Map of Disease Burden")
    region_coords = {
        'Africa': {'lat': 1.5, 'lon': 17.3},
        'Asia': {'lat': 34.0, 'lon': 100.6},
        'Europe': {'lat': 54.5, 'lon': 15.3},
        'America': {'lat': 10.0, 'lon': -70.0}
    }

    map_df = filtered_df.groupby('location_name')['val'].sum().reset_index()
    map_df['lat'] = map_df['location_name'].map(lambda x: region_coords.get(x, {}).get('lat'))
    map_df['lon'] = map_df['location_name'].map(lambda x: region_coords.get(x, {}).get('lon'))
    map_df = map_df.dropna()

    fig_map = px.scatter_geo(
        map_df,
        lat='lat', lon='lon',
        size='val', color='location_name',
        hover_name='location_name',
        size_max=60,
        projection='natural earth',
        title='Burden by Region '
    )
    fig_map.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

with tab2:
    gender_df = filtered_df.groupby('sex_name')['val'].sum().reset_index()
    age_df = filtered_df.groupby('age_name')['val'].sum().reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(gender_df, names='sex_name', values='val', title="Distribution by Gender"), use_container_width=True)
    with col2:
        fig_age = px.bar(age_df, x='age_name', y='val', title="Distribution by Age Group")
        fig_age.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_age, use_container_width=True)

    # Age Gender Heatmap
    st.markdown("#### Age-Gender Heatmap")
    pivot = filtered_df.pivot_table(values='val', index='age_name', columns='sex_name', aggfunc='mean')
    fig_heatmap = px.imshow(
        pivot,
        text_auto=True,
        title="Average Burden by Age & Gender",
        labels=dict(color="Burden"),
        aspect="auto",
    )
    fig_heatmap.update_layout(
        width=800,
        height=700,
        margin=dict(l=50, r=50, t=60, b=60),
        font=dict(size=14),
        xaxis_title="Gender",
        yaxis_title="Age Group",
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)

with tab3:
    time_df = filtered_df.groupby(['year', 'sex_name'])['val'].sum().reset_index()
    fig_time = px.line(time_df, x='year', y='val', color='sex_name', title="Trend by Gender")
    st.plotly_chart(fig_time, use_container_width=True)
    yearly_totals = filtered_df.groupby('year')['val'].sum().reset_index()
    yearly_totals['pct_change'] = yearly_totals['val'].pct_change() * 100
    st.plotly_chart(px.bar(yearly_totals.dropna(), x='year', y='pct_change', title="Year-over-Year Change (%)"), use_container_width=True)

with tab4:
    compare_type = st.selectbox("Compare by", ['Location', 'Age Group', 'Gender'])
    if compare_type == 'Location':
        top_locs = filtered_df.groupby('location_name')['val'].sum().nlargest(10).index.tolist()
        comp_df = filtered_df[filtered_df['location_name'].isin(top_locs)]
        fig = px.box(comp_df, x='location_name', y='val', title="Distribution by Location")
    elif compare_type == 'Age Group':
        fig = px.box(filtered_df, x='age_name', y='val', title="Distribution by Age Group")
    else:
        fig = px.box(filtered_df, x='sex_name', y='val', title="Distribution by Gender")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.subheader("Forecasting Future Disease Burden")

    # Apply all sidebar filters
    df_pred = df.copy()
    if selected_location != 'All':
        df_pred = df_pred[df_pred['location_name'] == selected_location]
    if selected_sex != 'All':
        df_pred = df_pred[df_pred['sex_name'] == selected_sex]
    if selected_measure != 'All':
        df_pred = df_pred[df_pred['measure_name'] == selected_measure]
    if selected_age != ['All']:
        df_pred = df_pred[df_pred['age_name'].isin(selected_age)]
    df_pred = df_pred[df_pred['metric_name'] == selected_metric]

    if df_pred.empty:
        st.warning("No data found for the selected filters.")
        st.stop()

    # Aggregate by year (sum or average — here we use sum)
    df_yearly = df_pred.groupby('year')['val'].sum().reset_index()

    if len(df_yearly) < 5:
        st.warning("Not enough yearly data points for reliable prediction.")
        st.stop()

    # Train linear regression model
    from sklearn.linear_model import LinearRegression

    X = df_yearly[['year']]
    y = df_yearly['val']
    model = LinearRegression()
    model.fit(X, y)
    df_yearly['prediction'] = model.predict(X)

    # Forecast next 5 years
    future_years = pd.DataFrame({'year': list(range(X['year'].max() + 1, X['year'].max() + 6))})
    future_years['prediction'] = model.predict(future_years)
    future_years['val'] = None  # No actual value yet

    # Combine actual and forecast for download/export
    df_result = pd.concat([df_yearly, future_years], ignore_index=True)
    df_result['type'] = ['Actual'] * len(df_yearly) + ['Forecast'] * len(future_years)

    # Plot
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['val'], mode='lines+markers', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df_yearly['year'], y=df_yearly['prediction'], mode='lines', name='Fitted', line=dict(dash='dot', color='green')))
    fig.add_trace(go.Scatter(x=future_years['year'], y=future_years['prediction'], mode='lines+markers', name='Forecast (Next 5 Years)', line=dict(dash='dash', color='red')))

    fig.update_layout(
        title="Forecasting Disease Burden Based on Selected Filters",
        xaxis_title="Year",
        yaxis_title=selected_metric,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.metric("Model R²", f"{model.score(X, y):.3f}")

    # CSV download
    import io
    csv_buffer = io.StringIO()
    df_result.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Forecast Data as CSV",
        data=csv_buffer.getvalue(),
        file_name="forecast_disease_burden.csv",
        mime="text/csv"
    )


