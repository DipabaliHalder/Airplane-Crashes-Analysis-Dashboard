import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(page_title="Airplane Crashes Analysis", layout="wide")

# Custom CSS to make the app more presentable and colorful
st.markdown("""
<style>
    .reportview-container {
        background: linear-gradient(to right, #1e3c72, #2a5298);
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: white;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
    }
    .stSelectbox {
        color: #FF4B4B;
    }
    .insight-box {
        background-color: rgba(255, 75, 75, 0.1);
        border-left: 5px solid #FF4B4B;
        padding: 10px;
        margin-bottom: 10px;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load and clean the data
@st.cache_data
def load_data():
    df = pd.read_csv('Airplane_Crashes_and_Fatalities_Since_1908.csv')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Aboard'] = pd.to_numeric(df['Aboard'], errors='coerce')
    df['Fatalities'] = pd.to_numeric(df['Fatalities'], errors='coerce')
    df['Ground'] = pd.to_numeric(df['Ground'], errors='coerce')
    return df

df = load_data()

# Helper function for presenting insights
def show_insight(text):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

# Define tab functions
def home_tab():
    st.subheader("**Welcome to the Airplane Crashes Analysis Dashboard!**") 
    st.markdown("**This application provides comprehensive insights into airplane crashes and fatalities since 1908.**")
    
    st.subheader("How to Navigate")
    st.write("""
    1. Use the tabs at the top to explore different sections of the dashboard:
       - **Exploratory Data Analysis (EDA)**: Various charts and graphs about the dataset.
       - **Advanced Clustering**: Perform clustering on selected features and visualize the results.
       - **Recommendations**: Summary of insights and actionable recommendations based on the data analysis.
    2. Interact with the visualizations to gain deeper insights.
    3. Read the insights and recommendations to understand key findings from the data.
    """)
    
    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Incidents", f"{len(df):,}", delta_color="inverse")
    with col2:
        st.metric("Date Range", f"{df['Date'].min().year} - {df['Date'].max().year}")
    with col3:
        st.metric("Total Fatalities", f"{df['Fatalities'].sum():,}", delta_color="inverse")
    
    st.subheader("Sample Dataset")
    st.dataframe(df.head(), use_container_width=True)

def eda_tab():
    st.subheader("Exploratory Data Analysis")
    
    # Time Series Analysis
    yearly_data = df.groupby('Year').agg({'Date': 'count', 'Fatalities': 'sum', 'Aboard': 'sum'}).reset_index()
    yearly_data.columns = ['Year', 'Crashes', 'Fatalities', 'Passengers']
    yearly_data['Fatality_Rate'] = yearly_data['Fatalities'] / yearly_data['Passengers'] * 100

    fig = px.line(yearly_data, x='Year', y=['Crashes', 'Fatalities', 'Fatality_Rate'], 
                  title='Crashes, Fatalities, and Fatality Rate Over Time')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)
    show_insight("There's a general increasing trend in crashes and fatalities over time, with notable spikes during world wars. However, the fatality rate has significantly decreased in recent decades, indicating improved aviation safety measures.")

    # Geographical Analysis
    location_crashes = df['Location'].value_counts().head(10)
    fig = px.bar(x=location_crashes.index, y=location_crashes.values, 
                 labels={'x': 'Location', 'y': 'Number of Crashes'},
                 title='Top 10 Locations with Most Crashes')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)', tickangle=-45),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
    )
    fig.update_traces(marker_color=px.colors.sequential.Plasma)
    st.plotly_chart(fig, use_container_width=True)
    show_insight(f"The location with the most crashes is {location_crashes.index[0]} with {location_crashes.values[0]} crashes, followed by {location_crashes.index[1]} with {location_crashes.values[1]} crashes. This suggests that certain geographical areas may have higher risk factors for aviation incidents.")

    Q1 = df['Fatalities'].quantile(0.25)
    Q3 = df['Fatalities'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df['Fatalities'] >= lower_bound) & (df['Fatalities'] <= upper_bound)]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_filtered['Fatalities'], nbinsx=50, name='Fatalities'))
    fig.add_trace(go.Box(x=df_filtered['Fatalities'], name='Box Plot', boxpoints='all', jitter=0.3, pointpos=-1.8))
    
    fig.update_layout(
        title='Distribution of Fatalities (Outliers Removed)',
        xaxis_title='Number of Fatalities',
        yaxis_title='Frequency',
        bargap=0.2,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    show_insight(f"After removing outliers, we can see that most crashes result in fewer than {int(df_filtered['Fatalities'].quantile(0.75))} fatalities. The median number of fatalities is {int(df_filtered['Fatalities'].median())}. However, it's important to note that we removed {len(df) - len(df_filtered)} outlier crashes with very high fatality counts for this visualization.")

    # Correlation Analysis
    corr_features = ['Year', 'Aboard', 'Fatalities', 'Ground']
    corr_matrix = df[corr_features].corr()
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", title="Correlation Matrix",
                    color_continuous_scale=px.colors.diverging.RdBu)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)
    show_insight("There's a strong positive correlation between 'Aboard' and 'Fatalities', indicating that crashes with more people aboard tend to have higher fatalities. This highlights the importance of safety measures for larger aircraft.")

    # Operator Analysis
    top_operators = df['Operator'].value_counts().head(10)
    fig = px.bar(x=top_operators.index, y=top_operators.values, 
                 labels={'x': 'Operator', 'y': 'Number of Crashes'},
                 title='Top 10 Operators with Most Crashes')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)', tickangle=-45),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
    )
    fig.update_traces(marker_color=px.colors.sequential.Viridis)
    st.plotly_chart(fig, use_container_width=True)
    show_insight(f"The operator with the most crashes is {top_operators.index[0]} with {top_operators.values[0]} crashes. This could be due to various factors such as fleet size, routes flown, or safety practices.")

    # Seasonal Analysis
    monthly_crashes = df.groupby('Month')['Date'].count().reset_index()
    monthly_crashes.columns = ['Month', 'Crashes']
    fig = px.line(monthly_crashes, x='Month', y='Crashes', title='Crashes by Month',
                  labels={'Crashes': 'Number of Crashes'})
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(255, 255, 255, 0.1)')
    )
    fig.update_traces(line=dict(color='#FF4B4B'))
    st.plotly_chart(fig, use_container_width=True)
    peak_month = monthly_crashes.loc[monthly_crashes['Crashes'].idxmax(), 'Month']
    show_insight(f"There appears to be some seasonal variation in crash frequency, with a peak in month {peak_month}. This could be related to weather patterns or increased air traffic during certain times of the year.")

    # Aircraft Type Analysis
    top_types = df['Type'].value_counts().head(10)
    fig = px.pie(values=top_types.values, names=top_types.index, title='Top 10 Aircraft Types Involved in Crashes')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, use_container_width=True)
    show_insight(f"The aircraft type most frequently involved in crashes is {top_types.index[0]}, accounting for {top_types.values[0]} incidents. This could be due to the popularity of this aircraft type or potential safety concerns.")

def clustering_tab():
    st.subheader("Advanced Clustering Analysis")

    # Feature selection for clustering
    numeric_features = ['Year', 'Aboard', 'Fatalities', 'Ground']
    st.markdown("Select features for clustering:")
    features = st.multiselect("Features", numeric_features, default=['Aboard', 'Fatalities', 'Ground'])
    
    if len(features) < 1:
        st.warning("Please select at least one feature for analysis.")
        return

    X = df[features].dropna()
    
    # Ensure X is always 2D
    if len(features) == 1:
        X = X.values.reshape(-1, 1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA Analysis
    st.subheader("Principal Component Analysis (PCA)")
    
    # Adjust PCA based on number of features
    if len(features) > 1:
        pca = PCA()
        pca_result = pca.fit_transform(X_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_
    else:
        pca_result = X_scaled
        explained_variance_ratio = [1.0]
    
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, len(explained_variance_ratio)+1)), 
                         y=explained_variance_ratio, name='Individual',
                         marker_color='#FF4B4B'))
    fig.add_trace(go.Scatter(x=list(range(1, len(cumulative_variance_ratio)+1)), 
                             y=cumulative_variance_ratio, name='Cumulative',
                             line=dict(color='#FFFFFF')))
    fig.update_layout(
        title='Explained Variance Ratio by Principal Components',
        xaxis_title='Principal Components',
        yaxis_title='Explained Variance Ratio',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF'),
        xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=1, color='#FFFFFF'),
        yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=1, color='#FFFFFF')
    )
    st.plotly_chart(fig, use_container_width=True)

    #st.subheader("Feature Contributions to Principal Components")
    if len(features) > 1:
        pca_components = pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(len(features))],
            index=features
        )
    else:
        pca_components = pd.DataFrame(
            [[1.0]],
            columns=['PC1'],
            index=features
        )
    
    # Create a heatmap for PCA components
    fig = px.imshow(pca_components, 
                    labels=dict(x="Principal Components", y="Features", color="Contribution"),
                    x=[f'PC{i+1}' for i in range(len(features))],
                    y=features,
                    color_continuous_scale="RdBu_r")
    fig.update_layout(
        title='Feature Contributions to Principal Components',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF')
    )
    st.plotly_chart(fig, use_container_width=True)

    # PCA Insights
    st.subheader("PCA Insights")
    total_variance_explained = sum(explained_variance_ratio[:min(2, len(features))])
    st.write(f"The first {'two' if len(features) > 1 else 'one'} principal component{'s' if len(features) > 1 else ''} explain{'' if len(features) > 1 else 's'} {total_variance_explained:.2%} of the total variance in the data.")
    
    if len(features) > 1:
        most_important_feature_pc1 = pca_components['PC1'].abs().idxmax()
        most_important_feature_pc2 = pca_components['PC2'].abs().idxmax()
        st.write(f"PC1 is most strongly influenced by {most_important_feature_pc1}, while PC2 is most strongly influenced by {most_important_feature_pc2}.")
    else:
        most_important_feature = pca_components['PC1'].abs().idxmax()
        st.write(f"The principal component is most strongly influenced by {most_important_feature}.")
    
    st.write("This suggests that these features are particularly important in differentiating between different types of airplane crashes.")

    # Clustering algorithm selection
    st.subheader("Algorithm Specific Analysis")
    algorithm = st.selectbox("Select clustering algorithm", ["K-means", "DBSCAN"])

    if algorithm == "K-means":
        # Elbow method for K-means
        inertias = []
        k_range = range(1, min(11, len(X)))
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Calculate the elbow point
        deltas = np.diff(inertias)
        elbow_point = np.argmin(deltas) + 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(k_range), y=inertias, mode='lines+markers',
                                 line=dict(color='#FF4B4B', width=2),
                                 marker=dict(size=8, color='#FF4B4B', symbol='circle')))
        fig.add_trace(go.Scatter(x=[elbow_point+1], y=[inertias[elbow_point]], mode='markers',
                                 marker=dict(size=12, color='#FFFFFF', symbol='star', line=dict(width=2, color='#FF4B4B')),
                                 name='Elbow Point'))
        fig.update_layout(
            title='Elbow Method for Optimal k',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Inertia',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF'),
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=1, color='#FFFFFF'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', gridwidth=1, color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"Based on the elbow method, the recommended number of clusters is: {elbow_point+1}")
        
        n_clusters = st.slider("Number of clusters", 2, min(10, len(X)-1), elbow_point+1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        if len(set(clusters)) > 1:
            silhouette = silhouette_score(X_scaled, clusters)
            st.markdown(f"Silhouette Score: {silhouette:.4f}")
        else:
            st.markdown("Silhouette Score cannot be calculated (insufficient clusters).")

        # K-means Insights
        st.subheader("K-means Clustering Insights")
        cluster_means = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
        
        # Create a heatmap for cluster means
        fig = px.imshow(cluster_means.T, 
                        labels=dict(x="Cluster", y="Features", color="Value"),
                        x=[f'Cluster {i}' for i in range(n_clusters)],
                        y=features,
                        color_continuous_scale="Viridis")
        fig.update_layout(
            title='Cluster Characteristics',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)

    else:  # DBSCAN
        eps = st.slider("DBSCAN eps", 0.1, 2.0, 0.5)
        min_samples = st.slider("DBSCAN min_samples", 2, 10, 5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
        if n_clusters > 1:
            silhouette = silhouette_score(X_scaled, clusters)
            st.markdown(f"Silhouette Score: {silhouette:.4f}")
        else:
            st.markdown("Silhouette Score cannot be calculated (insufficient clusters).")
        
        noise_points = sum(clusters == -1)
        st.write(f"Number of noise points: {noise_points}")

        # DBSCAN Insights
        st.subheader("DBSCAN Clustering Insights")
        cluster_means = pd.DataFrame(scaler.inverse_transform(X_scaled), columns=features)
        cluster_means['Cluster'] = clusters
        cluster_means_grouped = cluster_means.groupby('Cluster').mean()
        
        # Create a heatmap for cluster means
        fig = px.imshow(cluster_means_grouped.T, 
                        labels=dict(x="Cluster", y="Features", color="Value"),
                        x=[f'Cluster {i}' for i in cluster_means_grouped.index if i != -1] + ['Noise'],
                        y=features,
                        color_continuous_scale="Viridis")
        fig.update_layout(
            title='Cluster Characteristics',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF')
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"Noise points: {noise_points} crashes don't belong to any cluster and may represent outliers or unique cases.")

    # Visualization of clustering results
    if len(features) >= 3:
        fig = px.scatter_3d(X, x=features[0], y=features[1], z=features[2],
                            color=clusters, labels={'color': 'Cluster'},
                            title=f'3D Visualization of {algorithm} Clustering',
                            color_continuous_scale=px.colors.qualitative.Plotly)
    elif len(features) == 2:
        fig = px.scatter(X, x=features[0], y=features[1],
                         color=clusters, labels={'color': 'Cluster'},
                         title=f'2D Visualization of {algorithm} Clustering',
                         color_continuous_scale=px.colors.qualitative.Plotly)
    else:
        fig = px.scatter(X, x=features[0], y=[0]*len(X),
                         color=clusters, labels={'color': 'Cluster'},
                         title=f'1D Visualization of {algorithm} Clustering',
                         color_continuous_scale=px.colors.qualitative.Plotly)
        fig.update_yaxes(showticklabels=False, title_text='')
    
    fig.update_layout(
        scene=dict(bgcolor='rgba(0,0,0,0)',
                   xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#FFFFFF'),
                   yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#FFFFFF'),
                   zaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)', color='#FFFFFF')),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF', size=14)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Store clustering results in session state for use in the recommendations tab
    st.session_state['clusters'] = clusters
    st.session_state['clustering_algorithm'] = algorithm
    st.session_state['clustering_features'] = features
    st.session_state['silhouette_score'] = silhouette if 'silhouette' in locals() else None
    st.session_state['pca_components'] = pca_components
    st.session_state['explained_variance_ratio'] = explained_variance_ratio

def recommend_tab():
    #st.subheader("Insights and Recommendations")
    
    eda_recommendations = [
        ("Temporal Trends", 
         f"There's a general increase in both crashes and fatalities over time, with notable spikes during world wars. However, the fatality rate has decreased over recent decades. The peak year for crashes was {df.groupby('Year')['Date'].count().idxmax()} with {df.groupby('Year')['Date'].count().max()} crashes.",
         ["Focus on understanding and replicating the factors that led to improved safety in recent years.",
          "Analyze the specific safety measures implemented in the years following the peak crash year to identify key improvements.",
          "Continue investing in modern safety technologies and training programs to maintain the downward trend in fatality rates."]),
        ("Geographical Patterns",
         f"The top 3 locations with the most crashes are {', '.join(df['Location'].value_counts().head(3).index)}.",
         [f"Conduct in-depth investigations of these high-risk areas, particularly {df['Location'].value_counts().index[0]}, to identify common factors (e.g., weather patterns, terrain, air traffic congestion).",
          "Implement targeted safety measures and potentially reassess flight routes in these regions.",
          "Develop location-specific training programs for pilots and air traffic controllers in high-risk areas."]),
        ("Fatality Distribution",
         f"While most crashes result in relatively fewer fatalities, there are some with very high fatality counts. The crash with the highest fatalities had {df['Fatalities'].max()} deaths.",
         ["Prioritize understanding and preventing high-fatality crashes.",
          "Develop and implement specific safety protocols for larger aircraft or flights with high passenger counts.",
          "Enhance emergency response and evacuation procedures to minimize fatalities in the event of a crash."]),
        ("Operator Analysis",
         f"The operator with the most crashes is {df['Operator'].value_counts().index[0]} with {df['Operator'].value_counts().values[0]} crashes.",
         [f"Investigate the safety practices of {df['Operator'].value_counts().index[0]} and compare them with operators having fewer crashes.",
          "Implement industry-wide best practices based on this analysis.",
          "Consider more frequent safety audits and inspections for operators with higher crash rates."]),
        ("Seasonal Patterns",
         f"The month with the highest number of crashes is {df.groupby('Month')['Date'].count().idxmax()}.",
         ["Investigate potential causes for increased crashes during this month (e.g., weather conditions, holiday traffic).",
          "Implement additional safety measures or pilot training for high-risk periods.",
          "Consider reducing flight frequencies or adjusting routes during historically high-risk periods."]),
        ("Aircraft Type Analysis",
         f"The aircraft type most frequently involved in crashes is {df['Type'].value_counts().index[0]}.",
         ["Conduct a thorough review of this aircraft type's safety features and performance.",
          "Consider implementing additional safety measures or maintenance protocols for frequently involved aircraft types.",
          "Work with manufacturers to address any potential design flaws or recurring issues."])
    ]
    
    for title, insight, recommendations in eda_recommendations:
        st.subheader(f"{title}")
        show_insight(f"{insight}")
        st.write("**Recommendations:**")
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        st.write("---")
        
    # PCA Insights
    if 'pca_components' in st.session_state and 'explained_variance_ratio' in st.session_state:
        st.subheader("Principal Component Analysis (PCA) Insights")
        pca_components = st.session_state['pca_components']
        explained_variance_ratio = st.session_state['explained_variance_ratio']
        
        total_variance_explained = sum(explained_variance_ratio[:min(2, len(pca_components.columns))])
        show_insight(f"The first {'two' if len(pca_components.columns) > 1 else 'one'} principal component{'s' if len(pca_components.columns) > 1 else ''} explain{'' if len(pca_components.columns) > 1 else 's'} {total_variance_explained:.2%} of the total variance in the data.")
        
        if len(pca_components.columns) > 1:
            most_important_feature_pc1 = pca_components['PC1'].abs().idxmax()
            most_important_feature_pc2 = pca_components['PC2'].abs().idxmax()
            show_insight(f"PC1 is most strongly influenced by {most_important_feature_pc1}, while PC2 is most strongly influenced by {most_important_feature_pc2}.")
            
            st.write("**PCA-based Recommendations:**")
            st.write(f"1. Prioritize the analysis and improvement of factors related to {most_important_feature_pc1}, as it has the strongest influence on the primary dimension of variation in the data.")
            st.write(f"2. Consider the interaction between {most_important_feature_pc1} and {most_important_feature_pc2} in safety protocols and risk assessments.")
        else:
            most_important_feature = pca_components['PC1'].abs().idxmax()
            show_insight(f"The principal component is most strongly influenced by {most_important_feature}.")
            
            st.write("**PCA-based Recommendations:**")
            st.write(f"1. Focus on {most_important_feature} as the key factor in understanding and preventing airplane crashes.")
            st.write(f"2. Develop more sophisticated monitoring and safety systems specifically targeting {most_important_feature}.")
        
        st.write("3. Use the PCA results to guide feature selection in predictive models for crash risk assessment.")
        st.write("4. Conduct further research to understand the underlying factors that contribute to the importance of the identified features.")
    st.write("---")


# Main app
st.header("Airplane Crashes Analysis Dashboard")

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Exploratory Data Analysis", "Advanced Clustering", "Recommendations"])

with tab1:
    home_tab()

with tab2:
    eda_tab()

with tab3:
    clustering_tab()

with tab4:
    recommend_tab()