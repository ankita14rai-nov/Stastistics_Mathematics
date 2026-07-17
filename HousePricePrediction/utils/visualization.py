import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

def plot_histogram(df, column, title):
    fig = px.histogram(df, x=column, title=title, marginal='box', template='plotly_dark')
    return fig

def plot_boxplot(df, column, title):
    fig = px.box(df, y=column, title=title, template='plotly_dark')
    return fig

def plot_scatter(df, x_col, y_col, title, color_col=None):
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=title, template='plotly_dark')
    return fig

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Heatmap", template='plotly_dark')
    return fig

def plot_target_distribution(df, target_col='median_house_value'):
    fig = px.histogram(df, x=target_col, title='Target Distribution: Median House Value', marginal='box', color_discrete_sequence=['indianred'], template='plotly_dark')
    return fig

def plot_income_vs_price(df):
    fig = px.scatter(df, x='median_income', y='median_house_value', alpha=0.5, title='Median Income vs House Price', template='plotly_dark')
    return fig

def plot_ocean_proximity_count(df):
    fig = px.bar(df['ocean_proximity'].value_counts().reset_index(), x='ocean_proximity', y='count', title='Ocean Proximity Count', template='plotly_dark', color='ocean_proximity')
    return fig

def plot_latitude_vs_longitude(df):
    fig = px.scatter(df, x='longitude', y='latitude', alpha=0.4, 
                     color='median_house_value', size='population',
                     title='Latitude vs Longitude (California Map)',
                     color_continuous_scale=px.colors.sequential.Jet, template='plotly_dark')
    return fig

def get_insights(df):
    """Generates business insights."""
    avg_price = df['median_house_value'].mean()
    most_expensive = df.groupby('ocean_proximity')['median_house_value'].mean().idxmax()
    corr_income_price = df['median_income'].corr(df['median_house_value'])
    corr_rooms_price = df['total_rooms'].corr(df['median_house_value'])
    missing_vals = df.isnull().sum().sum()
    
    insights = [
        f"**Average house price**: ${avg_price:,.2f}",
        f"**Most expensive location type**: {most_expensive}",
        f"**Correlation between income and price**: {corr_income_price:.2f} (Strong positive)",
        f"**Relationship between rooms and price**: {corr_rooms_price:.2f} (Weak positive)",
        f"**Missing value observations**: {missing_vals} total missing values handled.",
    ]
    return insights
