# utils/visualization.py

import plotly.express as px
import plotly.graph_objects as go

def create_histogram(df, column):
    """
    Create a histogram for a numeric column.
    
    Args:
        df (pd.DataFrame): The dataset.
        column (str): Column name.
    
    Returns:
        plotly.graph_objects.Figure: Histogram figure.
    """
    fig = px.histogram(df, x=column, nbins=30, title=f"Distribution of {column}")
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None):
    """
    Create a scatter plot.
    
    Args:
        df (pd.DataFrame): The dataset.
        x_col (str): X-axis column.
        y_col (str): Y-axis column.
        color_col (str, optional): Column to color by.
    
    Returns:
        plotly.graph_objects.Figure: Scatter plot figure.
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                     title=f"{y_col} vs {x_col}")
    return fig

def create_box_plot(df, column, category_col):
    """
    Create a box plot.
    
    Args:
        df (pd.DataFrame): The dataset.
        column (str): Numeric column.
        category_col (str): Categorical column.
    
    Returns:
        plotly.graph_objects.Figure: Box plot figure.
    """
    fig = px.box(df, x=category_col, y=column, title=f"{column} by {category_col}")
    return fig

def create_heatmap(correlation_matrix):
    """
    Create a heatmap for correlation matrix.
    
    Args:
        correlation_matrix (pd.DataFrame): Correlation matrix.
    
    Returns:
        plotly.graph_objects.Figure: Heatmap figure.
    """
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )
    return fig

def create_pie_chart(df, column):
    """
    Create a pie chart for a categorical column.
    
    Args:
        df (pd.DataFrame): The dataset.
        column (str): Categorical column.
    
    Returns:
        plotly.graph_objects.Figure: Pie chart figure.
    """
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.pie(counts, names=column, values='count', title=f"Distribution of {column}")
    return fig

def create_choropleth(df, location_col, value_col):
    """
    Create a choropleth map.
    
    Args:
        df (pd.DataFrame): The dataset.
        location_col (str): Column with country names.
        value_col (str): Column with values to map.
    
    Returns:
        plotly.graph_objects.Figure: Choropleth map figure.
    """
    fig = px.choropleth(
        df,
        locations=location_col,
        locationmode='country names',
        color=value_col,
        title=f"{value_col} by {location_col}",
        color_continuous_scale='Blues'
    )
    return fig
