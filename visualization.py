# utils/visualization.py

import plotly.express as px
import pandas as pd

def create_histogram(df, column):
    """
    Create a histogram for a specified column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The column to plot.

    Returns:
        plotly.graph_objs._figure.Figure: The histogram figure.
    """
    fig = px.histogram(df, x=column, nbins=30, title=f'Histogram of {column}')
    return fig

def create_scatter_plot(df, x_col, y_col, color_col=None):
    """
    Create a scatter plot between two columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        x_col (str): The column for the X-axis.
        y_col (str): The column for the Y-axis.
        color_col (str, optional): The column to color-code the points.

    Returns:
        plotly.graph_objs._figure.Figure: The scatter plot figure.
    """
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                     title=f'Scatter Plot of {y_col} vs {x_col}')
    return fig

def create_box_plot(df, numeric_col, category_col):
    """
    Create a box plot for a numeric column grouped by a categorical column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        numeric_col (str): The numeric column to plot.
        category_col (str): The categorical column to group by.

    Returns:
        plotly.graph_objs._figure.Figure: The box plot figure.
    """
    fig = px.box(df, x=category_col, y=numeric_col, title=f'Box Plot of {numeric_col} by {category_col}')
    return fig

def create_heatmap(correlation_matrix):
    """
    Create a heatmap for a correlation matrix.

    Args:
        correlation_matrix (pd.DataFrame): The correlation matrix.

    Returns:
        plotly.graph_objs._figure.Figure: The heatmap figure.
    """
    fig = px.imshow(correlation_matrix, text_auto=True, aspect="auto",
                    title='Correlation Heatmap')
    return fig

def create_pie_chart(df, column):
    """
    Create a pie chart for a categorical column.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column (str): The categorical column to plot.

    Returns:
        plotly.graph_objs._figure.Figure: The pie chart figure.
    """
    counts = df[column].value_counts().reset_index()
    counts.columns = [column, 'count']
    fig = px.pie(counts, names=column, values='count', title=f'Pie Chart of {column}')
    return fig

def create_choropleth(df, location_col, value_col):
    """
    Create a choropleth map.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        location_col (str): The column with location data (e.g., country names or ISO codes).
        value_col (str): The column with values to visualize.

    Returns:
        plotly.graph_objs._figure.Figure: The choropleth map figure.
    """
    fig = px.choropleth(df, locations=location_col,
                        locationmode='country names',
                        color=value_col,
                        hover_name=location_col,
                        color_continuous_scale=px.colors.sequential.Plasma,
                        title=f'Choropleth Map of {value_col} by {location_col}')
    return fig
