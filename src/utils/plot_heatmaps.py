import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from io import BytesIO
import base64

def create_calendar_df(df: pd.DataFrame, start_date: str, end_date: str):
    """Returns a Series indexed by date, covering the date range, with zeros for missing days."""
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').resample('D')['Total Pallets'].sum().to_frame()
    full_range = pd.date_range(start=start_date, end=end_date)
    df = df.reindex(full_range, fill_value=0)
    df.index.name = 'Date'
    return df['Total Pallets']

def plot_single_calendar_with_months(ax, series: pd.Series, start: pd.Timestamp, end: pd.Timestamp, cmap='YlGn'):
    """Plots a single calendar heatmap with visible month boundaries and month labels."""
    num_days = (end - start).days + 1
    start_weekday = start.weekday()
    num_weeks = ((num_days + start_weekday - 1) // 7) + 2
    data = np.full((7, num_weeks), np.nan)

    month_ticks = {}
    for date in pd.date_range(start, end):
        week = ((date - start).days + start_weekday) // 7
        weekday = date.weekday()
        value = series.get(date, np.nan)
        data[weekday, week] = value
        if date.day == 1:
            month_name = date.strftime('%b %Y')
            month_ticks[week] = month_name

    im = ax.imshow(data, aspect='auto', cmap=cmap, origin='upper')
    ax.set_yticks(range(7))
    ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_xticks(list(month_ticks.keys()))
    ax.set_xticklabels(list(month_ticks.values()), rotation=45, ha='right')
    ax.tick_params(axis='x', which='both', bottom=False, top=False)

    for tick in month_ticks:
        ax.add_patch(patches.Rectangle((tick - 0.5, -0.5), 1, 7, fill=False, edgecolor='black', linewidth=1))

    return im
def plot_dual_calendar_heatmaps(df1: pd.DataFrame, df2: pd.DataFrame, save_path="calendar_dual_heatmap.png"):
    # Determine earliest and latest year in combined data
    min_year = min(df1['Date'].min().year, df2['Date'].min().year)
    max_year = max(df1['Date'].max().year, df2['Date'].max().year)

    # Pad to full years
    start_date = pd.to_datetime(f"{min_year}-01-01")
    end_date = pd.to_datetime(f"{max_year}-12-31")

    # Normalize input data to full-year span
    series1 = create_calendar_df(df1, start_date, end_date)
    series2 = create_calendar_df(df2, start_date, end_date)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(max(16, (end_date - start_date).days // 20), 6), constrained_layout=True)
    im1 = plot_single_calendar_with_months(axes[0], series1, start_date, end_date)
    im2 = plot_single_calendar_with_months(axes[1], series2, start_date, end_date)

    axes[0].set_title("Calendar Heatmap: Original Shipments")
    axes[1].set_title("Calendar Heatmap: Consolidated Shipments")

    # Shared colorbar with legend title
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6, pad=0.05)
    cbar.set_label('Total Pallets Shipped', fontsize=12, fontweight='bold', family='sans-serif', labelpad=10)


    # Save to file
    plt.savefig(save_path, dpi=300)

    # Return as base64
    image_bytes = BytesIO()
    plt.savefig(image_bytes, format='png')
    image_bytes.seek(0)
    plt.close(fig)
    
    return image_bytes, fig

def plot_dual_calendar_heatmaps_filtered(df1: pd.DataFrame, df2: pd.DataFrame,start_date,end_date, save_path="calendar_dual_heatmap.png"):
    # Determine earliest and latest year in combined data
    min_year = min(df1['Date'].min().year, df2['Date'].min().year)
    max_year = max(df1['Date'].max().year, df2['Date'].max().year)

    # Pad to full years
    # start_date = pd.to_datetime(f"{min_year}-01-01")
    # end_date = pd.to_datetime(f"{max_year}-12-31")

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Normalize input data to full-year span
    series1 = create_calendar_df(df1, start_date, end_date)
    series2 = create_calendar_df(df2, start_date, end_date)

    # Plotting
    fig, axes = plt.subplots(2, 1, figsize=(max(16, (end_date - start_date).days // 20), 6), constrained_layout=True)
    im1 = plot_single_calendar_with_months(axes[0], series1, start_date, end_date)
    im2 = plot_single_calendar_with_months(axes[1], series2, start_date, end_date)

    axes[0].set_title("Calendar Heatmap: Original Shipments")
    axes[1].set_title("Calendar Heatmap: Consolidated Shipments")

    # Shared colorbar with legend title
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), orientation='horizontal', shrink=0.6, pad=0.05)
    cbar.set_label('Total Pallets Shipped', fontsize=12, fontweight='bold', family='sans-serif', labelpad=10)


    # Save to file
    plt.savefig(save_path, dpi=300)

    # Return as base64
    image_bytes = BytesIO()
    plt.savefig(image_bytes, format='png')
    image_bytes.seek(0)
    plt.close(fig)
    
    return image_bytes

