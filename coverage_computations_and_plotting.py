import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Function to load and preprocess data
def load_and_merge_data():
    """Load datasets and merge them based on user_id and month."""
    monthly_user_data = pd.read_csv('path/to/monthly_user_data.csv')
    addiction_status = pd.read_csv('path/to/addiction_data.csv')

    # Add chronological month order
    monthly_user_data['month_chronological_order'] = monthly_user_data['window_id'] + 1

    # Merge data on user_id and month
    merged_df = pd.merge(
        monthly_user_data,
        addiction_status,
        on=['user_id', 'month_chronological_order'],
        how='inner'
    )

    # Add addiction group labels
    def get_addiction_group(row):
        if row['preds_3_label_criteria'] == 2:
            return 'Hard Addicted'
        elif row['preds_3_label_criteria'] == 1:
            return 'Soft Addicted'
        else:
            return 'Non-Addicted'

    merged_df['addiction_group'] = merged_df.apply(get_addiction_group, axis=1)

    return merged_df

# Function to calculate coverage ratios
def calculate_coverage_ratios(merged_df):
    """Calculate average coverage ratios and user counts per month and group."""
    coverage_df = merged_df.groupby(['month_chronological_order', 'addiction_group']).agg({
        'level_1_coverage_norm': 'mean',
        'level_2_coverage_norm': 'mean',
        'level_3_coverage_norm': 'mean',
        'user_id': 'count'
    }).reset_index()

    # Rename columns for clarity
    coverage_df.rename(columns={
        'level_1_coverage_norm': 'avg_level_1_coverage',
        'level_2_coverage_norm': 'avg_level_2_coverage',
        'level_3_coverage_norm': 'avg_level_3_coverage',
        'user_id': 'user_count'
    }, inplace=True)

    return coverage_df

# Function to plot bootstrap coverage ratios (Figure 1)
def plot_bootstrap_coverage(bootstrap_df):
    """Plot bootstrap-normalized coverage ratios for each level and group."""
    bootstrap_df['addiction_group'] = bootstrap_df['addiction_group'].replace(
        {'Non-Addicted': 'Non-Addicted',
         'Soft Addicted': 'Mildly Addicted',
         'Hard Addicted': 'Severely Addicted'}
    )
    group_order = ['Non-Addicted', 'Mildly Addicted', 'Severely Addicted']
    colors = ['#22974F', '#25B1E8', '#E8335A']
    linestyles = [':', '--', '-']
    alphas = [0.35, 0.55, 0.9]

    # Pivot data for plotting
    bootstrap_pivot = bootstrap_df.pivot(index='month_chronological_order', 
                                         columns='addiction_group',
                                         values=['level_1_coverage_normalized',
                                                 'level_2_coverage_normalized',
                                                 'level_3_coverage_normalized'])

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, level in enumerate(['level_1_coverage_normalized', 
                               'level_2_coverage_normalized', 
                               'level_3_coverage_normalized']):
        for group in group_order:
            ax.plot(
                bootstrap_pivot.index,
                bootstrap_pivot[level][group],
                label=f'L{i + 1} {group}',
                color=colors[i],
                linestyle=linestyles[group_order.index(group)],
                linewidth=4,
                alpha=alphas[group_order.index(group)]
            )
    
    # Customize and add legends
    ax.set_xlabel('Month', fontsize=24)
    ax.set_ylabel('Coverage Ratio', fontsize=24)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.spines[['right', 'top']].set_visible(False)

    # Custom legends
    color_handles = [Line2D([0], [0], color=color, lw=4, label=f'L{i + 1}') for i, color in enumerate(colors)]
    linestyle_handles = [Line2D([0], [0], color='black', linestyle=linestyle, lw=4, label=label) 
                         for linestyle, label in zip(linestyles, group_order)]

    color_legend = ax.legend(handles=color_handles, title='Levels', loc='upper left', fontsize=18, title_fontsize=18)
    ax.add_artist(color_legend)
    ax.legend(handles=linestyle_handles, title='Groups', loc='upper right', fontsize=18, title_fontsize=18)

    plt.tight_layout()
    plt.show()

# Function to plot average bootstrap coverage per group (Figure 2)
def plot_average_coverage(bootstrap_df):
    """Plot average coverage across levels for each addiction group."""
    bootstrap_df['average_coverage_normalized'] = (
        bootstrap_df[['level_1_coverage_normalized', 
                      'level_2_coverage_normalized', 
                      'level_3_coverage_normalized']].mean(axis=1)
    )
    group_order = ['Non-Addicted', 'Mildly Addicted', 'Severely Addicted']

    # Pivot data for plotting
    average_pivot = bootstrap_df.pivot(index='month_chronological_order', 
                                       columns='addiction_group',
                                       values='average_coverage_normalized')

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#2A5580']
    linestyles = [':', '--', '-']
    alphas = [0.35, 0.55, 0.9]

    for group in group_order:
        ax.plot(
            average_pivot.index,
            average_pivot[group],
            label=group,
            color=colors[0],
            linestyle=linestyles[group_order.index(group)],
            linewidth=4,
            alpha=alphas[group_order.index(group)]
        )

    # Customize and add legend
    ax.set_xlabel('Month', fontsize=24)
    ax.set_ylabel('Coverage Ratio', fontsize=24)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.legend(loc='upper left', fontsize=18)
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    plt.show()

# Function to plot combined coverage of Addicted vs Non-Addicted (Figure 3)
# Function to plot bootstrap coverage ratios (Figure 1)
def plot_bootstrap_coverage(bootstrap_df):
    """Plot bootstrap-normalized coverage ratios for each level and group."""
    bootstrap_df['addiction_group'] = bootstrap_df['addiction_group'].replace(
        {'Non-Addicted': 'Non-Addicted',
         'Soft Addicted': 'Mildly Addicted',
         'Hard Addicted': 'Severely Addicted'}
    )
    group_order = ['Non-Addicted', 'Mildly Addicted', 'Severely Addicted']
    colors = ['#22974F', '#25B1E8', '#E8335A']
    linestyles = [':', '--', '-']
    alphas = [0.35, 0.55, 0.9]

    # Pivot data for plotting
    bootstrap_pivot = bootstrap_df.pivot(index='month_chronological_order', 
                                         columns='addiction_group',
                                         values=['level_1_coverage_normalized',
                                                 'level_2_coverage_normalized',
                                                 'level_3_coverage_normalized'])

    fig, ax = plt.subplots(figsize=(9, 6))
    for i, level in enumerate(['level_1_coverage_normalized', 
                               'level_2_coverage_normalized', 
                               'level_3_coverage_normalized']):
        for group in group_order:
            ax.plot(
                bootstrap_pivot.index,
                bootstrap_pivot[level][group],
                label=f'L{i + 1} {group}',
                color=colors[i],
                linestyle=linestyles[group_order.index(group)],
                linewidth=4,
                alpha=alphas[group_order.index(group)]
            )
    
    # Customize and add legends
    ax.set_xlabel('Month', fontsize=24)
    ax.set_ylabel('Coverage Ratio', fontsize=24)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.spines[['right', 'top']].set_visible(False)

    # Custom legends
    color_handles = [Line2D([0], [0], color=color, lw=4, label=f'L{i + 1}') for i, color in enumerate(colors)]
    linestyle_handles = [Line2D([0], [0], color='black', linestyle=linestyle, lw=4, label=label) 
                         for linestyle, label in zip(linestyles, group_order)]

    color_legend = ax.legend(handles=color_handles, title='Levels', loc='upper left', fontsize=18, title_fontsize=18)
    ax.add_artist(color_legend)
    ax.legend(handles=linestyle_handles, title='Groups', loc='upper right', fontsize=18, title_fontsize=18)

    plt.tight_layout()
    # Save as PDF
    plt.savefig('figure_bootstrap_coverage.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory

# Function to plot average bootstrap coverage per group (Figure 2)
def plot_average_coverage(bootstrap_df):
    """Plot average coverage across levels for each addiction group."""
    bootstrap_df['average_coverage_normalized'] = (
        bootstrap_df[['level_1_coverage_normalized', 
                      'level_2_coverage_normalized', 
                      'level_3_coverage_normalized']].mean(axis=1)
    )
    group_order = ['Non-Addicted', 'Mildly Addicted', 'Severely Addicted']

    # Pivot data for plotting
    average_pivot = bootstrap_df.pivot(index='month_chronological_order', 
                                       columns='addiction_group',
                                       values='average_coverage_normalized')

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ['#2A5580']
    linestyles = [':', '--', '-']
    alphas = [0.35, 0.55, 0.9]

    for group in group_order:
        ax.plot(
            average_pivot.index,
            average_pivot[group],
            label=group,
            color=colors[0],
            linestyle=linestyles[group_order.index(group)],
            linewidth=4,
            alpha=alphas[group_order.index(group)]
        )

    # Customize and add legend
    ax.set_xlabel('Month', fontsize=24)
    ax.set_ylabel('Coverage Ratio', fontsize=24)
    ax.tick_params(axis='y', labelsize=22)
    ax.tick_params(axis='x', labelsize=22)
    ax.legend(loc='upper left', fontsize=18)
    ax.spines[['right', 'top']].set_visible(False)

    plt.tight_layout()
    # Save as PDF
    plt.savefig('figure_average_coverage.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory


def preprocess_combined_addicted_group(bootstrap_df):
    """Combine 'Mildly Addicted' and 'Severely Addicted' into one group called 'Addicted'."""
    # Replace group names
    bootstrap_df['addiction_group'] = bootstrap_df['addiction_group'].replace(
        {'Mildly Addicted': 'Addicted', 'Severely Addicted': 'Addicted'}
    )

    # Debugging: Check the replacement
    print("Unique groups after replacement:", bootstrap_df['addiction_group'].unique())
    print(bootstrap_df['addiction_group'].value_counts())

    # Group by month and addiction group, and calculate the mean for all numeric columns
    aggregated_df = bootstrap_df.groupby(
        ['month_chronological_order', 'addiction_group']
    ).mean().reset_index()

    # Debugging: Check the aggregated data
    print("Aggregated DataFrame (first rows):")
    print(aggregated_df.head())

    return aggregated_df



def plot_combined_coverage(bootstrap_df):
    """Plot combined coverage for Addicted and Non-Addicted groups."""
    # Preprocess data to combine "Mildly Addicted" and "Severely Addicted" into "Addicted"
    combined_data = preprocess_combined_addicted_group(bootstrap_df)

    # Pivot the data for plotting
    combined_pivot = combined_data.pivot(
        index='month_chronological_order',
        columns='addiction_group',
        values=['level_1_coverage_normalized',
                'level_2_coverage_normalized',
                'level_3_coverage_normalized']
    )

    # Debugging: Check pivot table columns
    print("Pivot table columns:", combined_pivot.columns)

    # Plot settings
    group_order = ['Non-Addicted', 'Addicted']
    colors = ['#22974F', '#25B1E8', '#E8335A']
    linestyles = ['--', '-']
    alphas = [0.40, 0.9]

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 6))

    for i, level in enumerate(['level_1_coverage_normalized',
                               'level_2_coverage_normalized',
                               'level_3_coverage_normalized']):
        for group in group_order:
            if group in combined_pivot[level].columns:  # Ensure group exists in the pivot table
                ax.plot(
                    combined_pivot.index,  # X-axis: Months
                    combined_pivot[level][group],  # Y-axis: Coverage Ratio
                    label=f'L{i + 1} {group}',  # Legend label
                    color=colors[i],  # Same color for each level
                    linestyle=linestyles[group_order.index(group)],  # Linestyle by group
                    linewidth=4,
                    alpha=alphas[group_order.index(group)]  # Different alpha for each group
                )

    # Customize the plot
    ax.set_xlabel('Month', fontsize=24)
    ax.set_ylabel('Coverage Ratio', fontsize=24)
    ax.tick_params(axis='y', labelsize=24)
    ax.tick_params(axis='x', labelsize=24)

    # Add legend
    ax.legend(loc='upper left', fontsize=18)
    ax.spines[['right', 'top']].set_visible(False)

    # Save as PDF
    plt.tight_layout()
    plt.savefig('figure_combined_coverage.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory
    
    
    
def plot_average_combined_coverage(bootstrap_df):
    """Plot average coverage across levels for Addicted and Non-Addicted groups."""
    # Preprocess data
    combined_data = preprocess_combined_addicted_group(bootstrap_df)

    # Compute the average coverage across levels
    combined_data['average_coverage_normalized'] = combined_data[
        ['level_1_coverage_normalized', 
         'level_2_coverage_normalized', 
         'level_3_coverage_normalized']
    ].mean(axis=1)

    # Pivot the data for plotting
    average_pivot = combined_data.pivot(
        index='month_chronological_order',
        columns='addiction_group',
        values='average_coverage_normalized'
    )

    # Debugging: Check pivot table columns
    print("Average pivot table columns:", average_pivot.columns)

    # Plot settings
    group_order = ['Non-Addicted', 'Addicted']
    colors = ['#2A5580', '#FF6F61']
    linestyles = ['--', '-']
    alphas = [0.55, 0.9]

    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 6))

    for group in group_order:
        if group in average_pivot.columns:  # Ensure group exists in the pivot table
            ax.plot(
                average_pivot.index,  # X-axis: Months
                average_pivot[group],  # Y-axis: Coverage Ratio
                label=group,  # Legend label
                color=colors[group_order.index(group)],  # Color by group
                linestyle=linestyles[group_order.index(group)],  # Linestyle by group
                linewidth=5,
                alpha=alphas[group_order.index(group)]  # Different alpha for each group
            )

    # Customize the plot
    ax.set_xlabel('Month', fontsize=24)
    ax.set_ylabel('Coverage Ratio', fontsize=24)
    ax.tick_params(axis='y', labelsize=24) 
    ax.tick_params(axis='x', labelsize=24) 

    # Add legend
    ax.legend(loc='upper left', fontsize=20)
    ax.spines[['right', 'top']].set_visible(False)

    # Save as PDF
    plt.tight_layout()
    plt.savefig('figure_average_combined_coverage.pdf', format='pdf', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to save memory
    
    
if __name__ == "__main__":
    merged_data = load_and_merge_data()
    coverage_ratios = calculate_coverage_ratios(merged_data)

    # Load bootstrap data
    bootstrap_data = pd.read_csv('bootstrap_normalized_results.csv')

    # Generate the plots
    plot_bootstrap_coverage(bootstrap_data)  # Figure 1
    plot_average_coverage(bootstrap_data)  # Figure 2
    plot_combined_coverage(bootstrap_data)  # Figure 3
    plot_average_combined_coverage(bootstrap_data)  # Figure 4