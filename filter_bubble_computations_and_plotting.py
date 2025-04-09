import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Load Data ###
monthly_user_data = pd.read_csv('path/to/monthly_user_data.csv')
addiction_status = pd.read_csv('path/to/addiction_data.csv')

# Add chronological month order to `monthly_user_data`
monthly_user_data['month_chronological_order'] = monthly_user_data['window_id'] + 1

# Ensure month alignment in `addiction_status`
if 'month' not in addiction_status.columns:
    addiction_status['month_chronological_order'] = addiction_status['month_chronological_order']

# Merge dataframes on `user_id` and `month_chronological_order`
merged_df = pd.merge(
    monthly_user_data,
    addiction_status,
    on=['user_id', 'month_chronological_order'],
    how='inner'
)

# Use merged data for further processing
monthly_user_data_ratio = merged_df

### Helper Functions ###

def calculate_quantiles(df):
    """Calculate median filter bubble thresholds for each level."""
    return {
        'level_1': df['level_1_coverage'].quantile(0.5),
        'level_2': df['level_2_coverage'].quantile(0.5),
        'level_3': df['level_3_coverage'].quantile(0.5),
    }

def calculate_filter_bubble_with_bootstrap(df, group_name, quantiles, group_size, num_bootstrap=1000):
    """Calculate filter bubble percentages for each group with bootstrapping."""
    fb_list = []
    for month in df['month_chronological_order'].unique():
        df_month = df[df['month_chronological_order'] == month]
        
        # Initialize counters for each filter bubble level
        fb_l1_counts, fb_l2_counts, fb_l3_counts = [], [], []
        
        # Bootstrap sampling
        for _ in range(num_bootstrap):
            sampled_df = df_month.sample(n=group_size, replace=True)
            
            fb_l1 = sampled_df['level_1_coverage'] < quantiles['level_1']
            fb_l2 = (sampled_df['level_2_coverage'] < quantiles['level_2']) & (~fb_l1)
            fb_l3 = (sampled_df['level_3_coverage'] < quantiles['level_3']) & (~fb_l1) & (~fb_l2)
            
            fb_l1_counts.append(fb_l1.sum())
            fb_l2_counts.append(fb_l2.sum())
            fb_l3_counts.append(fb_l3.sum())
        
        # Calculate mean proportions
        fb_pct_l1 = np.mean(fb_l1_counts) / group_size
        fb_pct_l2 = np.mean(fb_l2_counts) / group_size
        fb_pct_l3 = np.mean(fb_l3_counts) / group_size
        
        fb_list.append(
            pd.Series([fb_pct_l1, fb_pct_l2, fb_pct_l3],
                      index=[f'Level 1 - {group_name}', f'Level 2 - {group_name}', f'Level 3 - {group_name}'],
                      name=month)
        )
    
    return pd.concat(fb_list, axis=1).T

### Filter Data by Groups ###
nonaddicted_df = monthly_user_data_ratio[
    monthly_user_data_ratio['user_id'].isin(
        addiction_status[addiction_status['preds_3_label_criteria'] == 0]['user_id']
    )
]

soft_addicted_df = monthly_user_data_ratio[
    monthly_user_data_ratio['user_id'].isin(
        addiction_status[addiction_status['preds_3_label_criteria'] == 1]['user_id']
    )
]

hard_addicted_df = monthly_user_data_ratio[
    monthly_user_data_ratio['user_id'].isin(
        addiction_status[addiction_status['preds_3_label_criteria'] == 2]['user_id']
    )
]

# Calculate the smallest group size for bootstrapping
smallest_group_size = min(
    nonaddicted_df['user_id'].nunique(),
    soft_addicted_df['user_id'].nunique(),
    hard_addicted_df['user_id'].nunique()
)

### Calculate Quantiles and Filter Bubble Data ###
q5_nonaddicted = calculate_quantiles(nonaddicted_df)
q5_soft_addicted = calculate_quantiles(soft_addicted_df)
q5_hard_addicted = calculate_quantiles(hard_addicted_df)

filter_bubble_nonaddicted = calculate_filter_bubble_with_bootstrap(
    nonaddicted_df, "Non-Addicted", q5_nonaddicted, group_size=smallest_group_size
)

filter_bubble_soft_addicted = calculate_filter_bubble_with_bootstrap(
    soft_addicted_df, "Mildly Addicted", q5_soft_addicted, group_size=smallest_group_size
)

filter_bubble_hard_addicted = calculate_filter_bubble_with_bootstrap(
    hard_addicted_df, "Severely Addicted", q5_hard_addicted, group_size=smallest_group_size
)

# Combine results for plotting
combined_df = pd.concat([filter_bubble_nonaddicted, filter_bubble_soft_addicted, filter_bubble_hard_addicted], axis=1)
combined_df.index = pd.to_numeric(combined_df.index)
combined_df.sort_index(inplace=True)

### Plot 1: Non-Addicted, Mildly Addicted, Severely Addicted ###
fig, ax = plt.subplots(figsize=(15, 8))

x = np.arange(len(combined_df.index)) + 1
bar_width = 0.25

colors = ['#22974F', '#76c8e9', '#e9768f']

# Plot Non-Addicted
ax.bar(x - bar_width, combined_df['Level 1 - Non-Addicted'], width=bar_width, color=colors[0], label='Level 1 - Non-Addicted', alpha=0.3)
ax.bar(x - bar_width, combined_df['Level 2 - Non-Addicted'], width=bar_width, color=colors[1], bottom=combined_df['Level 1 - Non-Addicted'], alpha=0.3)
ax.bar(x - bar_width, combined_df['Level 3 - Non-Addicted'], width=bar_width, color=colors[2], bottom=combined_df['Level 1 - Non-Addicted'] + combined_df['Level 2 - Non-Addicted'], alpha=0.3)

# Plot Mildly Addicted
ax.bar(x, combined_df['Level 1 - Mildly Addicted'], width=bar_width, color=colors[0], label='Level 1 - Mildly Addicted', alpha=0.55)
ax.bar(x, combined_df['Level 2 - Mildly Addicted'], width=bar_width, color=colors[1], bottom=combined_df['Level 1 - Mildly Addicted'], alpha=0.55)
ax.bar(x, combined_df['Level 3 - Mildly Addicted'], width=bar_width, color=colors[2], bottom=combined_df['Level 1 - Mildly Addicted'] + combined_df['Level 2 - Mildly Addicted'], alpha=0.55)

# Plot Severely Addicted
ax.bar(x + bar_width, combined_df['Level 1 - Severely Addicted'], width=bar_width, color=colors[0], label='Level 1 - Severely Addicted', alpha=0.99)
ax.bar(x + bar_width, combined_df['Level 2 - Severely Addicted'], width=bar_width, color=colors[1], bottom=combined_df['Level 1 - Severely Addicted'], alpha=0.99)
ax.bar(x + bar_width, combined_df['Level 3 - Severely Addicted'], width=bar_width, color=colors[2], bottom=combined_df['Level 1 - Severely Addicted'] + combined_df['Level 2 - Severely Addicted'], alpha=0.99)

ax.set_xlabel('Month', fontsize=16)
ax.set_ylabel('Ratio of Users in Filter Bubble', fontsize=16)
ax.set_xticks(x)
ax.legend(
    handles=[
        plt.Line2D([0], [0], color='#22974F', lw=6, alpha=1, label='Level 1'),
        plt.Line2D([0], [0], color='#76c8e9', lw=6, alpha=1, label='Level 2'),
        plt.Line2D([0], [0], color='#e9768f', lw=6, alpha=1, label='Level 3'),
        plt.Line2D([0], [0], color='black', lw=0, label='Non-Addicted', marker='s', markersize=10, alpha=0.3),
        plt.Line2D([0], [0], color='black', lw=0, label='Mildly Addicted', marker='s', markersize=10, alpha=0.55),
        plt.Line2D([0], [0], color='black', lw=0, label='Severely Addicted', marker='s', markersize=10, alpha=0.99),
    ],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.2),
    ncol=2,
    fontsize=12,
    title="Levels and Groups"
)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('filter_bubble_ratios_all_groups.pdf')

### Plot 2: Non-Addicted vs Combined Addicted ###
addicted = combined_df.filter(like='Mildly Addicted').add(combined_df.filter(like='Severely Addicted'), fill_value=0)

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(x - bar_width / 2, combined_df['Level 1 - Non-Addicted'], width=bar_width, color=colors[0], label='Level 1 - Non-Addicted', alpha=0.5)
ax.bar(x - bar_width / 2, combined_df['Level 2 - Non-Addicted'], width=bar_width, color=colors[1], bottom=combined_df['Level 1 - Non-Addicted'], alpha=0.5)
ax.bar(x - bar_width / 2, combined_df['Level 3 - Non-Addicted'], width=bar_width, color=colors[2], bottom=combined_df['Level 1 - Non-Addicted'] + combined_df['Level 2 - Non-Addicted'], alpha=0.5)

ax.bar(x + bar_width / 2, addicted.iloc[:, 0], width=bar_width, color=colors[0], label='Level 1 - Addicted', alpha=0.99)
ax.bar(x + bar_width / 2, addicted.iloc[:, 1], width=bar_width, color=colors[1], bottom=addicted.iloc[:, 0], alpha=0.99)
ax.bar(x + bar_width / 2, addicted.iloc[:, 2], width=bar_width, color=colors[2], bottom=addicted.iloc[:, 0] + addicted.iloc[:, 1], alpha=0.99)

ax.set_xlabel('Month', fontsize=16)
ax.set_ylabel('Ratio of Users in Filter Bubble', fontsize=16)
ax.set_xticks(x)
ax.legend(
    handles=[
        plt.Line2D([0], [0], color='#22974F', lw=6, alpha=1, label='Level 1'),
        plt.Line2D([0], [0], color='#76c8e9', lw=6, alpha=1, label='Level 2'),
        plt.Line2D([0], [0], color='#e9768f', lw=6, alpha=1, label='Level 3'),
        plt.Line2D([0], [0], color='black', lw=0, label='Non-Addicted', marker='s', markersize=10, alpha=0.5),
        plt.Line2D([0], [0], color='black', lw=0, label='Addicted', marker='s', markersize=10, alpha=0.99),
    ],
    loc='upper center',
    bbox_to_anchor=(0.5, 1.2),
    ncol=2,
    fontsize=12,
    title="Levels and Groups"
)
ax.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('filter_bubble_ratios_combined_addicted.pdf')