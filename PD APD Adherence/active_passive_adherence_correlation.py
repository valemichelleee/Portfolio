import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import os

EXCLUDED_PATIENTS = [
    'pat001', 'pat005', 'pat014', 'pat117', 'pat120', 'pat124', 'pat127', 'pat131', 
    'pat133', 'pat134', 'pat154', 'pat212', 'pat233', 'pat238', 'pat314', 'PAT402', 
    'pat403', 'pat407', 'PAT408', 'PAT411', 'pat412', 'PAT414'
]

DISEASES = ['MSA', 'PD', 'PSP']
DISEASE_COLORS = {'MSA': '#1f77b4', 'PD': '#ff7f0e', 'PSP': '#2ca02c'}

def load_adherence_data():
    """Load active and passive adherence data and merge them."""
    excluded_upper = [pat.upper() for pat in EXCLUDED_PATIENTS]
    
    active_file = 'active_monitoring/active_test_behavior_summary.csv'
    if not os.path.exists(active_file):
        print("ERROR: active_test_behavior_summary.csv not found.")
        print("Please run active_monitoring.py first.")
        return None
    
    active_df = pd.read_csv(active_file).rename(columns={'Week1': 'Active_Week1', 'Week8': 'Active_Week8'})
    
    week1_file = 'passive_monitoring/week1/week1_validity_rate.csv'
    week8_file = 'passive_monitoring/week8/week8_validity_rate.csv'
    
    if not os.path.exists(week1_file) or not os.path.exists(week8_file):
        print("ERROR: Passive monitoring validity rate files not found.")
        print("Please run passive_monitoring.py first.")
        return None
    
    week1_passive = pd.read_csv(week1_file)
    week8_passive = pd.read_csv(week8_file)
    
    week1_passive['Passive_Week1'] = week1_passive['valid_days_rate'] * 100
    week8_passive['Passive_Week8'] = week8_passive['valid_days_rate'] * 100
    
    passive_df = week1_passive[['PATID', 'Label', 'Passive_Week1']].merge(
        week8_passive[['PATID', 'Passive_Week8']], on='PATID', how='outer'
    )
    
    combined_df = active_df.merge(passive_df, on='PATID', how='outer')
    combined_df['Disease'] = combined_df['Disease'].fillna(combined_df['Label'])
    combined_df = combined_df.drop(columns=['Label'], errors='ignore')
    combined_df = combined_df[~combined_df['PATID'].isin(excluded_upper)]
    
    print("="*70)
    print("ACTIVE vs PASSIVE ADHERENCE CORRELATION ANALYSIS")
    print("="*70)
    print(f"\nExcluded {len(EXCLUDED_PATIENTS)} dropout patients: {EXCLUDED_PATIENTS}")
    print(f"Total patients in combined dataset: {len(combined_df)}")
    print(f"Patients with both Active Week 1 and Passive Week 1: {((combined_df['Active_Week1'].notna()) & (combined_df['Passive_Week1'].notna())).sum()}")
    print(f"Patients with both Active Week 8 and Passive Week 8: {((combined_df['Active_Week8'].notna()) & (combined_df['Passive_Week8'].notna())).sum()}")
    
    return combined_df


def calculate_correlations(x, y):
    """Calculate Spearman and Pearson correlations for two variables."""
    if len(x) < 3:
        return None
    spearman_r, spearman_p = spearmanr(x, y)
    pearson_r, pearson_p = pearsonr(x, y)
    return {
        'N': len(x),
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_Significant': spearman_p < 0.05,
        'Pearson_Significant': pearson_p < 0.05
    }

def overall_correlation_analysis(df):
    """Perform overall correlation analysis between active and passive adherence."""
    correlations = []
    
    for week_name, active_col, passive_col in [
        ('Week 1', 'Active_Week1', 'Passive_Week1'),
        ('Week 8', 'Active_Week8', 'Passive_Week8')
    ]:
        data = df[[active_col, passive_col]].dropna()
        corr = calculate_correlations(data[active_col], data[passive_col])
        if corr:
            correlations.append({'Week': week_name, **corr})
    
    df['Avg_Active'] = df[['Active_Week1', 'Active_Week8']].mean(axis=1)
    df['Avg_Passive'] = df[['Passive_Week1', 'Passive_Week8']].mean(axis=1)
    avg_data = df[['Avg_Active', 'Avg_Passive']].dropna()
    corr = calculate_correlations(avg_data['Avg_Active'], avg_data['Avg_Passive'])
    if corr:
        correlations.append({'Week': 'Average', **corr})
    
    return pd.DataFrame(correlations)


def disease_specific_correlation_analysis(df):
    """Perform disease-specific correlation analysis for each week."""
    disease_correlations = []
    
    for disease in DISEASES:
        disease_data = df[df['Disease'] == disease].copy()
        if len(disease_data) < 3:
            continue
        
        for week_name, active_col, passive_col in [
            ('Week 1', 'Active_Week1', 'Passive_Week1'),
            ('Week 8', 'Active_Week8', 'Passive_Week8')
        ]:
            data = disease_data[[active_col, passive_col]].dropna()
            corr = calculate_correlations(data[active_col], data[passive_col])
            if corr:
                disease_correlations.append({'Disease': disease, 'Week': week_name, **corr})
        
        disease_data['Avg_Active'] = disease_data[['Active_Week1', 'Active_Week8']].mean(axis=1)
        disease_data['Avg_Passive'] = disease_data[['Passive_Week1', 'Passive_Week8']].mean(axis=1)
        avg_data = disease_data[['Avg_Active', 'Avg_Passive']].dropna()
        corr = calculate_correlations(avg_data['Avg_Active'], avg_data['Avg_Passive'])
        if corr:
            disease_correlations.append({'Disease': disease, 'Week': 'Average', **corr})
    
    return pd.DataFrame(disease_correlations)


def plot_scatter_with_trend(ax, x_data, y_data, disease_col, xlabel, ylabel, title):
    """Create scatter plot with trend line and correlation stats."""
    plot_data = pd.DataFrame({
        'x': x_data,
        'y': y_data,
        'Disease': disease_col
    }).dropna()
    
    if len(plot_data) == 0:
        return
    
    for disease in DISEASES:
        disease_data = plot_data[plot_data['Disease'] == disease]
        if len(disease_data) > 0:
            ax.scatter(disease_data['x'], disease_data['y'], 
                      alpha=0.7, s=60, label=disease, color=DISEASE_COLORS[disease])
    
    if len(plot_data) > 2:
        z = np.polyfit(plot_data['x'], plot_data['y'], 1)
        p = np.poly1d(z)
        ax.plot(plot_data['x'].sort_values(), p(plot_data['x'].sort_values()), 
               "k--", alpha=0.8, linewidth=1.5)
        
        spear_r, spear_p = spearmanr(plot_data['x'], plot_data['y'])
        pear_r, pear_p = pearsonr(plot_data['x'], plot_data['y'])
        ax.text(0.05, 0.95, 
               f'Spearman r = {spear_r:.3f}, p = {spear_p:.3f}\nPearson r = {pear_r:.3f}, p = {pear_p:.3f}', 
               transform=ax.transAxes, verticalalignment='top', fontsize=12,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)

def create_correlation_plots(df):
    """Create scatter plots showing correlations between active and passive adherence."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Active vs Passive Adherence Correlations', fontsize=16, fontweight='bold')
    
    plot_scatter_with_trend(
        axes[0, 0], df['Active_Week1'], df['Passive_Week1'], df['Disease'],
        'Active Adherence Week 1 (%)', 'Passive Adherence Week 1 (%)', 
        'Week 1: Active vs Passive Adherence'
    )
    
    plot_scatter_with_trend(
        axes[0, 1], df['Active_Week8'], df['Passive_Week8'], df['Disease'],
        'Active Adherence Week 8 (%)', 'Passive Adherence Week 8 (%)', 
        'Week 8: Active vs Passive Adherence'
    )
    
    df['Avg_Active'] = df[['Active_Week1', 'Active_Week8']].mean(axis=1)
    df['Avg_Passive'] = df[['Passive_Week1', 'Passive_Week8']].mean(axis=1)
    
    plot_scatter_with_trend(
        axes[1, 0], df['Avg_Active'], df['Avg_Passive'], df['Disease'],
        'Average Active Adherence (%)', 'Average Passive Adherence (%)', 
        'Average: Active vs Passive Adherence'
    )
    
    df['Active_Change'] = df['Active_Week8'] - df['Active_Week1']
    df['Passive_Change'] = df['Passive_Week8'] - df['Passive_Week1']
    
    plot_scatter_with_trend(
        axes[1, 1], df['Active_Change'], df['Passive_Change'], df['Disease'],
        'Change in Active Adherence (Week8 - Week1)', 
        'Change in Passive Adherence (Week8 - Week1)', 
        'Changes: Active vs Passive Adherence'
    )
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('active_passive_adherence_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_disease_week(ax, disease_data, active_col, passive_col, disease, week_title):
    """Plot correlation for a specific disease and week."""
    data = disease_data[[active_col, passive_col]].dropna()
    if len(data) < 2:
        return
    
    ax.scatter(data[active_col], data[passive_col], 
              color=DISEASE_COLORS[disease], alpha=0.7, s=80)
    
    if len(data) > 2:
        z = np.polyfit(data[active_col], data[passive_col], 1)
        p = np.poly1d(z)
        ax.plot(data[active_col].sort_values(), p(data[active_col].sort_values()), 
               "k--", alpha=0.8, linewidth=2)
        
        spear_r, spear_p = spearmanr(data[active_col], data[passive_col])
        pear_r, pear_p = pearsonr(data[active_col], data[passive_col])
        ax.text(0.05, 0.95, 
               f'n = {len(data)}\nSpearman r = {spear_r:.3f}\np = {spear_p:.3f}\nPearson r = {pear_r:.3f}\np = {pear_p:.3f}', 
               transform=ax.transAxes, verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel(f'Active Adherence {week_title} (%)', fontsize=14)
    ax.set_ylabel(f'Passive Adherence {week_title} (%)', fontsize=14)
    ax.set_title(f'{disease} - {week_title}', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='both', which='major', labelsize=12)

def create_disease_specific_plots(df):
    """Create separate correlation plots for each disease."""
    fig, axes = plt.subplots(len(DISEASES), 2, figsize=(12, 15))
    fig.suptitle('Disease-Specific Active vs Passive Adherence Correlations', 
                fontsize=16, fontweight='bold')
    
    for i, disease in enumerate(DISEASES):
        disease_data = df[df['Disease'] == disease].copy()
        plot_disease_week(axes[i, 0], disease_data, 'Active_Week1', 'Passive_Week1', 
                         disease, 'Week 1')
        plot_disease_week(axes[i, 1], disease_data, 'Active_Week8', 'Passive_Week8', 
                         disease, 'Week 8')
    
    plt.tight_layout()
    plt.savefig('disease_specific_active_passive_adherence_correlations.png', 
               dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the Active vs Passive Adherence correlation analysis."""
    print("\nStarting Active vs Passive Adherence Correlation Analysis...")
    
    df = load_adherence_data()
    if df is None:
        return
    
    print("\n" + "="*70)
    print("OVERALL CORRELATION ANALYSIS")
    print("="*70)
    
    overall_correlations = overall_correlation_analysis(df)
    print("\nOverall Correlations (Active vs Passive Adherence):")
    print(overall_correlations.to_string(index=False))
    overall_correlations.to_csv('active_passive_correlations_overall.csv', index=False)
    
    print("\n" + "="*70)
    print("DISEASE-SPECIFIC CORRELATION ANALYSIS")
    print("="*70)
    
    disease_correlations = disease_specific_correlation_analysis(df)
    if len(disease_correlations) > 0:
        print("\nDisease-Specific Correlations:")
        print(disease_correlations.to_string(index=False))
        disease_correlations.to_csv('active_passive_correlations_by_disease.csv', index=False)
    else:
        print("Insufficient data for disease-specific correlations.")
    
    print("\n" + "="*70)
    print("GENERATING CORRELATION PLOTS")
    print("="*70)
    
    create_correlation_plots(df)
    create_disease_specific_plots(df)
    
    df.to_csv('active_passive_adherence_combined.csv', index=False)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    print("\nKey Findings:")
    
    for idx, row in overall_correlations.iterrows():
        significance = "significant" if row['Spearman_Significant'] else "not significant"
        print(f"{idx + 1}. {row['Week']}: r = {row['Spearman_r']:.3f}, p = {row['Spearman_p']:.3f} ({significance}), N = {row['N']}")
    
    print(f"\nFiles generated:")
    print(f"- active_passive_correlations_overall.csv: Overall correlation results")
    print(f"- active_passive_correlations_by_disease.csv: Disease-specific correlation results")
    print(f"- active_passive_adherence_combined.csv: Combined dataset")
    print(f"- active_passive_adherence_correlations.png: Main correlation plots")
    print(f"- disease_specific_active_passive_adherence_correlations.png: Disease-specific plots")
    print(f"\nAnalysis completed successfully!")

if __name__ == '__main__':
    main()

