import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
import os

data_dir = os.path.join(os.path.dirname(__file__), "../data")
output_dir = os.path.dirname(__file__)

EXCLUDED_PATIENTS = [
    'pat001', 'pat005', 'pat014', 'pat117', 'pat120', 'pat124', 'pat127', 'pat131', 
    'pat133', 'pat134', 'pat154', 'pat212', 'pat233', 'pat238', 'pat314', 'PAT402', 
    'pat403', 'pat407', 'PAT408', 'PAT411', 'pat412', 'PAT414'
]

DISEASES = ['MSA', 'PD', 'PSP']
DISEASE_COLORS = {'MSA': '#1f77b4', 'PD': '#ff7f0e', 'PSP': '#2ca02c'}


def load_and_prepare_data():
    """Load passive adherence data and SUS scores, merge them for analysis."""
    
    excluded_upper = [pat.upper() for pat in EXCLUDED_PATIENTS]
    
    week_files = {
        'week1': os.path.join(output_dir, 'week1', 'week1_detailed_metrics.csv'),
        'week8': os.path.join(output_dir, 'week8', 'week8_detailed_metrics.csv')
    }
    
    if not all(os.path.exists(f) for f in week_files.values()):
        print("ERROR: Passive monitoring detailed metrics files not found. Please run passive_monitoring.py first.")
        return None
    
    week_dfs = {week: pd.read_csv(file) for week, file in week_files.items()}
    
    all_patients = set(week_dfs['week1']['PATID'].unique()) | set(week_dfs['week8']['PATID'].unique())
    
    adherence_data = []
    for patient in all_patients:
        patient_data = {'PATID': patient}
        
        for week_num, week_key in [(1, 'week1'), (8, 'week8')]:
            week_patient = week_dfs[week_key][week_dfs[week_key]['PATID'] == patient]
            if len(week_patient) > 0:
                patient_data[f'Week{week_num}_PassiveAdherence'] = week_patient['valid_days_rate'].iloc[0] * 100
                if 'Disease' not in patient_data:
                    patient_data['Disease'] = week_patient['Label'].iloc[0]
            else:
                patient_data[f'Week{week_num}_PassiveAdherence'] = np.nan
        
        adherence_data.append(patient_data)
    
    adherence_df = pd.DataFrame(adherence_data)
    
    visit_dfs = {
        2: pd.read_excel(os.path.join(data_dir, 'RCT_Clinical_Data.xlsx'), sheet_name='Visit2'),
        5: pd.read_excel(os.path.join(data_dir, 'RCT_Clinical_Data.xlsx'), sheet_name='Visit5')
    }
    
    visit_dfs = {k: df[~df['PATID'].isin(excluded_upper)] for k, df in visit_dfs.items()}
    
    print(f"Excluded {len(EXCLUDED_PATIENTS)} dropout patients from SUS analysis: {EXCLUDED_PATIENTS}")
    
    sus_scores = {
        'SUS_Week1': visit_dfs[2][['PATID', 'SUS']].rename(columns={'SUS': 'SUS_Week1'}),
        'SUS_Week8': visit_dfs[5][['PATID', 'SUS']].rename(columns={'SUS': 'SUS_Week8'})
    }
    
    combined_df = adherence_df
    for sus_df in sus_scores.values():
        combined_df = combined_df.merge(sus_df, on='PATID', how='left')
    
    print("="*70)
    print("PASSIVE ADHERENCE vs SUS CORRELATION ANALYSIS")
    print("="*70)
    
    print(f"Patients in Visit2 SUS data after exclusions: {len(visit_dfs[2])}")
    print(f"Patients in Visit5 SUS data after exclusions: {len(visit_dfs[5])}")
    print(f"Total patients in passive adherence data: {len(adherence_df)}")
    print(f"Patients with Week 1 SUS scores: {combined_df['SUS_Week1'].notna().sum()}")
    print(f"Patients with Week 8 SUS scores: {combined_df['SUS_Week8'].notna().sum()}")
    print(f"Patients with both Week 1 passive adherence and SUS: {((combined_df['Week1_PassiveAdherence'].notna()) & (combined_df['SUS_Week1'].notna())).sum()}")
    print(f"Patients with both Week 8 passive adherence and SUS: {((combined_df['Week8_PassiveAdherence'].notna()) & (combined_df['SUS_Week8'].notna())).sum()}")
    
    return combined_df

def calculate_correlation(df, col1, col2, comparison_name):
    """Calculate Spearman correlation for two columns."""
    data = df[[col1, col2]].dropna()
    if len(data) <= 2:
        return None
    
    spearman_r, spearman_p = spearmanr(data[col1], data[col2])
    return {
        'Comparison': comparison_name,
        'N': len(data),
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'Spearman_Significant': spearman_p < 0.05
    }

def correlation_analysis(df):
    """Perform correlation analysis between passive adherence and SUS scores."""
    
    correlations_config = [
        ('Week1_PassiveAdherence', 'SUS_Week1', 'Week1 Passive Adherence vs Week1 SUS'),
        ('Week8_PassiveAdherence', 'SUS_Week8', 'Week8 Passive Adherence vs Week8 SUS'),
        ('Week1_PassiveAdherence', 'SUS_Week8', 'Week1 Passive Adherence vs Week8 SUS'),
        ('Week8_PassiveAdherence', 'SUS_Week1', 'Week8 Passive Adherence vs Week1 SUS'),
    ]
    
    correlations = [calculate_correlation(df, col1, col2, name) 
                    for col1, col2, name in correlations_config]
    
    df['Avg_PassiveAdherence'] = df[['Week1_PassiveAdherence', 'Week8_PassiveAdherence']].mean(axis=1)
    df['Avg_SUS'] = df[['SUS_Week1', 'SUS_Week8']].mean(axis=1)
    
    avg_corr = calculate_correlation(df, 'Avg_PassiveAdherence', 'Avg_SUS', 
                                     'Average Passive Adherence vs Average SUS')
    if avg_corr:
        correlations.append(avg_corr)
    
    return pd.DataFrame([c for c in correlations if c is not None])

def disease_specific_correlations(df):
    """Analyze correlations within each disease group."""
    
    disease_correlations = []
    week_configs = [
        ('Week1_PassiveAdherence', 'SUS_Week1', 'Week1 Passive Adherence vs Week1 SUS'),
        ('Week8_PassiveAdherence', 'SUS_Week8', 'Week8 Passive Adherence vs Week8 SUS')
    ]
    
    for disease in DISEASES:
        disease_data = df[df['Disease'] == disease].copy()
        
        if len(disease_data) < 3:
            continue
        
        for col1, col2, comparison in week_configs:
            try:
                corr = calculate_correlation(disease_data, col1, col2, comparison)
                if corr:
                    corr['Disease'] = disease
                    disease_correlations.append(corr)
            except:
                pass
    
    return pd.DataFrame(disease_correlations)

def plot_scatter_with_trend(ax, data, x_col, y_col, title, xlabel, ylabel):
    """Create scatter plot with trend line and correlation info."""
    if len(data) == 0:
        return
    
    for disease in DISEASES:
        disease_data = data[data['Disease'] == disease]
        if len(disease_data) > 0:
            ax.scatter(disease_data[x_col], disease_data[y_col], 
                      alpha=0.7, s=60, label=disease, color=DISEASE_COLORS[disease])
    
    if len(data) > 2:
        z = np.polyfit(data[x_col], data[y_col], 1)
        p = np.poly1d(z)
        ax.plot(data[x_col].sort_values(), p(data[x_col].sort_values()), 
               "k--", alpha=0.8, linewidth=1.5)
        
        spear_r, spear_p = spearmanr(data[x_col], data[y_col])
        ax.text(0.05, 0.95, f'r = {spear_r:.3f}, p = {spear_p:.3f}', 
               transform=ax.transAxes, verticalalignment='top', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)

def create_correlation_plots(df):
    """Create scatter plots showing correlations between passive adherence and SUS scores."""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Passive Monitoring Adherence vs System Usability Scale (SUS) Scores', 
                 fontsize=14, fontweight='bold')
    
    week1_data = df[['Week1_PassiveAdherence', 'SUS_Week1', 'Disease']].dropna()
    plot_scatter_with_trend(axes[0], week1_data, 'Week1_PassiveAdherence', 'SUS_Week1',
                           'Week 1: Passive Adherence vs SUS', 
                           'Week 1 Passive Adherence (%)', 'Week 1 SUS Score')
    
    week8_data = df[['Week8_PassiveAdherence', 'SUS_Week8', 'Disease']].dropna()
    plot_scatter_with_trend(axes[1], week8_data, 'Week8_PassiveAdherence', 'SUS_Week8',
                           'Week 8: Passive Adherence vs SUS', 
                           'Week 8 Passive Adherence (%)', 'Week 8 SUS Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'passive_adherence_sus_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_disease_week(ax, data, x_col, y_col, disease, week_num):
    """Plot single disease-week correlation."""
    week_data = data[[x_col, y_col]].dropna()
    
    if len(week_data) > 1:
        ax.scatter(week_data[x_col], week_data[y_col], 
                  color=DISEASE_COLORS[disease], alpha=0.7, s=80)
        
        if len(week_data) > 2:
            z = np.polyfit(week_data[x_col], week_data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(week_data[x_col].sort_values(), p(week_data[x_col].sort_values()), 
                   "k--", alpha=0.8, linewidth=2)
            
            spear_r, spear_p = spearmanr(week_data[x_col], week_data[y_col])
            ax.text(0.05, 0.95, f'r = {spear_r:.3f}\np = {spear_p:.3f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel(f'Week {week_num} Passive Adherence (%)', fontsize=14)
    ax.set_ylabel(f'Week {week_num} SUS Score', fontsize=14)
    ax.set_title(f'{disease} - Week {week_num}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='both', which='major', labelsize=14)

def create_disease_specific_plots(df):
    """Create separate correlation plots for each disease."""
    
    fig, axes = plt.subplots(len(DISEASES), 2, figsize=(12, 15))
    fig.suptitle('Disease-Specific Passive Adherence vs SUS Correlations', 
                 fontsize=14, fontweight='bold')
    
    for i, disease in enumerate(DISEASES):
        disease_data = df[df['Disease'] == disease].copy()
        plot_disease_week(axes[i, 0], disease_data, 'Week1_PassiveAdherence', 'SUS_Week1', disease, 1)
        plot_disease_week(axes[i, 1], disease_data, 'Week8_PassiveAdherence', 'SUS_Week8', disease, 8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'disease_specific_passive_adherence_sus_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_stats(series):
    """Calculate descriptive statistics for a series."""
    if len(series) == 0:
        return None
    
    return {
        'N': len(series),
        'Mean': series.mean(),
        'Std': series.std(),
        'Median': series.median(),
        'Min': series.min(),
        'Max': series.max(),
        'Q25': series.quantile(0.25),
        'Q75': series.quantile(0.75)
    }

def passive_adherence_descriptive_stats(df):
    """Generate descriptive statistics for passive adherence scores."""
    
    print("\n" + "="*50)
    print("PASSIVE ADHERENCE DESCRIPTIVE STATISTICS")
    print("="*50)
    
    adherence_stats = {}
    for week_col in ['Week1_PassiveAdherence', 'Week8_PassiveAdherence']:
        stats = calculate_stats(df[week_col].dropna())
        if stats:
            adherence_stats[week_col] = stats
    
    disease_adherence_stats = {}
    for disease in DISEASES:
        disease_data = df[df['Disease'] == disease]
        disease_adherence_stats[disease] = {}
        
        for week_num, week_col in [(1, 'Week1_PassiveAdherence'), (8, 'Week8_PassiveAdherence')]:
            stats = calculate_stats(disease_data[week_col].dropna())
            if stats:
                disease_adherence_stats[disease][f'Week{week_num}'] = {
                    k: stats[k] for k in ['N', 'Mean', 'Std', 'Median']
                }
    
    print("\nOverall Passive Adherence Statistics:")
    for week, stats in adherence_stats.items():
        print(f"\n{week}:")
        print(f"  N: {stats['N']}")
        print(f"  Mean: {stats['Mean']:.2f}% ± {stats['Std']:.2f}%")
        print(f"  Median: {stats['Median']:.2f}% (Q25: {stats['Q25']:.2f}%, Q75: {stats['Q75']:.2f}%)")
        print(f"  Range: {stats['Min']:.2f}% - {stats['Max']:.2f}%")
    
    print("\nPassive Adherence Statistics by Disease:")
    for disease, disease_stats in disease_adherence_stats.items():
        print(f"\n{disease}:")
        for week, stats in disease_stats.items():
            print(f"  {week}: N={stats['N']}, Mean={stats['Mean']:.2f}%±{stats['Std']:.2f}%, Median={stats['Median']:.2f}%")
    
    return adherence_stats, disease_adherence_stats

def print_section(title):
    """Print formatted section header."""
    print("\n" + "="*50)
    print(title)
    print("="*50)

def print_correlation_summary(correlations, comparison_name):
    """Print formatted correlation result."""
    corr = correlations[correlations['Comparison'] == comparison_name]
    if len(corr) > 0:
        row = corr.iloc[0]
        significance = "significant" if row['Spearman_Significant'] else "not significant"
        return f"r = {row['Spearman_r']:.3f}, p = {row['Spearman_p']:.3f} ({significance})"
    return None

def main():
    """Main function to run the Passive Adherence vs SUS correlation analysis."""
    
    print("\nStarting Passive Adherence vs SUS Correlation Analysis...")
    
    df = load_and_prepare_data()
    if df is None:
        return
    
    passive_adherence_descriptive_stats(df)
    
    print_section("OVERALL CORRELATION ANALYSIS")
    overall_correlations = correlation_analysis(df)
    print("\nOverall Correlations (Passive Adherence vs SUS):")
    print(overall_correlations.to_string(index=False))
    overall_correlations.to_csv(os.path.join(output_dir, 'passive_sus_correlations_overall.csv'), index=False)
    
    print_section("DISEASE-SPECIFIC CORRELATION ANALYSIS")
    disease_correlations = disease_specific_correlations(df)
    if len(disease_correlations) > 0:
        print("\nDisease-Specific Correlations:")
        print(disease_correlations.to_string(index=False))
        disease_correlations.to_csv(os.path.join(output_dir, 'passive_sus_correlations_by_disease.csv'), index=False)
    else:
        print("Insufficient data for disease-specific correlations.")
    
    print_section("GENERATING CORRELATION PLOTS")
    create_correlation_plots(df)
    create_disease_specific_plots(df)
    
    df.to_csv(os.path.join(output_dir, 'passive_adherence_with_sus_scores.csv'), index=False)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE - SUMMARY")
    print("="*70)
    
    print("\nKey Findings:")
    findings = [
        ('Week1 Passive Adherence vs Week1 SUS', 'Week 1 Passive Adherence vs Week 1 SUS'),
        ('Week8 Passive Adherence vs Week8 SUS', 'Week 8 Passive Adherence vs Week 8 SUS'),
        ('Average Passive Adherence vs Average SUS', 'Average Passive Adherence vs Average SUS')
    ]
    
    for i, (comparison, label) in enumerate(findings, 1):
        summary = print_correlation_summary(overall_correlations, comparison)
        if summary:
            print(f"{i}. {label}: {summary}")
    
    print(f"\nFiles generated:")
    print(f"- passive_sus_correlations_overall.csv: Overall correlation results")
    print(f"- passive_sus_correlations_by_disease.csv: Disease-specific correlation results")
    print(f"- passive_adherence_with_sus_scores.csv: Enhanced dataset with SUS scores")
    print(f"- passive_adherence_sus_correlations.png: Main correlation plots")
    print(f"- disease_specific_passive_adherence_sus_correlations.png: Disease-specific plots")
    
    print(f"\nAnalysis completed successfully!")

if __name__ == '__main__':
    main()
