import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
OUTPUT_DIR = SCRIPT_DIR

EXCLUDED_PATIENTS = [
    'pat001', 'pat005', 'pat014', 'pat117', 'pat120', 'pat124', 'pat127', 'pat131', 
    'pat133', 'pat134', 'pat154', 'pat212', 'pat233', 'pat238', 'pat314', 'PAT402', 
    'pat403', 'pat407', 'PAT408', 'PAT411', 'pat412', 'PAT414'
]

DISEASES = ['MSA', 'PD', 'PSP']
DISEASE_COLORS = {'MSA': '#1f77b4', 'PD': '#ff7f0e', 'PSP': '#2ca02c'}

def load_and_prepare_data():
    """Load adherence data and SUS scores, merge them for analysis."""
    excluded_patients_upper = [pat.upper() for pat in EXCLUDED_PATIENTS]
    
    adherence_file = os.path.join(OUTPUT_DIR, 'active_test_behavior_summary.csv')
    if not os.path.exists(adherence_file):
        print(f"ERROR: active_test_behavior_summary.csv not found at {adherence_file}")
        print("Please run active_monitoring.py first.")
        return None
    
    adherence_df = pd.read_csv(adherence_file)
    
    clinical_data_file = os.path.join(DATA_DIR, 'RCT_Clinical_Data.xlsx')
    if not os.path.exists(clinical_data_file):
        print(f"ERROR: RCT_Clinical_Data.xlsx not found at {clinical_data_file}")
        return None
    
    visits = {
        'Visit2': 'SUS_Week1',
        'Visit5': 'SUS_Week8'
    }
    
    combined_df = adherence_df.copy()
    for sheet, col_name in visits.items():
        visit_df = pd.read_excel(clinical_data_file, sheet_name=sheet)
        visit_df = visit_df[~visit_df['PATID'].isin(excluded_patients_upper)]
        sus_data = visit_df[['PATID', 'SUS']].rename(columns={'SUS': col_name})
        combined_df = combined_df.merge(sus_data, on='PATID', how='left')
    
    print(f"Excluded {len(EXCLUDED_PATIENTS)} dropout patients from SUS analysis: {EXCLUDED_PATIENTS}")
    print("=" * 70)
    print("ACTIVE ADHERENCE vs SUS CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"Total patients in adherence data: {len(adherence_df)}")
    print(f"Patients with Week 1 SUS scores: {combined_df['SUS_Week1'].notna().sum()}")
    print(f"Patients with Week 8 SUS scores: {combined_df['SUS_Week8'].notna().sum()}")
    print(f"Patients with both adherence and Week 1 SUS: {(combined_df['Week1'].notna() & combined_df['SUS_Week1'].notna()).sum()}")
    print(f"Patients with both adherence and Week 8 SUS: {(combined_df['Week8'].notna() & combined_df['SUS_Week8'].notna()).sum()}")
    
    return combined_df

def calculate_correlation(data1, data2):
    """Calculate Spearman and Pearson correlations between two series."""
    if len(data1) < 3:
        return None
    spearman_r, spearman_p = spearmanr(data1, data2)
    pearson_r, pearson_p = pearsonr(data1, data2)
    return {
        'N': len(data1),
        'Spearman_r': spearman_r,
        'Spearman_p': spearman_p,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Spearman_Significant': spearman_p < 0.05,
        'Pearson_Significant': pearson_p < 0.05
    }

def correlation_analysis(df):
    """Perform correlation analysis between adherence and SUS scores."""
    df['Avg_Adherence'] = df[['Week1', 'Week8']].mean(axis=1)
    df['Avg_SUS'] = df[['SUS_Week1', 'SUS_Week8']].mean(axis=1)
    
    comparisons = [
        ('Week1 Adherence vs Week1 SUS', 'Week1', 'SUS_Week1'),
        ('Week8 Adherence vs Week8 SUS', 'Week8', 'SUS_Week8'),
        ('Week1 Adherence vs Week8 SUS', 'Week1', 'SUS_Week8'),
        ('Week8 Adherence vs Week1 SUS', 'Week8', 'SUS_Week1'),
        ('Average Adherence vs Average SUS', 'Avg_Adherence', 'Avg_SUS')
    ]
    
    correlations = []
    for name, col1, col2 in comparisons:
        data = df[[col1, col2]].dropna()
        corr_result = calculate_correlation(data[col1], data[col2])
        if corr_result:
            corr_result['Comparison'] = name
            correlations.append(corr_result)
    
    return pd.DataFrame(correlations)

def disease_specific_correlations(df):
    """Analyze correlations within each disease group."""
    week_comparisons = [
        ('Week1 Adherence vs Week1 SUS', 'Week1', 'SUS_Week1'),
        ('Week8 Adherence vs Week8 SUS', 'Week8', 'SUS_Week8')
    ]
    
    disease_correlations = []
    for disease in DISEASES:
        disease_data = df[df['Disease'] == disease].copy()
        if len(disease_data) < 3:
            continue
        
        for comp_name, col1, col2 in week_comparisons:
            data = disease_data[[col1, col2]].dropna()
            try:
                corr_result = calculate_correlation(data[col1], data[col2])
                if corr_result:
                    corr_result['Disease'] = disease
                    corr_result['Comparison'] = comp_name
                    disease_correlations.append(corr_result)
            except:
                pass
    
    return pd.DataFrame(disease_correlations)

def plot_scatter_with_trend(ax, data, x_col, y_col, title, x_label, y_label):
    """Helper to create scatter plot with trend line and correlation info."""
    plot_data = data[[x_col, y_col, 'Disease']].dropna()
    if len(plot_data) == 0:
        return
    
    for disease in DISEASES:
        disease_data = plot_data[plot_data['Disease'] == disease]
        if len(disease_data) > 0:
            ax.scatter(disease_data[x_col], disease_data[y_col], 
                      alpha=0.7, s=60, label=disease, color=DISEASE_COLORS[disease])
    
    if len(plot_data) > 2:
        z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
        p = np.poly1d(z)
        ax.plot(plot_data[x_col].sort_values(), p(plot_data[x_col].sort_values()), 
               "k--", alpha=0.8, linewidth=1.5)
        
        spear_r, spear_p = spearmanr(plot_data[x_col], plot_data[y_col])
        ax.text(0.05, 0.95, f'r = {spear_r:.3f}, p = {spear_p:.3f}', 
               transform=ax.transAxes, verticalalignment='top', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=14)

def create_correlation_plots(df):
    """Create scatter plots showing correlations between adherence and SUS scores."""
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Active Test Adherence vs System Usability Scale (SUS) Scores', 
                 fontsize=14, fontweight='bold')
    
    plot_scatter_with_trend(axes[0], df, 'Week1', 'SUS_Week1', 
                           'Week 1: Adherence vs SUS', 
                           'Week 1 Active Adherence (%)', 'Week 1 SUS Score')
    
    plot_scatter_with_trend(axes[1], df, 'Week8', 'SUS_Week8', 
                           'Week 8: Adherence vs SUS', 
                           'Week 8 Active Adherence (%)', 'Week 8 SUS Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'active_adherence_sus_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_disease_panel(ax, disease_data, x_col, y_col, disease, title):
    """Helper to create a single disease-specific scatter plot panel."""
    data = disease_data[[x_col, y_col]].dropna()
    if len(data) > 1:
        ax.scatter(data[x_col], data[y_col], color=DISEASE_COLORS[disease], alpha=0.7, s=80)
        
        if len(data) > 2:
            z = np.polyfit(data[x_col], data[y_col], 1)
            p = np.poly1d(z)
            ax.plot(data[x_col].sort_values(), p(data[x_col].sort_values()), 
                   "k--", alpha=0.8, linewidth=2)
            
            spear_r, spear_p = spearmanr(data[x_col], data[y_col])
            ax.text(0.05, 0.95, f'r = {spear_r:.3f}\np = {spear_p:.3f}', 
                   transform=ax.transAxes, verticalalignment='top', fontsize=14,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel(f'{title.split("-")[1].strip()} Active Adherence (%)', fontsize=14)
    ax.set_ylabel(f'{title.split("-")[1].strip()} SUS Score', fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-5, 105)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='both', which='major', labelsize=14)

def create_disease_specific_plots(df):
    """Create separate correlation plots for each disease."""
    fig, axes = plt.subplots(len(DISEASES), 2, figsize=(12, 15))
    fig.suptitle('Disease-Specific Active Adherence vs SUS Correlations', 
                 fontsize=14, fontweight='bold')
    
    for i, disease in enumerate(DISEASES):
        disease_data = df[df['Disease'] == disease].copy()
        plot_disease_panel(axes[i, 0], disease_data, 'Week1', 'SUS_Week1', 
                          disease, f'{disease} - Week 1')
        plot_disease_panel(axes[i, 1], disease_data, 'Week8', 'SUS_Week8', 
                          disease, f'{disease} - Week 8')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'disease_specific_adherence_sus_correlations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def calculate_descriptive_stats(series):
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

def sus_descriptive_stats(df):
    """Generate descriptive statistics for SUS scores."""
    print("\n" + "=" * 50)
    print("SUS SCORE DESCRIPTIVE STATISTICS")
    print("=" * 50)
    
    sus_stats = {}
    for week_col, week_name in [('SUS_Week1', 'Week1_SUS'), ('SUS_Week8', 'Week8_SUS')]:
        stats = calculate_descriptive_stats(df[week_col].dropna())
        if stats:
            sus_stats[week_name] = stats
    
    disease_sus_stats = {}
    for disease in DISEASES:
        disease_data = df[df['Disease'] == disease]
        disease_sus_stats[disease] = {}
        
        for week_col, week_name in [('SUS_Week1', 'Week1'), ('SUS_Week8', 'Week8')]:
            stats = calculate_descriptive_stats(disease_data[week_col].dropna())
            if stats:
                disease_sus_stats[disease][week_name] = stats
    
    print("\nOverall SUS Statistics:")
    for week, stats in sus_stats.items():
        print(f"\n{week}:")
        print(f"  N: {stats['N']}")
        print(f"  Mean: {stats['Mean']:.2f} ± {stats['Std']:.2f}")
        print(f"  Median: {stats['Median']:.2f} (Q25: {stats['Q25']:.2f}, Q75: {stats['Q75']:.2f})")
        print(f"  Range: {stats['Min']:.2f} - {stats['Max']:.2f}")
    
    print("\nSUS Statistics by Disease:")
    for disease, disease_stats in disease_sus_stats.items():
        print(f"\n{disease}:")
        for week, stats in disease_stats.items():
            print(f"  {week}: N={stats['N']}, Mean={stats['Mean']:.2f}±{stats['Std']:.2f}, Median={stats['Median']:.2f}")
    
    return sus_stats, disease_sus_stats

def print_section(title, width=50):
    """Print a formatted section header."""
    print(f"\n{'=' * width}\n{title}\n{'=' * width}")

def print_correlation_finding(correlations, comparison_name, index):
    """Print a single correlation finding."""
    corr = correlations[correlations['Comparison'] == comparison_name]
    if len(corr) > 0:
        row = corr.iloc[0]
        sig = "significant" if row['Spearman_Significant'] else "not significant"
        print(f"{index}. {comparison_name}: r = {row['Spearman_r']:.3f}, p = {row['Spearman_p']:.3f} ({sig})")

def main():
    """Main function to run the SUS-Adherence correlation analysis."""
    print("\nStarting Active Adherence vs SUS Correlation Analysis...")
    
    df = load_and_prepare_data()
    if df is None:
        return
    
    sus_stats, disease_sus_stats = sus_descriptive_stats(df)
    
    print_section("OVERALL CORRELATION ANALYSIS")
    overall_correlations = correlation_analysis(df)
    print("\nOverall Correlations (Active Adherence vs SUS):")
    print(overall_correlations.to_string(index=False))
    overall_correlations.to_csv(os.path.join(OUTPUT_DIR, 'active_sus_correlations_overall.csv'), index=False)
    
    print_section("DISEASE-SPECIFIC CORRELATION ANALYSIS")
    disease_correlations = disease_specific_correlations(df)
    if len(disease_correlations) > 0:
        print("\nDisease-Specific Correlations:")
        print(disease_correlations.to_string(index=False))
        disease_correlations.to_csv(os.path.join(OUTPUT_DIR, 'active_sus_correlations_by_disease.csv'), index=False)
    else:
        print("Insufficient data for disease-specific correlations.")
    
    print_section("GENERATING CORRELATION PLOTS")
    create_correlation_plots(df)
    create_disease_specific_plots(df)
    df.to_csv(os.path.join(OUTPUT_DIR, 'active_adherence_with_sus_scores.csv'), index=False)
    
    print_section("ANALYSIS COMPLETE - SUMMARY", 70)
    print("\nKey Findings:")
    comparisons = [
        ('Week1 Adherence vs Week1 SUS', 'Week 1 Active Adherence vs Week 1 SUS'),
        ('Week8 Adherence vs Week8 SUS', 'Week 8 Active Adherence vs Week 8 SUS'),
        ('Average Adherence vs Average SUS', 'Average Active Adherence vs Average SUS')
    ]
    for i, (comp_key, comp_display) in enumerate(comparisons, 1):
        print_correlation_finding(overall_correlations, comp_key, i)
    
    output_files = [
        ('active_sus_correlations_overall.csv', 'Overall correlation results'),
        ('active_sus_correlations_by_disease.csv', 'Disease-specific correlation results'),
        ('active_adherence_with_sus_scores.csv', 'Enhanced dataset with SUS scores'),
        ('active_adherence_sus_correlations.png', 'Main correlation plots'),
        ('disease_specific_adherence_sus_correlations.png', 'Disease-specific plots')
    ]
    
    print("\nFiles generated:")
    for filename, description in output_files:
        print(f"- {filename}: {description}")
    
    print("\nAnalysis completed successfully!")

if __name__ == '__main__':
    main()
