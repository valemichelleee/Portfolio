import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

DISEASES = ['PD', 'PSP', 'MSA']
ADHERENCE_METRICS = [
    ('Week1_SimpleAverage', 'Week 1 Simple Average Adherence'),
    ('Week8_SimpleAverage', 'Week 8 Simple Average Adherence'),
    ('SimpleAverage_Change', 'Simple Average Change (W8-W1)')
]
SEVERITY_WEEKS = [1, 8]

def load_data():
    active_df = pd.read_csv(r'active_monitoring\active_adherence_with_sus_scores.csv')
    passive_df = pd.read_csv(r'passive_monitoring\passive_adherence_with_sus_scores.csv')
    severity_df = pd.read_excel('data/RCT_Clinical_Data.xlsx', sheet_name='Severity')
    return active_df, passive_df, severity_df

def create_simple_average_adherence(active_df, passive_df):
    merged_df = pd.merge(
        active_df[['PATID', 'Disease', 'Week1', 'Week8']], 
        passive_df[['PATID', 'Week1_PassiveAdherence', 'Week8_PassiveAdherence']], 
        on='PATID', 
        how='inner'
    ).rename(columns={'Week1': 'Week1_ActiveAdherence', 'Week8': 'Week8_ActiveAdherence'})
    
    for week in ['Week1', 'Week8']:
        merged_df[f'{week}_SimpleAverage'] = (
            merged_df[f'{week}_ActiveAdherence'] + merged_df[f'{week}_PassiveAdherence']
        ) / 2
    
    merged_df['SimpleAverage_Change'] = merged_df['Week8_SimpleAverage'] - merged_df['Week1_SimpleAverage']
    
    print(f"Created simple average adherence for {len(merged_df)} patients")
    for week in ['Week1', 'Week8']:
        col = f'{week}_SimpleAverage'
        print(f"\n{week.replace('Week', 'Week ')} Simple Average: Mean={merged_df[col].mean():.2f}%, Std={merged_df[col].std():.2f}%")
    
    return merged_df

def merge_with_severity(adherence_df, severity_df):
    merged_df = pd.merge(adherence_df, severity_df, on='PATID', how='inner')
    merged_df['Disease'] = merged_df['Label']
    print(f"\nMerged dataset: {len(merged_df)} patients with both adherence and severity data")
    return merged_df

def get_disease_specific_severity_column(disease, week):
    severity_cols = {
        'PD': f'MDS_UPDRS_week{week}',
        'PSP': f'PSP-RS_week{week}',
        'MSA': f'UMSARS_wek{week}' if week == 1 else f'UMSARS_week{week}'
    }
    return severity_cols.get(disease)

def get_significance(p_value):
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return "ns"

def compute_correlation(data, adherence_col, severity_col):
    valid_data = data[[adherence_col, severity_col]].dropna()
    if len(valid_data) < 5:
        return None
    
    corr, p_value = pearsonr(valid_data[adherence_col], valid_data[severity_col])
    spearman_corr, spearman_p = spearmanr(valid_data[adherence_col], valid_data[severity_col])
    
    return {
        'pearson_r': corr,
        'pearson_p': p_value,
        'pearson_sig': get_significance(p_value),
        'spearman_rho': spearman_corr,
        'spearman_p': spearman_p,
        'spearman_sig': get_significance(spearman_p),
        'n': len(valid_data)
    }

def calculate_correlations(merged_df):
    results = []
    
    print("\n" + "="*80)
    print("DISEASE-SPECIFIC CORRELATIONS (Simple Average Method)")
    print("="*80)
    
    for disease in DISEASES:
        disease_data = merged_df[merged_df['Disease'] == disease]
        print(f"\n{'='*80}")
        print(f"{disease} (n={len(disease_data)} patients)")
        print(f"{'='*80}")
        
        for adherence_col, adherence_name in ADHERENCE_METRICS:
            print(f"\n{adherence_name}:")
            
            for week in SEVERITY_WEEKS:
                sev_col = get_disease_specific_severity_column(disease, week)
                if not sev_col or sev_col not in disease_data.columns:
                    continue
                
                corr_result = compute_correlation(disease_data, adherence_col, sev_col)
                if not corr_result:
                    continue
                
                print(f"  Week {week} Severity: r={corr_result['pearson_r']:7.3f} (p={corr_result['pearson_p']:.4f}) {corr_result['pearson_sig']:3s} | "
                      f"ρ={corr_result['spearman_rho']:7.3f} (p={corr_result['spearman_p']:.4f}) {corr_result['spearman_sig']:3s} | n={corr_result['n']}")
                
                results.append({
                    'Disease': disease,
                    'Adherence_Metric': adherence_name,
                    'Severity_Week': f'Week {week}',
                    'Severity_Scale': sev_col,
                    'Pearson_r': corr_result['pearson_r'],
                    'Pearson_p': corr_result['pearson_p'],
                    'Pearson_Sig': corr_result['pearson_sig'],
                    'Spearman_rho': corr_result['spearman_rho'],
                    'Spearman_p': corr_result['spearman_p'],
                    'Spearman_Sig': corr_result['spearman_sig'],
                    'Sample_Size': corr_result['n']
                })
    
    return pd.DataFrame(results)

def plot_week_correlations(merged_df, week, adherence_col, ylabel):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'{ylabel} vs Disease Severity', fontsize=16, fontweight='bold')
    
    for i, disease in enumerate(DISEASES):
        disease_data = merged_df[merged_df['Disease'] == disease]
        sev_col = get_disease_specific_severity_column(disease, week)
        
        if sev_col and sev_col in disease_data.columns:
            valid_data = disease_data[[adherence_col, sev_col]].dropna()
            
            if len(valid_data) >= 5:
                axes[i].scatter(valid_data[sev_col], valid_data[adherence_col], 
                               alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
                
                z = np.polyfit(valid_data[sev_col], valid_data[adherence_col], 1)
                p = np.poly1d(z)
                axes[i].plot(valid_data[sev_col], p(valid_data[sev_col]), "r--", alpha=0.8, linewidth=2)
                
                corr, p_val = pearsonr(valid_data[adherence_col], valid_data[sev_col])
                sig = get_significance(p_val)
                
                axes[i].set_title(f'{disease} (n={len(valid_data)})\nr={corr:.3f}, p={p_val:.3f} {sig}', 
                                 fontsize=12, fontweight='bold')
                axes[i].set_xlabel(f'{sev_col.replace("_", " ").replace("wek", "week")}', fontsize=11)
                axes[i].set_ylabel(ylabel, fontsize=11)
                axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f'adherence_severity_simple_average_week{week}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    return filename

def create_correlation_plots(merged_df):
    filenames = [
        plot_week_correlations(merged_df, 1, 'Week1_SimpleAverage', 'Week 1 Simple Average Adherence (%)'),
        plot_week_correlations(merged_df, 8, 'Week8_SimpleAverage', 'Week 8 Simple Average Adherence (%)')
    ]
    
    print("\n✓ Plots saved:")
    for filename in filenames:
        print(f"  - {filename}")

def main():
    """Main function to execute the simple average correlation analysis"""
    print("="*80)
    print("ADHERENCE-SEVERITY CORRELATION ANALYSIS")
    print("Method: SIMPLE AVERAGE (Arithmetic Mean of Raw Percentages)")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    active_df, passive_df, severity_df = load_data()
    
    # Create simple average adherence
    print("\nCalculating simple average adherence...")
    adherence_df = create_simple_average_adherence(active_df, passive_df)
    
    # Merge with severity
    print("Merging with severity data...")
    merged_df = merge_with_severity(adherence_df, severity_df)
    
    # Calculate correlations
    print("\nCalculating correlations...")
    correlation_results = calculate_correlations(merged_df)
    
    # Create plots
    print("\nCreating correlation plots...")
    create_correlation_plots(merged_df)
    
    # Save results
    correlation_results.to_csv('adherence_severity_simple_average_correlations.csv', index=False)
    print(f"\n✓ Results saved to: adherence_severity_simple_average_correlations.csv")
    
    # Save adherence data
def print_summary(correlation_results):
    print("\n" + "="*80)
    print("SUMMARY OF SIGNIFICANT CORRELATIONS (p < 0.05)")
    print("="*80)
    significant = correlation_results[correlation_results['Pearson_p'] < 0.05].sort_values('Pearson_p')
    
    if len(significant) > 0:
        for _, row in significant.iterrows():
            print(f"\n{row['Disease']} - {row['Adherence_Metric']} vs {row['Severity_Week']} {row['Severity_Scale']}:")
            print(f"  Pearson:  r={row['Pearson_r']:7.3f}, p={row['Pearson_p']:.4f} {row['Pearson_Sig']}")
            print(f"  Spearman: ρ={row['Spearman_rho']:7.3f}, p={row['Spearman_p']:.4f} {row['Spearman_Sig']}")
            print(f"  Sample size: n={row['Sample_Size']}")
    else:
        print("\nNo statistically significant correlations found at p < 0.05")
    
    print("\n" + "="*80)
    print("INTERPRETATION GUIDE")
    print("="*80)
    print("Correlation strength: |r| < 0.3 (weak), 0.3 ≤ |r| < 0.7 (moderate), |r| ≥ 0.7 (strong)")
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("\nNegative correlations: Higher severity → Lower adherence")
    print("Positive correlations: Higher severity → Higher adherence")

def main():
    print("="*80)
    print("ADHERENCE-SEVERITY CORRELATION ANALYSIS")
    print("Method: SIMPLE AVERAGE (Arithmetic Mean of Raw Percentages)")
    print("="*80)
    
    print("\nLoading data...")
    active_df, passive_df, severity_df = load_data()
    
    print("\nCalculating simple average adherence...")
    adherence_df = create_simple_average_adherence(active_df, passive_df)
    
    print("Merging with severity data...")
    merged_df = merge_with_severity(adherence_df, severity_df)
    
    print("\nCalculating correlations...")
    correlation_results = calculate_correlations(merged_df)
    
    print("\nCreating correlation plots...")
    create_correlation_plots(merged_df)
    
    correlation_results.to_csv('adherence_severity_simple_average_correlations.csv', index=False)
    print(f"\nResults saved to: adherence_severity_simple_average_correlations.csv")
    
    output_cols = ['PATID', 'Disease', 
                   'Week1_ActiveAdherence', 'Week1_PassiveAdherence', 'Week1_SimpleAverage',
                   'Week8_ActiveAdherence', 'Week8_PassiveAdherence', 'Week8_SimpleAverage',
                   'SimpleAverage_Change']
    merged_df[output_cols].to_csv('simple_average_adherence_scores.csv', index=False)
    print(f"Adherence scores saved to: simple_average_adherence_scores.csv")
    
    print_summary(correlation_results)

if __name__ == "__main__":
    main()