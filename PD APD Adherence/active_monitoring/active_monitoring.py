import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
from scipy.stats import kruskal, wilcoxon
import seaborn as sns
import scikit_posthocs as sp

data_dir = os.path.join(os.path.dirname(__file__), "../data")
output_dir = os.path.dirname(__file__)

EXCLUDED_PATIENTS = [
    'pat001', 'pat005', 'pat014', 'pat117', 'pat120', 'pat124', 'pat127', 'pat131', 
    'pat133', 'pat134', 'pat154', 'pat212', 'pat233', 'pat238', 'pat314', 'PAT402', 
    'pat403', 'pat407', 'PAT408', 'PAT411', 'pat412', 'PAT414'
]
EXCLUDED_PATIENTS_UPPER = [pat.upper() for pat in EXCLUDED_PATIENTS]
REPORT_COLS = ['report_test_1', 'report_test_2', 'report_test_3']
TEST_MAPPING = {'test_0': 'report_test_1', 'test_1': 'report_test_2', 'test_2': 'report_test_3'}
DESIRED_DISEASE_ORDER = ['MSA', 'PD', 'PSP']
QUALITY_CATEGORIES = ['Completed', 'Incomplete', 'No Test']
QUALITY_COLORS = ['#2ca02c', '#ff7f0e', '#d62728']

def clean_data_for_stats(data_array):
    if len(data_array) == 0:
        return np.array([])
    return data_array[~np.isnan(data_array)]

def get_category(val):
    if pd.isna(val) or (isinstance(val, str) and val.lower() == 'no data'):
        return 'No Data'
    val = str(val)
    if val.startswith('*') and not val.startswith('**'):
        return 'Completed'
    elif val.startswith('**') and not val.startswith('***'):
        return 'Incomplete'
    elif val.startswith('***'):
        suffixes = {'': 'No Test', '(A)': 'No Test (A: Diary but no signal)',
                   '(B)': 'No Test (B: Diary and signal but unidentified)',
                   '(C)': 'No Test (C: Diary says not performed)'}
        return suffixes.get(val[3:], f'No Test ({val[3:]})')
    return 'Unknown'

def compute_adherence(df, week_label):
    wk = df[df['Week'] == week_label]
    if len(wk) == 0:
        return 0.0
    wk = wk.copy()
    wk['completed_tests'] = wk[REPORT_COLS].apply(
        lambda row: sum(isinstance(val, str) and (val.startswith('*') and not val.startswith('**') or val == '*F(2min)') 
                       for val in row), axis=1)
    total_completed = wk['completed_tests'].sum()
    return min(total_completed / 21.0, 1.0) * 100

def load_and_merge_data():
    overview = pd.read_excel(os.path.join(data_dir, 'active_monitoring.xlsx'), sheet_name='Sheet1')
    clinical = pd.read_excel(os.path.join(data_dir, 'rct_clinical_data.xlsx'), sheet_name='Visit4')
    overview_correct = pd.read_excel(os.path.join(data_dir, 'active_monitoring.xlsx'), sheet_name='Correct_files')
    labels = clinical[['PATID', 'Label']].drop_duplicates().set_index('PATID')
    overview = overview.merge(labels, on='PATID', how='left')
    return overview, clinical, overview_correct, labels

def print_exclusion_info(clinical):
    print(f"Excluding {len(EXCLUDED_PATIENTS)} patients from analyses (except time distribution): {EXCLUDED_PATIENTS}")
    print(f"Excluded patients (uppercase): {EXCLUDED_PATIENTS_UPPER}")
    
    patients_all = clinical['PATID'].unique()
    patients = [pat for pat in patients_all if pat not in EXCLUDED_PATIENTS_UPPER]
    
    print(f"\nTotal patients in clinical data: {len(patients_all)}")
    print(f"Patients included in adherence analysis: {len(patients)}")
    print(f"Patients excluded: {len(EXCLUDED_PATIENTS)}")
    print(f"Actually excluded: {len(patients_all) - len(patients)}")
    
    found_excluded = set(patients_all) & set(EXCLUDED_PATIENTS_UPPER)
    not_found_excluded = set(EXCLUDED_PATIENTS_UPPER) - set(patients_all)
    print(f"Excluded patients found in data: {sorted(found_excluded)}")
    print(f"Excluded patients NOT found in data: {sorted(not_found_excluded)}")
    
    return patients

def analyze_patient_padding(overview, clinical, patients):
    print("\n" + "="*60)
    print("ACTIVE MONITORING: PATIENT PADDING ANALYSIS")
    print("="*60)
    
    completely_padded, week1_padded, week8_padded, patients_with_data = [], [], [], []
    results = []
    
    for pat in patients:
        sub = overview[overview['PATID'] == pat]
        disease_info = clinical[clinical['PATID'] == pat]['Label']
        disease = disease_info.iloc[0] if len(disease_info) > 0 and not pd.isna(disease_info.iloc[0]) else 'Unknown'
        
        has_week1_data = len(sub[sub['Week'] == 'week_1']) > 0
        has_week8_data = len(sub[sub['Week'] == 'week_8']) > 0
        
        if len(sub) == 0:
            rate_w1, rate_w8 = 0.0, 0.0
            completely_padded.append((pat, disease))
            print(f"COMPLETELY PADDED: Patient {pat} ({disease}) - No data in either week (0% adherence both weeks)")
        else:
            rate_w1 = compute_adherence(sub, 'week_1')
            rate_w8 = compute_adherence(sub, 'week_8')
            
            if not has_week1_data and has_week8_data:
                week1_padded.append((pat, disease))
                print(f"WEEK 1 PADDED: Patient {pat} ({disease}) - Missing Week 1 data (padded with 0%)")
            elif has_week1_data and not has_week8_data:
                week8_padded.append((pat, disease))
                print(f"WEEK 8 PADDED: Patient {pat} ({disease}) - Missing Week 8 data (padded with 0%)")
            elif has_week1_data and has_week8_data:
                patients_with_data.append((pat, disease))
            else:
                completely_padded.append((pat, disease))
                print(f"COMPLETELY PADDED: Patient {pat} ({disease}) - No valid week data found")
        
        results.append({'PATID': pat, 'Disease': disease, 'Week1': rate_w1, 'Week8': rate_w8})
    
    print_padding_summary(completely_padded, week1_padded, week8_padded, patients_with_data, patients)
    return pd.DataFrame(results), completely_padded

def print_padding_summary(completely_padded, week1_padded, week8_padded, patients_with_data, patients):
    print(f"\nACTIVE MONITORING PADDING SUMMARY:")
    print(f"- Completely padded patients: {len(completely_padded)}")
    print(f"- Week 1 only padded: {len(week1_padded)}")
    print(f"- Week 8 only padded: {len(week8_padded)}")
    print(f"- Patients with real data: {len(patients_with_data)}")
    print(f"- Total patients: {len(patients)}")
    
    if completely_padded:
        padded_by_disease = {}
        for pat, disease in completely_padded:
            padded_by_disease[disease] = padded_by_disease.get(disease, 0) + 1
        print(f"\nCompletely padded patients by disease:")
        for disease, count in padded_by_disease.items():
            print(f"  {disease}: {count} patients")

def identify_never_participated(df_results, overview):
    patients_with_week1 = set(overview[overview['Week'] == 'week_1']['PATID'].unique()) - set(EXCLUDED_PATIENTS_UPPER)
    patients_with_week8 = set(overview[overview['Week'] == 'week_8']['PATID'].unique()) - set(EXCLUDED_PATIENTS_UPPER)
    
    df_results['HasWeek1Data'] = df_results['PATID'].isin(patients_with_week1)
    df_results['HasWeek8Data'] = df_results['PATID'].isin(patients_with_week8)
    
    never_participated = df_results[(~df_results['HasWeek1Data']) & (~df_results['HasWeek8Data'])]
    
    print(f"\n=== LIST OF {len(never_participated)} PATIENTS WHO NEVER PARTICIPATED IN MONITORING ===")
    for _, patient in never_participated.iterrows():
        print(f"Patient {patient['PATID']}: Disease = {patient['Disease']}")
    
    never_participated_by_disease = never_participated['Disease'].value_counts()
    print(f"\nBreakdown by disease:")
    for disease, count in never_participated_by_disease.items():
        total_disease = len(df_results[df_results['Disease'] == disease])
        percentage = (count / total_disease) * 100
        print(f"  {disease}: {count} patients ({percentage:.1f}% of all {disease} patients)")
    
    never_participated[['PATID', 'Disease']].to_csv(
        os.path.join(output_dir, 'never_participated_patients.csv'), index=False)
    print(f"\nList saved to: never_participated_patients.csv")
    
    return df_results

def compute_overall_stats(df_results):
    overall_avg_w1 = df_results['Week1'].mean()
    overall_avg_w8 = df_results['Week8'].mean()
    overall_change = overall_avg_w8 - overall_avg_w1
    print(f"Overall average adherence Week 1: {overall_avg_w1:.2f}%")
    print(f"Overall average adherence Week 8: {overall_avg_w8:.2f}%")
    print(f"Overall average change: {overall_change:.2f}%")
    df_results['Change'] = df_results['Week8'] - df_results['Week1']
    return overall_avg_w1, overall_avg_w8, overall_change

def perform_paired_ttest(df_results):
    paired_overall = df_results[(df_results['HasWeek1Data']) & (df_results['HasWeek8Data'])][['Week1', 'Week8']].dropna()
    if len(paired_overall) > 1:
        t_stat, p_value = stats.ttest_rel(paired_overall['Week1'], paired_overall['Week8'])
        print(f"Paired t-test comparing Week 1 vs Week 8 adherence (n={len(paired_overall)}): t={t_stat:.3f}, p={p_value:.3f}")
        print("There is a statistically significant change in adherence between Week 1 and Week 8" if p_value < 0.05 
              else "No statistically significant change in adherence between Week 1 and Week 8")
        return p_value
    print("Paired t-test comparing Week 1 vs Week 8 adherence: N/A (insufficient paired data)")
    return None

def compute_disease_stats(df_results, overall_avg_w1, overall_avg_w8, overall_change):
    disease_stats = df_results.groupby('Disease')[['Week1', 'Week8', 'Change']].agg(['mean', 'std', 'count'])
    
    overall_stats = pd.DataFrame({
        ('Week1', 'mean'): [overall_avg_w1], ('Week1', 'std'): [df_results['Week1'].std()],
        ('Week1', 'count'): [len(df_results)], ('Week8', 'mean'): [overall_avg_w8],
        ('Week8', 'std'): [df_results['Week8'].std()], ('Week8', 'count'): [len(df_results)],
        ('Change', 'mean'): [overall_change], ('Change', 'std'): [df_results['Change'].std()],
        ('Change', 'count'): [len(df_results)]
    }, index=['All Patients'])
    
    disease_stats = pd.concat([disease_stats, overall_stats])
    disease_stats.to_csv(os.path.join(output_dir, 'disease_adherence_stats.csv'))
    print("\nAdherence by disease group:")
    print(disease_stats)
    return disease_stats

def prepare_plot_data(df_results, diseases):
    participating_patients = df_results.copy()
    print(f"Including ALL {len(participating_patients)} patients in disease comparisons (intention-to-treat)")
    print(f"This includes {(~df_results['HasWeek1Data'] & ~df_results['HasWeek8Data']).sum()} never-participated patients")
    
    week1_data = [participating_patients[participating_patients['Disease'] == d]['Week1'].values for d in diseases]
    week8_data = [participating_patients[participating_patients['Disease'] == d]['Week8'].values for d in diseases]
    
    week1_counts = [len(data) for data in week1_data]
    week8_counts = [len(data) for data in week8_data]
    week1_means = [np.nanmean(data) if len(data) > 0 else np.nan for data in week1_data]
    week8_means = [np.nanmean(data) if len(data) > 0 else np.nan for data in week8_data]
    
    return week1_data, week8_data, week1_counts, week8_counts, week1_means, week8_means

def perform_kruskal_wallis(week1_data, week8_data):
    clean_week1 = [clean_data_for_stats(d) for d in week1_data if len(clean_data_for_stats(d)) > 0]
    clean_week8 = [clean_data_for_stats(d) for d in week8_data if len(clean_data_for_stats(d)) > 0]
    
    kw_pval_w1 = kruskal(*clean_week1)[1] if len(clean_week1) >= 2 else None
    kw_pval_w8 = kruskal(*clean_week8)[1] if len(clean_week8) >= 2 else None
    
    return kw_pval_w1, kw_pval_w8

def perform_within_disease_tests(df_results, diseases):
    within_disease_pvals = {}
    within_disease_text = []
    
    for disease in diseases:
        disease_data = df_results[(df_results['Disease'] == disease) &
                                  (df_results['HasWeek1Data']) &
                                  (df_results['HasWeek8Data'])]
        paired = disease_data[['Week1', 'Week8']].dropna()
        
        if len(paired) > 0:
            try:
                statistic, p_val = wilcoxon(paired['Week1'].values, paired['Week8'].values, alternative='two-sided')
                within_disease_pvals[disease] = p_val
                within_disease_text.append(f"{disease}: p={p_val:.3f}")
            except ValueError:
                within_disease_pvals[disease] = 1.0
                within_disease_text.append(f"{disease}: p=1.000 =")
        else:
            within_disease_pvals[disease] = None
            within_disease_text.append(f"{disease}: N/A")
    
    return within_disease_pvals, within_disease_text

def perform_posthoc_dunn(participating_patients, diseases, week_name, p_value):
    if p_value is None or p_value >= 0.05:
        return None, []
    
    all_data = []
    for d in diseases:
        disease_data = participating_patients[participating_patients['Disease'] == d][week_name].dropna()
        for val in disease_data:
            all_data.append({'value': val, 'group': d})
    all_data = pd.DataFrame(all_data)
    
    if len(all_data) == 0:
        return None, []
    
    posthoc_dunn = sp.posthoc_dunn(all_data, val_col='value', group_col='group', p_adjust='bonferroni')
    sig_pairs = []
    
    for i in range(len(posthoc_dunn.columns)):
        for j in range(i+1, len(posthoc_dunn.columns)):
            p = posthoc_dunn.iloc[i, j]
            if not pd.isna(p) and p < 0.05:
                sig_pairs.append(f"{posthoc_dunn.columns[i]} vs {posthoc_dunn.columns[j]}: p={p:.3f}")
    
    return posthoc_dunn, sig_pairs

def create_violin_boxplot(diseases, week1_data, week8_data, week1_counts, week8_counts, 
                          week1_means, week8_means, kw_pval_w1, kw_pval_w8, 
                          within_disease_text, posthoc_results):
    plt.figure(figsize=(12, 8))
    
    clean_week1_data = [clean_data_for_stats(d) for d in week1_data]
    clean_week8_data = [clean_data_for_stats(d) for d in week8_data]
    
    positions = np.arange(1, len(diseases) + 1)
    width = 0.35
    
    plt.violinplot([d for d in clean_week1_data if len(d) > 0], positions=positions-width/2, 
                  showextrema=False, widths=width*1.2)
    plt.violinplot([d for d in clean_week8_data if len(d) > 0], positions=positions+width/2, 
                  showextrema=False, widths=width*1.2)
    
    box1 = plt.boxplot([d for d in clean_week1_data if len(d) > 0], positions=positions-width/2, 
                       widths=width*0.3, patch_artist=True, showfliers=False,
                       medianprops=dict(color='darkblue', linewidth=2))
    box2 = plt.boxplot([d for d in clean_week8_data if len(d) > 0], positions=positions+width/2, 
                       widths=width*0.3, patch_artist=True, showfliers=False,
                       medianprops=dict(color='darkred', linewidth=2))
    
    for box in box1['boxes']:
        box.set(facecolor='skyblue')
    for box in box2['boxes']:
        box.set(facecolor='salmon')
    
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='skyblue', label='Week 1'),
                      Patch(facecolor='salmon', label='Week 8')]
    plt.legend(handles=legend_elements, loc='upper right', ncol=2, fontsize=14)
    
    plt.xticks(positions, diseases, fontsize=14)
    plt.title('Adherence by Disease and Week', fontsize=14)
    plt.ylabel('Adherence Rate (%)', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    posthoc_text = []
    for week, result_df in posthoc_results.items():
        sig_pairs = []
        for i in range(len(result_df.columns)):
            for j in range(i+1, len(result_df.columns)):
                p = result_df.iloc[i, j]
                if not pd.isna(p) and p < 0.05:
                    sig_pairs.append(f"{result_df.columns[i]} vs {result_df.columns[j]}: p={p:.3f}")
        if sig_pairs:
            posthoc_text.append(f"{week} Significant Pairs:")
            posthoc_text.extend(sig_pairs)
    
    if posthoc_text:
        posthoc_str = '\n'.join(posthoc_text)
        plt.figtext(0.5, -0.30, posthoc_str, ha='center', fontsize=14, 
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.subplots_adjust(bottom=0.3)
    
    table_data = [
        ['Week 1', 'N'] + [str(n) for n in week1_counts],
        ['', ''] + [f"{m:.2f}" if not np.isnan(m) else "N/A" for m in week1_means],
        ['Week 8', 'N'] + [str(n) for n in week8_counts],
        ['', ''] + [f"{m:.2f}" if not np.isnan(m) else "N/A" for m in week8_means]
    ]
    
    bottom_margin = 0.45 if posthoc_text else 0.35
    plt.subplots_adjust(bottom=bottom_margin)
    table = plt.table(cellText=table_data, colLabels=['', ''] + diseases,
                     cellLoc='center', loc='bottom', bbox=[0, -0.2, 1, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 2)
    
    w1_pval_str = f"{kw_pval_w1:.3f}" if kw_pval_w1 is not None else "N/A"
    w8_pval_str = f"{kw_pval_w8:.3f}" if kw_pval_w8 is not None else "N/A"
    kw_text = f"Kruskal-Wallis Test p-values: Week 1: {w1_pval_str}, Week 8: {w8_pval_str}"
    plt.figtext(0.5, -0.10, kw_text, ha='center', fontsize=14, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    if within_disease_text:
        within_text = "Paired tests (Week 1 vs Week 8): " + " | ".join(within_disease_text)
        plt.figtext(0.5, -0.15, within_text, ha='center', fontsize=14, 
                   bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'adherence_boxplots_by_disease.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_posthoc_and_within_disease(posthoc_results, within_disease_pvals):
    for week, result_df in posthoc_results.items():
        result_df.to_csv(os.path.join(output_dir, f'adherence_{week.replace(" ", "_")}_posthoc.csv'))
    
    within_disease_df = pd.DataFrame([
        {'Disease': disease, 'Week1_vs_Week8_pvalue': p_val, 
         'Significant': p_val < 0.05 if p_val is not None else False}
        for disease, p_val in within_disease_pvals.items()
    ])
    within_disease_df.to_csv(os.path.join(output_dir, 'adherence_within_disease_comparisons.csv'), index=False)
    
    print("\nWithin-disease adherence changes (Week 1 vs Week 8):")
    for disease, p_val in within_disease_pvals.items():
        if p_val is not None:
            significance = "significant" if p_val < 0.05 else "not significant"
            print(f"{disease}: p={p_val:.3f} ({significance})")
        else:
            print(f"{disease}: N/A (insufficient data)")

def analyze_test_status(overview):
    overview_filtered = overview[~overview['PATID'].isin(EXCLUDED_PATIENTS_UPPER)]
    melted = pd.melt(overview_filtered, id_vars=['PATID', 'Week', 'Label'], 
                    value_vars=REPORT_COLS, var_name='Test', value_name='Status')
    melted = melted.rename(columns={'Label': 'Disease'})
    melted['Category'] = melted['Status'].apply(get_category)
    melted['Main_Category'] = melted['Category'].apply(lambda x: x.split(' ')[0])
    
    print(f"\nTest status analysis excludes {len(EXCLUDED_PATIENTS)} patients")
    print(f"Analyzing test data for {overview_filtered['PATID'].nunique()} patients")
    
    unknowns = melted[melted['Category'] == 'Unknown']['Status'].unique()
    if len(unknowns) > 0:
        print("\nUnknown statuses:", unknowns)
    
    counts = melted.groupby(['Week', 'Disease', 'Category']).size().reset_index(name='Count')
    counts.to_csv(os.path.join(output_dir, 'test_status_summary.csv'), index=False)
    
    main_cat_counts = melted.groupby(['Week', 'Main_Category']).size().reset_index(name='Count')
    main_cat_pivot = main_cat_counts.pivot(index='Week', columns='Main_Category', values='Count').fillna(0)
    main_cat_pivot_pct = main_cat_pivot.div(main_cat_pivot.sum(axis=1), axis=0) * 100
    
    print("\nTest status distribution by week (%):")
    print(main_cat_pivot_pct)
    
    return melted

def compute_completion_metrics(melted):
    completion_metrics = {}
    for week in ['week_1', 'week_8']:
        week_data = melted[melted['Week'] == week]
        total_tests = len(week_data)
        if total_tests > 0:
            completed = sum(week_data['Main_Category'] == 'Completed')
            incomplete = sum(week_data['Main_Category'] == 'Incomplete')
            no_test = sum(week_data['Main_Category'] == 'No')
            no_data = sum(week_data['Main_Category'] == 'No')
            
            completion_metrics[week] = {
                'Total_Tests': total_tests,
                'Completed_Rate': completed / total_tests * 100,
                'Incomplete_Rate': incomplete / total_tests * 100,
                'No_Test_Rate': no_test / total_tests * 100,
                'No_Data_Rate': no_data / total_tests * 100,
            }
    
    print("\nTest completion metrics by week:")
    print(pd.DataFrame(completion_metrics).T)

def compute_additional_metrics(df_results, melted):
    zero_w1 = (df_results['Week1'] == 0).sum() / len(df_results) * 100
    zero_w8 = (df_results['Week8'] == 0).sum() / len(df_results) * 100
    perfect_w1 = (df_results['Week1'] == 100).sum() / len(df_results) * 100
    perfect_w8 = (df_results['Week8'] == 100).sum() / len(df_results) * 100
    
    print(f'\nPercentage of patients with 0% adherence in Week 1: {zero_w1:.2f}%')
    print(f'Percentage of patients with 0% adherence in Week 8: {zero_w8:.2f}%')
    print(f'Percentage of patients with 100% adherence in Week 1: {perfect_w1:.2f}%')
    print(f'Percentage of patients with 100% adherence in Week 8: {perfect_w8:.2f}%')
    
    test_number_analysis = melted.groupby(['Week', 'Test', 'Main_Category']).size().unstack(fill_value=0)
    test_number_analysis_pct = test_number_analysis.div(test_number_analysis.sum(axis=1), axis=0) * 100
    
    print("\nAdherence by test number (%):")
    print(test_number_analysis_pct)
    
    detailed_metrics = df_results.copy()
    detailed_metrics['Zero_Adherence_W1'] = detailed_metrics['Week1'] == 0
    detailed_metrics['Zero_Adherence_W8'] = detailed_metrics['Week8'] == 0
    detailed_metrics['Perfect_Adherence_W1'] = detailed_metrics['Week1'] == 100
    detailed_metrics['Perfect_Adherence_W8'] = detailed_metrics['Week8'] == 100
    detailed_metrics['Adherence_Improved'] = detailed_metrics['Week8'] > detailed_metrics['Week1']
    detailed_metrics['Adherence_Declined'] = detailed_metrics['Week8'] < detailed_metrics['Week1']
    detailed_metrics.to_csv(os.path.join(output_dir, 'detailed_adherence_metrics.csv'), index=False)
    
    no_test_categories = melted[melted['Main_Category'] == 'No']['Category'].value_counts()
    print("\nDistribution of 'No Test' categories:")
    print(no_test_categories)

def extract_timestamp_data(overview_correct, labels):
    timestamp_data = []
    
    for _, row in overview_correct.iterrows():
        patient_id = row['PATID']
        week = row['Week']
        disease = labels.loc[patient_id, 'Label'] if patient_id in labels.index else 'Unknown'
        
        for test_col in ['test_0', 'test_1', 'test_2']:
            timestamp_str = row[test_col]
            if pd.notna(timestamp_str) and timestamp_str != '':
                try:
                    timestamp = pd.to_datetime(timestamp_str, errors='coerce')
                    if pd.notna(timestamp):
                        timestamp_data.append({
                            'PATID': patient_id, 'Disease': disease, 'Week': week,
                            'Timestamp': timestamp,
                            'Hour': timestamp.hour + timestamp.minute/60.0,
                            'Test_Column': test_col
                        })
                except:
                    continue
    
    return pd.DataFrame(timestamp_data)

def create_time_distribution_plot(df_timestamps_clean, week_name, diseases, colors):
    week_data = df_timestamps_clean[df_timestamps_clean['Week'] == week_name]
    
    if len(week_data) == 0:
        return
    
    plt.figure(figsize=(12, 8))
    
    for i, disease in enumerate(diseases):
        disease_data = week_data[week_data['Disease'] == disease]
        if len(disease_data) > 1:
            sns.kdeplot(data=disease_data['Hour'], label=f'{disease} (n={len(disease_data)})', 
                       color=colors[i % len(colors)], linewidth=2.5)
    
    plt.xlabel('Time of Day (Hours)', fontsize=14)
    plt.ylabel('Normalized Density', fontsize=14)
    week_title = "Week 1" if week_name == 'week_1' else "Week 8"
    plt.title(f'Test Time Distribution by Disease - {week_title}', 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlim(0, 24)
    
    hour_ticks = range(0, 25, 3)
    hour_labels = [f"{h:02d}:00" for h in hour_ticks]
    plt.xticks(hour_ticks, hour_labels, rotation=45, fontsize=14)
    
    plt.legend(fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    
    filename = f'time_distribution_{week_name}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {filename}")
    plt.close()

def print_time_distribution_summary(df_timestamps_clean, diseases):
    print("\nSummary by Disease and Week:")
    for week_name in ['week_1', 'week_8']:
        print(f"\n--- {week_name.upper().replace('_', ' ')} ---")
        week_data = df_timestamps_clean[df_timestamps_clean['Week'] == week_name]
        
        for disease in diseases:
            disease_data = week_data[week_data['Disease'] == disease]
            if len(disease_data) == 0:
                continue
            
            mean_hour = disease_data['Hour'].mean()
            std_hour = disease_data['Hour'].std()
            median_hour = disease_data['Hour'].median()
            
            print(f"\n{disease}:")
            print(f"  Number of tests: {len(disease_data)}")
            print(f"  Mean time: {mean_hour:.1f}h ({int(mean_hour):02d}:{int((mean_hour % 1) * 60):02d})")
            print(f"  Median time: {median_hour:.1f}h ({int(median_hour):02d}:{int((median_hour % 1) * 60):02d})")
            print(f"  Standard deviation: {std_hour:.1f}h")
            
            if len(disease_data) > 0:
                hour_counts = disease_data['Hour'].round().value_counts()
                if len(hour_counts) > 0:
                    most_common_hour = int(hour_counts.index[0])
                    print(f"  Most common hour: {most_common_hour}:00 ({hour_counts.iloc[0]} tests)")

def analyze_time_distribution(overview_correct, labels):
    print("\n" + "="*50)
    print("GENERATING TIME DISTRIBUTION ANALYSIS (WEEK 1 & WEEK 8)")
    print("="*50)
    print(f"TIME DISTRIBUTION ANALYSIS: Using Correct_files sheet - Including ALL patients (even those excluded from other analyses)")
    
    df_timestamps = extract_timestamp_data(overview_correct, labels)
    
    if len(df_timestamps) == 0:
        print("No valid timestamps found in the data")
        return
    
    print(f"Found {len(df_timestamps)} valid timestamps across all tests")
    
    df_timestamps_clean = df_timestamps[df_timestamps['Disease'] != 'Unknown']
    
    if len(df_timestamps_clean) == 0:
        print("No valid timestamps found after filtering unknown diseases")
        return
    
    available_diseases = set(df_timestamps_clean['Disease'].unique())
    diseases = [d for d in DESIRED_DISEASE_ORDER if d in available_diseases]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for week_name in ['week_1', 'week_8']:
        create_time_distribution_plot(df_timestamps_clean, week_name, diseases, colors)
    
    print_time_distribution_summary(df_timestamps_clean, diseases)

def extract_quality_data(overview_correct, labels):
    quality_bar_data = []
    
    for _, row in overview_correct.iterrows():
        patient_id = row['PATID']
        week = row['Week']
        disease = labels.loc[patient_id, 'Label'] if patient_id in labels.index else 'Unknown'
        
        for test_col, report_col in TEST_MAPPING.items():
            test_result = row[report_col]
            
            if pd.notna(test_result) and test_result != '':
                quality = get_category(test_result)
                if quality.startswith('No Test') or quality == 'No Data':
                    quality = 'No Test'
                
                quality_bar_data.append({
                    'PATID': patient_id, 'Disease': disease, 'Week': week,
                    'Test_Column': test_col, 'Test_Result': test_result, 'Quality': quality
                })
    
    return pd.DataFrame(quality_bar_data)

def create_quality_bar_plot(week_data, diseases, week_name):
    fig, axes = plt.subplots(1, len(diseases), figsize=(6*len(diseases), 6))
    if len(diseases) == 1:
        axes = [axes]
    
    max_y_value = 0
    
    for i, disease in enumerate(diseases):
        disease_data = week_data[week_data['Disease'] == disease]
        
        if len(disease_data) == 0:
            continue
        
        quality_counts = disease_data['Quality'].value_counts()
        total_disease_tests = len(disease_data)
        
        x_positions, heights, colors_list, bar_labels = [], [], [], []
        
        for j, quality in enumerate(QUALITY_CATEGORIES):
            count = quality_counts.get(quality, 0)
            percentage = (count / total_disease_tests) * 100 if total_disease_tests > 0 else 0
            
            x_positions.append(j)
            heights.append(percentage)
            colors_list.append(QUALITY_COLORS[j])
            bar_labels.append(f'{quality}\n({count}/{total_disease_tests})')
            max_y_value = max(max_y_value, percentage)
        
        bars = axes[i].bar(x_positions, heights, color=colors_list, alpha=0.7, edgecolor='black', linewidth=1)
        
        for bar, height in zip(bars, heights):
            if height > 0:
                axes[i].text(bar.get_x() + bar.get_width()/2., height + max_y_value*0.01,
                           f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        axes[i].set_xticks(x_positions)
        axes[i].set_xticklabels(bar_labels, fontsize=14)
        axes[i].set_ylabel('Percentage of Tests (%)', fontsize=14)
        axes[i].set_title(f'{disease}\n(Total: {total_disease_tests} tests)', fontsize=14, fontweight='bold')
        axes[i].set_ylim(0, max_y_value * 1.15)
        axes[i].grid(True, alpha=0.3, axis='y')
        axes[i].tick_params(axis='both', which='major', labelsize=14)
    
    for ax in axes:
        ax.set_ylim(0, max_y_value * 1.15)
    
    week_title = "Week 1" if week_name == 'week_1' else "Week 8"
    fig.suptitle(f'Test Quality Distribution - {week_title}', fontsize=14, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    filename = f'test_quality_{week_name}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Quality bar plot saved as: {filename}")
    plt.close()

def print_quality_summary(df_quality_clean, diseases):
    print("\nSummary by Test Quality, Disease, and Week:")
    for week_name in ['week_1', 'week_8']:
        print(f"\n--- {week_name.upper().replace('_', ' ')} ---")
        week_data = df_quality_clean[df_quality_clean['Week'] == week_name]
        
        for disease in diseases:
            disease_data = week_data[week_data['Disease'] == disease]
            if len(disease_data) == 0:
                continue
            
            total_disease_tests = len(disease_data)
            quality_counts = disease_data['Quality'].value_counts()
            
            print(f"\n{disease} (Total tests: {total_disease_tests}):")
            for quality in QUALITY_CATEGORIES:
                count = quality_counts.get(quality, 0)
                percentage = (count / total_disease_tests) * 100 if total_disease_tests > 0 else 0
                print(f"  {quality}: {count} tests ({percentage:.1f}%)")

def analyze_quality_distribution(overview_correct, labels):
    print("\n" + "="*60)
    print("GENERATING BAR PLOT ANALYSIS BY TEST QUALITY")
    print("="*60)
    
    df_quality_bar = extract_quality_data(overview_correct, labels)
    
    if len(df_quality_bar) == 0:
        print("No valid test quality data found")
        return
    
    print(f"Found {len(df_quality_bar)} tests with quality information")
    
    df_quality_clean = df_quality_bar[df_quality_bar['Disease'] != 'Unknown']
    
    if len(df_quality_clean) == 0:
        print("No valid test quality data found after filtering unknown diseases")
        return
    
    print("\nActual quality categories found:")
    quality_distribution = df_quality_clean['Quality'].value_counts()
    print(quality_distribution)
    
    available_diseases = set(df_quality_clean['Disease'].unique())
    diseases = [d for d in DESIRED_DISEASE_ORDER if d in available_diseases]
    
    for week_name in ['week_1', 'week_8']:
        week_data = df_quality_clean[df_quality_clean['Week'] == week_name]
        if len(week_data) > 0:
            create_quality_bar_plot(week_data, diseases, week_name)
    
    print_quality_summary(df_quality_clean, diseases)

def main():
    overview, clinical, overview_correct, labels = load_and_merge_data()
    patients = print_exclusion_info(clinical)
    
    df_results, completely_padded = analyze_patient_padding(overview, clinical, patients)
    df_results = identify_never_participated(df_results, overview)
    
    overall_avg_w1, overall_avg_w8, overall_change = compute_overall_stats(df_results)
    df_results.to_csv(os.path.join(output_dir, 'active_test_behavior_summary.csv'), index=False)
    
    perform_paired_ttest(df_results)
    compute_disease_stats(df_results, overall_avg_w1, overall_avg_w8, overall_change)
    
    diseases = sorted(df_results['Disease'].unique())
    week1_data, week8_data, week1_counts, week8_counts, week1_means, week8_means = prepare_plot_data(df_results, diseases)
    
    kw_pval_w1, kw_pval_w8 = perform_kruskal_wallis(week1_data, week8_data)
    within_disease_pvals, within_disease_text = perform_within_disease_tests(df_results, diseases)
    
    posthoc_results = {}
    participating_patients = df_results.copy()
    
    posthoc_w1, sig_pairs_w1 = perform_posthoc_dunn(participating_patients, diseases, 'Week1', kw_pval_w1)
    if posthoc_w1 is not None:
        posthoc_results['Week 1'] = posthoc_w1
    
    posthoc_w8, sig_pairs_w8 = perform_posthoc_dunn(participating_patients, diseases, 'Week8', kw_pval_w8)
    if posthoc_w8 is not None:
        posthoc_results['Week 8'] = posthoc_w8
    
    create_violin_boxplot(diseases, week1_data, week8_data, week1_counts, week8_counts,
                         week1_means, week8_means, kw_pval_w1, kw_pval_w8,
                         within_disease_text, posthoc_results)
    
    save_posthoc_and_within_disease(posthoc_results, within_disease_pvals)
    
    melted = analyze_test_status(overview)
    compute_completion_metrics(melted)
    compute_additional_metrics(df_results, melted)
    
    analyze_time_distribution(overview_correct, labels)
    analyze_quality_distribution(overview_correct, labels)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
