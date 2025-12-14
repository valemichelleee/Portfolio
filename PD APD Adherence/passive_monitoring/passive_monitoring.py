import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy import stats
import scikit_posthocs as sp

def duration_pattern(avg_worn_min, avg_worn_max, thresh_hours=1):
    difference = avg_worn_max - avg_worn_min
    
    if difference <= thresh_hours:
        return 'Consistent'
    elif difference <= 2 * thresh_hours:
        return 'Moderate difference'
    else:
        return 'High difference'

def usage_pattern(row):
    left_worn = row['avg_worn_LF'] > 0
    right_worn = row['avg_worn_RF'] > 0
    if left_worn and right_worn:
        return 'Both'
    elif left_worn:
        return 'Left Only'
    elif right_worn:
        return 'Right Only'
    else:
        return 'None'

def leg_dominance(avg_worn_LF, avg_worn_RF, thresh_minutes=15):
    thresh_hours = thresh_minutes / 60
    difference = avg_worn_LF - avg_worn_RF
    
    if difference >= thresh_hours:
        return 'Left-dominant'
    elif difference <= -thresh_hours:
        return 'Right-dominant'
    else:
        return 'Balanced'

def process_week_data(df, disease_df, week_num):
    dff = df[df['week'] == week_num].copy()
    dff['day'] = dff.groupby('PATID').cumcount() + 1

    dff['min_worn_time'] = dff[['worn_time_LF', 'worn_time_RF']].min(axis=1)
    dff['max_worn_time'] = dff[['worn_time_LF', 'worn_time_RF']].max(axis=1)

    dff['valid_day'] = (dff['max_worn_time'] >= 8).astype(int)


    agg = dff.groupby('PATID').agg(
        total_days=('day', 'nunique'),
        valid_days=('valid_day', 'sum'),
        avg_worn_min=('min_worn_time', 'mean'),
        avg_worn_max=('max_worn_time', 'mean'),
        avg_worn_LF=('worn_time_LF', 'mean'),
        avg_worn_RF=('worn_time_RF', 'mean'),
        avg_recording=('measurement_time', 'mean')
    ).reset_index()

    real_patients = set(agg['PATID'])

    agg['denom'] = agg['total_days'].apply(lambda x: min(x, 7))
    agg['valid_days_capped'] = agg['valid_days'].clip(upper=7)
    agg['valid_days_rate'] = (agg['valid_days_capped'] / agg['denom']).round(3)

    agg['ratio_min'] = (agg['avg_worn_min'] / agg['avg_recording']).replace([np.inf, -np.inf], np.nan).round(3)
    agg['ratio_max'] = (agg['avg_worn_max'] / agg['avg_recording']).replace([np.inf, -np.inf], np.nan).round(3)
    
    agg['ratio_LF'] = (agg['avg_worn_LF'] / agg['avg_recording']).replace([np.inf, -np.inf], np.nan).round(3)
    agg['ratio_RF'] = (agg['avg_worn_RF'] / agg['avg_recording']).replace([np.inf, -np.inf], np.nan).round(3)
    

    all_patids = set(disease_df['PATID'])
    existing_patids = set(agg['PATID'])
    missing_patids = all_patids - existing_patids
    
    if week_num == 1:
        print(f"\n" + "="*60)
        print(f"PASSIVE MONITORING: PATIENT PADDING ANALYSIS")
        print("="*60)
    
    print(f"\nWEEK {week_num} PADDING:")
    print(f"- Total patients in clinical data: {len(all_patids)}")
    print(f"- Patients with real wearing time data: {len(existing_patids)}")
    print(f"- Patients being padded: {len(missing_patids)}")
    
    if missing_patids:
        missing_with_disease = disease_df[disease_df['PATID'].isin(missing_patids)][['PATID', 'Label']]
        print(f"Padded patients with their diseases:")
        for _, row in missing_with_disease.iterrows():
            print(f"  Patient {row['PATID']} ({row['Label']}) - Padded with zeros for all metrics")
        
        disease_counts = missing_with_disease['Label'].value_counts()
        print(f"Padded patients by disease:")
        for disease, count in disease_counts.items():
            print(f"  {disease}: {count} patients")
    
    if missing_patids:
        padding_data = []
        for patid in missing_patids:
            padding_data.append({
                'PATID': patid,
                'total_days': 7,
                'valid_days': 0,
                'avg_worn_min': 0,
                'avg_worn_max': 0,
                'avg_worn_LF': 0,
                'avg_worn_RF': 0,
                'avg_recording': 0,
                'denom': 7,
                'valid_days_capped': 0,
                'valid_days_rate': 0,
                'ratio_min': 0,
                'ratio_max': 0,
                'ratio_LF': 0,
                'ratio_RF': 0
            })
        
        padding_df = pd.DataFrame(padding_data)
        agg = pd.concat([agg, padding_df], ignore_index=True)
        
        print(f"Padding values used for Week {week_num}:")
        print(f"  - total_days: 7, valid_days: 0, avg_worn_min: 0, avg_worn_max: 0")
        print(f"  - avg_worn_LF: 0, avg_worn_RF: 0, avg_recording: 0")
        print(f"  - valid_days_rate: 0, all ratios: 0, duration_pattern: 'No data'")
        print(f"  - leg_dominance: 'Balanced' (when both feet = 0)")
    else:
        print(f"No padding needed for Week {week_num} - all patients have data")

    authored = agg.merge(disease_df[['PATID', 'Label']], on='PATID', how='left')
    
    authored['duration_difference'] = authored['avg_worn_max'] - authored['avg_worn_min']
    authored['duration_ratio'] = (authored['avg_worn_min'] / authored['avg_worn_max']).replace([np.inf, -np.inf], np.nan)
    
    authored['duration_pattern'] = authored.apply(
        lambda row: duration_pattern(row['avg_worn_min'], row['avg_worn_max']) 
        if row['avg_worn_max'] > 0 else 'No data', axis=1
    )
    
    authored['leg_ratio'] = (authored['avg_worn_LF'] / authored['avg_worn_RF']).replace([np.inf, -np.inf], np.nan)
    authored['leg_dominance'] = authored.apply(lambda row: leg_dominance(row['avg_worn_LF'], row['avg_worn_RF']), axis=1)
    
    return authored, real_patients

def generate_weekly_plots(authored, week_num, output_dir, divisors, disease_data_by_metric=None):
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = ['avg_worn_min', 'avg_worn_max', 'avg_recording', 'valid_days_rate']
    disease_labels = sorted(divisors.keys())
    
    if disease_data_by_metric is not None:
        for m in metrics:
            if m not in disease_data_by_metric:
                disease_data_by_metric[m] = {}
            
            disease_data_by_metric[m][week_num] = {
                'data': [authored.loc[authored['Label'] == lbl, m].dropna() for lbl in disease_labels],
                'counts': [len(authored.loc[authored['Label'] == lbl, m].dropna()) for lbl in disease_labels],
                'means': [authored.loc[authored['Label'] == lbl, m].dropna().mean() if not authored.loc[authored['Label'] == lbl, m].dropna().empty else np.nan for lbl in disease_labels]
            }

    metric_groups = {
        'worn_time': ['avg_worn_min', 'avg_worn_max'],
        'recording_time': ['avg_recording'],
        'rates': ['valid_days_rate']
    }
    
    group_ylims = {}
    for group_name, group_metrics in metric_groups.items():
        all_values = []
        for m in group_metrics:
            if m in metrics:
                metric_data = [authored[m].dropna()] + [authored.loc[authored['Label'] == lbl, m].dropna() for lbl in disease_labels]
                for data_array in metric_data:
                    if len(data_array) > 0:
                        all_values.extend(data_array.tolist())
        
        if all_values:
            group_min = min(all_values)
            group_max = max(all_values)
            
            if group_name == 'worn_time':
                group_max = min(group_max, 15)
            
            y_padding = (group_max - group_min) * 0.1
            group_ylims[group_name] = (group_min - y_padding, group_max + y_padding)
        else:
            group_ylims[group_name] = (0, 1)
    
    for m in metrics:
        data = [authored[m].dropna()] + [authored.loc[authored['Label'] == lbl, m].dropna() for lbl in disease_labels]
        labels = disease_labels
        
        fig, ax = plt.subplots()
        ax.violinplot(data, showextrema=False)
        ax.boxplot(data, widths=0.1, positions=np.arange(1, len(data) + 1), patch_artist=True, showfliers=False)

        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, fontsize=14)
        ax.set_title(f'Week {week_num}: {m} by Disease', fontsize=14)
        ax.set_ylabel(m, fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        for group_name, group_metrics in metric_groups.items():
            if m in group_metrics:
                ax.set_ylim(group_ylims[group_name])
                break
        
        for i, d in enumerate(data, start=1):
            n_count = len(d)
            med = np.median(d)
            ax.text(i, med, f'Med={med:.2f}\n(N={n_count})', ha='center', va='bottom', fontsize=14)

        fig.savefig(os.path.join(output_dir, f'week{week_num}_violinbox_{m}.png'))
        plt.close(fig)

def generate_week_comparison_plots(disease_data_by_metric, disease_labels, output_dir, week1_df, week8_df, week1_real_patients, week8_real_patients):
    os.makedirs(output_dir, exist_ok=True)
    
    metric_groups = {
        'worn_time': ['avg_worn_min', 'avg_worn_max'],
        'recording_time': ['avg_recording'],
        'rates': ['valid_days_rate']
    }
    
    group_ylims = {}
    for group_name, group_metrics in metric_groups.items():
        all_group_values = []
        for metric_name in group_metrics:
            if metric_name in disease_data_by_metric:
                week_data = disease_data_by_metric[metric_name]
                week1_data = week_data[1]['data']
                week8_data = week_data[8]['data']
                
                for data_list in week1_data + week8_data:
                    if len(data_list) > 0:
                        all_group_values.extend(data_list.tolist())
        
        if all_group_values:
            group_min = min(all_group_values)
            group_max = max(all_group_values)
            
            if group_name == 'worn_time' or group_name == 'recording_time':
                group_max = 20
                group_min = 0
            
            y_padding = (group_max - group_min) * 0.1
            group_ylims[group_name] = (group_min - y_padding, group_max + y_padding)
        else:
            group_ylims[group_name] = (0, 1)
    
    for metric, week_data in disease_data_by_metric.items():
        plt.figure(figsize=(12, 8))
        
        week1_data = week_data[1]['data']
        week8_data = week_data[8]['data']
        week1_counts = week_data[1]['counts']
        week8_counts = week_data[8]['counts']
        week1_means = week_data[1]['means']
        week8_means = week_data[8]['means']
        
        current_y_limits = (0, 1)
        for group_name, group_metrics in metric_groups.items():
            if metric in group_metrics:
                current_y_limits = group_ylims[group_name]
                break
        
        positions = np.arange(1, len(disease_labels) + 1)
        width = 0.35
        
        plt.violinplot([d for d in week1_data if len(d) > 0], 
                      positions=positions-width/2, 
                      showextrema=False, 
                      widths=width*1.2)
        plt.violinplot([d for d in week8_data if len(d) > 0], 
                      positions=positions+width/2, 
                      showextrema=False, 
                      widths=width*1.2)
        
        box1 = plt.boxplot([d for d in week1_data if len(d) > 0], 
                         positions=positions-width/2, 
                         widths=width*0.3, 
                         patch_artist=True,
                         showfliers=False,
                         medianprops=dict(color='darkblue', linewidth=2))
        box2 = plt.boxplot([d for d in week8_data if len(d) > 0], 
                         positions=positions+width/2, 
                         widths=width*0.3, 
                         patch_artist=True,
                         showfliers=False,
                         medianprops=dict(color='darkred', linewidth=2))
        
        for box in box1['boxes']:
            box.set(facecolor='skyblue')
        for box in box2['boxes']:
            box.set(facecolor='salmon')
        
        plt.ylim(current_y_limits)
        

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='skyblue', label='Week 1'),
            Patch(facecolor='salmon', label='Week 8')
        ]
        plt.legend(handles=legend_elements, loc='upper left', ncol=2, fontsize=14)
        
        plt.xticks(positions, disease_labels, fontsize=14)
        plt.title(f'{metric} by Disease and Week', fontsize=14)
        plt.ylabel(metric, fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=14)
        
        table_data = []
        table_data.append(['Week 1', 'N'] + [str(n) for n in week1_counts])
        table_data.append(['', ''] + [f"{m:.2f}" for m in week1_means])
        table_data.append(['Week 8', 'N'] + [str(n) for n in week8_counts])
        table_data.append(['', ''] + [f"{m:.2f}" for m in week8_means])
        
        plt.subplots_adjust(bottom=0.2)
        table = plt.table(cellText=table_data,
                 colLabels=['', ''] + disease_labels,
                 loc='bottom',
                 bbox=[0, -0.30, 1, 0.22])
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        
        week1_valid_data = [d for d in week1_data if len(d) > 1]
        week8_valid_data = [d for d in week8_data if len(d) > 1]
        
        if len(week1_valid_data) > 1:
            _, p_val_w1 = stats.kruskal(*week1_valid_data)
        else:
            p_val_w1 = np.nan
            
        if len(week8_valid_data) > 1:
            _, p_val_w8 = stats.kruskal(*week8_valid_data)
        else:
            p_val_w8 = np.nan
            
        paired_test_results = {}
        paired_test_text = []
        
        for i, disease in enumerate(disease_labels):
            week1_disease_data = week1_df[week1_df['Label'] == disease]
            week8_disease_data = week8_df[week8_df['Label'] == disease]
            
            week1_real_disease = set(week1_disease_data['PATID']) & week1_real_patients
            week8_real_disease = set(week8_disease_data['PATID']) & week8_real_patients
            real_paired_patients = week1_real_disease & week8_real_disease
            
            if len(real_paired_patients) >= 3:
                week1_paired = week1_disease_data[week1_disease_data['PATID'].isin(real_paired_patients)].set_index('PATID')[metric]
                week8_paired = week8_disease_data[week8_disease_data['PATID'].isin(real_paired_patients)].set_index('PATID')[metric]
                
                week1_paired = week1_paired.reindex(real_paired_patients).dropna()
                week8_paired = week8_paired.reindex(real_paired_patients).dropna()
                
                valid_patients = week1_paired.index.intersection(week8_paired.index)
                
                if len(valid_patients) >= 3:
                    week1_values = week1_paired.loc[valid_patients]
                    week8_values = week8_paired.loc[valid_patients]
                    
                    try:
                        _, p_val_paired = stats.wilcoxon(week1_values, week8_values, alternative='two-sided')
                        paired_test_results[disease] = {
                            'p_value': p_val_paired,
                            'n_pairs': len(valid_patients),
                            'week1_median': week1_values.median(),
                            'week8_median': week8_values.median()
                        }
                        paired_test_text.append(f"{disease}: p={p_val_paired:.3f}")
                    except ValueError:
                        paired_test_text.append(f"{disease}: No variation")
                else:
                    paired_test_text.append(f"{disease}: Insufficient valid paired data")
            else:
                paired_test_text.append(f"{disease}: Insufficient real paired data")
        
        kw_text = f"Kruskal-Wallis Test p-values: Week 1: {p_val_w1:.3f}, Week 8: {p_val_w8:.3f}"
        paired_text = "Paired tests (Week 1 vs Week 8): " + "; ".join(paired_test_text)
        
        plt.figtext(0.5, -0.06, kw_text, ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.figtext(0.5, -0.12, paired_text, ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        posthoc_results = {}
        posthoc_text = []
        
        if not np.isnan(p_val_w1) and p_val_w1 < 0.05 and len(week1_valid_data) > 1:

            all_data_w1 = pd.DataFrame()
            for i, data in enumerate(week1_valid_data):
                group_df = pd.DataFrame({'value': data, 'group': disease_labels[i]})
                all_data_w1 = pd.concat([all_data_w1, group_df])
            
            posthoc_dunn_w1 = sp.posthoc_dunn(all_data_w1, val_col='value', group_col='group', p_adjust='bonferroni')
            posthoc_results['Week 1'] = posthoc_dunn_w1
            
            sig_pairs_w1 = []
            for i in range(len(posthoc_dunn_w1.columns)):
                for j in range(i+1, len(posthoc_dunn_w1.columns)):
                    p = posthoc_dunn_w1.iloc[i, j]
                    if p < 0.05:
                        sig_pairs_w1.append(f"{posthoc_dunn_w1.columns[i]} vs {posthoc_dunn_w1.columns[j]}: p={p:.3f}")
            
            if sig_pairs_w1:
                posthoc_text.append("Week 1 Significant Pairs:")
                posthoc_text.extend(sig_pairs_w1)
            
        if not np.isnan(p_val_w8) and p_val_w8 < 0.05 and len(week8_valid_data) > 1:

            all_data_w8 = pd.DataFrame()
            for i, data in enumerate(week8_valid_data):
                group_df = pd.DataFrame({'value': data, 'group': disease_labels[i]})
                all_data_w8 = pd.concat([all_data_w8, group_df])
            
            posthoc_dunn_w8 = sp.posthoc_dunn(all_data_w8, val_col='value', group_col='group', p_adjust='bonferroni')
            posthoc_results['Week 8'] = posthoc_dunn_w8
            
            sig_pairs_w8 = []
            for i in range(len(posthoc_dunn_w8.columns)):
                for j in range(i+1, len(posthoc_dunn_w8.columns)):
                    p = posthoc_dunn_w8.iloc[i, j]
                    if p < 0.05:
                        sig_pairs_w8.append(f"{posthoc_dunn_w8.columns[i]} vs {posthoc_dunn_w8.columns[j]}: p={p:.3f}")
            
            if sig_pairs_w8:
                posthoc_text.append("Week 8 Significant Pairs:")
                posthoc_text.extend(sig_pairs_w8)
        
        if posthoc_text:
            posthoc_str = '\n'.join(posthoc_text)
            plt.figtext(0.5, -0.25, posthoc_str, ha='center', fontsize=14, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            
            plt.gcf().set_size_inches(12, 12)
        else:
            plt.gcf().set_size_inches(12, 10)
        

        plt.savefig(os.path.join(output_dir, f'{metric}.png'), bbox_inches='tight')
        plt.close()
        
        if posthoc_results:
            for week, result_df in posthoc_results.items():
                result_df.to_csv(os.path.join(output_dir, f'{metric}_{week.replace(" ", "_")}_posthoc.csv'))
        
        if paired_test_results:
            paired_df = pd.DataFrame.from_dict(paired_test_results, orient='index')
            paired_df.index.name = 'Disease'
            paired_df.to_csv(os.path.join(output_dir, f'{metric}_paired_tests.csv'))

def main():
    divisors = {"PD": 72, "MSA": 44, "PSP": 51}

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "..", "data")
    plots_dir = os.path.join(script_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    excluded_patients = [
        'pat001', 'pat005', 'pat014', 'pat117', 'pat120', 'pat124', 'pat127', 'pat131', 
        'pat133', 'pat134', 'pat154', 'pat212', 'pat233', 'pat238', 'pat314', 'PAT402', 
        'pat403', 'pat407', 'PAT408', 'PAT411', 'pat412', 'PAT414', 'PAT113'
    ]

    print(f"Excluding {len(excluded_patients)} patients from passive monitoring analyses: {excluded_patients}")

    df = pd.read_excel(os.path.join(data_dir, "passive_monitoring.xlsx"))
    disease_df = pd.read_excel(os.path.join(data_dir, "RCT_Clinical_Data.xlsx"), sheet_name='Visit2')
    df['week'] = df['name'].str.extract(r'_week_(\d+)_').astype(int)

    excluded_patients_upper = [pat.upper() for pat in excluded_patients]
    print(f"Excluded patients (uppercase): {excluded_patients_upper}")
    
    print(f"Sample patient IDs in disease_df: {sorted(disease_df['PATID'].head(10).tolist())}")
    
    disease_df_filtered = disease_df[~disease_df['PATID'].isin(excluded_patients_upper)]
    
    print(f"Original clinical data: {len(disease_df)} patients")
    print(f"Filtered clinical data: {len(disease_df_filtered)} patients")
    print(f"Excluded from analysis: {len(excluded_patients)} patients")
    print(f"Actually excluded: {len(disease_df) - len(disease_df_filtered)} patients")
    
    found_excluded = set(disease_df['PATID']) & set(excluded_patients_upper)
    not_found_excluded = set(excluded_patients_upper) - set(disease_df['PATID'])
    print(f"Excluded patients found in data: {sorted(found_excluded)}")
    print(f"Excluded patients NOT found in data: {sorted(not_found_excluded)}")

    week1_df, week1_real_patients = process_week_data(df, disease_df_filtered, 1)
    week8_df, week8_real_patients = process_week_data(df, disease_df_filtered, 8)

    disease_data_by_metric = {}
    disease_labels = sorted(divisors.keys())
    
    print(f"\nWARNING: Original divisors were {divisors}")
    print(f"After excluding patients, actual patient counts may differ from these divisors")
    excluded_by_disease = disease_df[disease_df['PATID'].isin(excluded_patients_upper)]['Label'].value_counts()
    print(f"Excluded patients by disease: {dict(excluded_by_disease)}")

    for wk, authored_df in [(1, week1_df), (8, week8_df)]:
        output_dir = os.path.join(script_dir, f"week{wk}")
        os.makedirs(output_dir, exist_ok=True)
        authored_df.to_csv(os.path.join(output_dir, f'week{wk}_detailed_metrics.csv'), index=False)
        
        valid_summary = authored_df[['PATID', 'Label', 'valid_days_rate']]
        valid_summary.to_csv(os.path.join(output_dir, f'week{wk}_validity_rate.csv'), index=False)

        sum_rates = authored_df.groupby('Label')['valid_days_rate'].sum()
        adherence = (sum_rates / pd.Series(divisors) * 100).round(2)
        adherence_df = adherence.reset_index().rename(columns={0: 'Adherence%'})
        adherence_df.to_csv(os.path.join(output_dir, f'week{wk}_adherence_by_label.csv'), index=False)
        
        generate_weekly_plots(authored_df, wk, output_dir, divisors, disease_data_by_metric)


    generate_week_comparison_plots(disease_data_by_metric, disease_labels, plots_dir, week1_df, week8_df, week1_real_patients, week8_real_patients)

    print(f"\n" + "="*60)
    print(f"PASSIVE MONITORING: CROSS-WEEK PADDING SUMMARY")
    print("="*60)
    
    all_patients = set(disease_df_filtered['PATID'])  # Use filtered data
    patients_with_data_both_weeks = week1_real_patients & week8_real_patients
    patients_with_data_week1_only = week1_real_patients - week8_real_patients
    patients_with_data_week8_only = week8_real_patients - week1_real_patients
    patients_with_no_data = all_patients - (week1_real_patients | week8_real_patients)
    
    print(f"Total patients in filtered clinical data: {len(all_patients)}")
    print(f"Patients with real data in BOTH weeks: {len(patients_with_data_both_weeks)}")
    print(f"Patients with real data ONLY in Week 1: {len(patients_with_data_week1_only)}")
    print(f"Patients with real data ONLY in Week 8: {len(patients_with_data_week8_only)}")
    print(f"Patients with NO real data (completely padded): {len(patients_with_no_data)}")
    
    if patients_with_no_data:
        completely_padded_diseases = disease_df_filtered[disease_df_filtered['PATID'].isin(patients_with_no_data)]['Label'].value_counts()
        print(f"\nCompletely padded patients by disease:")
        for disease, count in completely_padded_diseases.items():
            print(f"  {disease}: {count} patients")
        print(f"Completely padded patient IDs: {sorted(list(patients_with_no_data))}")

    print("Passive analysis done for weeks 1 and 8.")
    print("Week comparison plots with paired statistical tests generated in ./plots directory.")
    print("Paired test results (Week 1 vs Week 8 within each disease) saved as CSV files.")
    print(f"Real patients Week 1: {len(week1_real_patients)}, Week 8: {len(week8_real_patients)}")
    print(f"Patients with real data in both weeks: {len(week1_real_patients & week8_real_patients)}")

if __name__ == '__main__':
    main()
