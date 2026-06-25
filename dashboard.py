import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Cấu hình font hệ thống tránh lỗi hiển thị ký tự
plt.rcParams['font.family'] = 'Arial'

def process_it_data(df1, df2, df3, df4):
    it_occupations = df3[df3['O*NET-SOC Code'].str.startswith('15-', na=False)]['Occupation (O*NET-SOC Title)'].unique()
    
    df1_it = df1[df1['Occupation (O*NET-SOC Title)'].isin(it_occupations)].copy()
    df2_it = df2[df2['Occupation (O*NET-SOC Title)'].isin(it_occupations)].copy()
    df3_it = df3[df3['Occupation (O*NET-SOC Title)'].isin(it_occupations)].copy()
    
    reason_cols = [
        'Reasons for Automation Desire - Repetitive',
        'Reasons for Automation Desire - Human Error',
        'Reasons for Automation Desire - Stress',
        'Reasons for Automation Desire - Difficulty',
        'Reasons for Automation Desire - Scale'
    ]
    for col in reason_cols:
        if col in df1_it.columns:
            df1_it[col] = df1_it[col].astype(str).str.strip().str.upper().map({'TRUE': 1, '1': 1, 'FALSE': 0, '0': 0})
            df1_it[col] = df1_it[col].fillna(0).astype(int)

    expert_task_metrics = df2_it.groupby('Task ID').agg({
        'Automation Capacity Rating': 'mean',
        'Physical Action Requirement': 'mean',
        'Involved Uncertainty': 'mean',
        'Domain Expertise Requirement': 'mean',
        'Interpersonal Communication Requirement': 'mean',
        'Human Agency Scale Rating': 'mean'
    }).reset_index()

    worker_task_metrics = df1_it.groupby('Task ID').agg({
        'Automation Desire Rating': 'mean',
        'Time': 'mean',
        'Core Skill Rating': 'mean',
        'Job Security Rating': 'mean',
        'Enjoyment Rating': 'mean',
        'Reasons for Automation Desire - Repetitive': 'mean',
        'Reasons for Automation Desire - Human Error': 'mean',
        'Reasons for Automation Desire - Stress': 'mean',
        'Reasons for Automation Desire - Difficulty': 'mean',
        'Reasons for Automation Desire - Scale': 'mean'
    }).reset_index()

    worker_task_metrics.rename(columns={
        'Reasons for Automation Desire - Repetitive': 'Pct_Reason_Repetitive',
        'Reasons for Automation Desire - Human Error': 'Pct_Reason_Human_Error',
        'Reasons for Automation Desire - Stress': 'Pct_Reason_Stress',
        'Reasons for Automation Desire - Difficulty': 'Pct_Reason_Difficulty',
        'Reasons for Automation Desire - Scale': 'Pct_Reason_Scale'
    }, inplace=True)

    df_it_final = df3_it[['O*NET-SOC Code', 'Occupation (O*NET-SOC Title)', 'Task ID', 'Task', 'Skill (O*NET Work Activity)']].copy()
    df_it_final = pd.merge(df_it_final, expert_task_metrics, on='Task ID', how='inner')
    df_it_final = pd.merge(df_it_final, worker_task_metrics, on='Task ID', how='inner')
    
    return df_it_final

def draw_recommendation_matrix(df_data, selected_occupation=None):
    if selected_occupation and selected_occupation != "Tất cả các ngành IT":
        df_plot = df_data[df_data['Occupation (O*NET-SOC Title)'] == selected_occupation]
    else:
        df_plot = df_data

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.set_theme(style="whitegrid")
    
    if df_plot.empty:
        ax.text(0.5, 0.5, 'Không có dữ liệu phù hợp', ha='center', va='center')
        return fig

    df_distinct_task = df_plot.drop_duplicates(subset=['Task ID'])

    scatter = sns.scatterplot(
        data=df_distinct_task,
        x='Automation Capacity Rating',
        y='Automation Desire Rating',
        hue='Domain Expertise Requirement',
        palette='viridis',
        size='Core Skill Rating',
        sizes=(25, 120),
        alpha=0.65,
        ax=ax
    )

    ax.axvline(x=3.0, color='darkgray', linestyle='--', linewidth=0.8)
    ax.axhline(y=3.0, color='darkgray', linestyle='--', linewidth=0.8)

    ax.text(4.1, 4.5, 'CƠ HỘI TRIỂN KHAI', fontsize=8, fontweight='bold', color='green', ha='center')
    ax.text(4.1, 1.5, 'AI TRỢ LÝ HỖ TRỢ', fontsize=8, fontweight='bold', color='blue', ha='center')
    ax.text(1.9, 4.5, 'THÁCH THỨC CÔNG NGHỆ', fontsize=8, fontweight='bold', color='orange', ha='center')
    ax.text(1.9, 1.5, 'VAI TRÒ CỐT LÕI CON NGƯỜI', fontsize=8, fontweight='bold', color='red', ha='center')

    ax.set_title('BẢN ĐỒ PHÂN VỊ KHUYẾN NGHỊ TÍCH HỢP AI AGENT', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Khả năng tự động hóa (Chuyên gia)', fontsize=8)
    ax.set_ylabel('Mức độ mong muốn (Nhân sự)', fontsize=8)

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.set_xlim(0.8, 5.2)
    ax.set_ylim(0.8, 5.2)
    
    # ĐƯA CHÚ THÍCH RA NGOÀI BIÊN BÊN PHẢI (bbox_to_anchor x=1.02)
    ax.legend(title='Yêu cầu chuyên môn', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, fontsize=7, title_fontsize=8)
    plt.tight_layout()
    
    return fig

def draw_top_desire_tasks(df_data, selected_occupation=None, top_n=10):
    if selected_occupation and selected_occupation != "Tất cả các ngành IT":
        df_plot = df_data[df_data['Occupation (O*NET-SOC Title)'] == selected_occupation]
    else:
        df_plot = df_data

    if df_plot.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
        return fig

    df_top = df_plot.drop_duplicates(subset=['Task ID']).sort_values(by='Automation Desire Rating', ascending=False).head(top_n).copy()
    df_top['Task_Short'] = df_top['Task'].apply(lambda x: x[:35] + '...' if len(x) > 35 else x)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    sns.set_theme(style="white")

    r_repetitive = df_top['Pct_Reason_Repetitive']
    r_error = df_top['Pct_Reason_Human_Error']
    r_stress = df_top['Pct_Reason_Stress']

    sns.barplot(x=r_repetitive + r_error + r_stress, y='Task_Short', data=df_top, label='Áp lực tinh thần', color='#f39c12', ax=ax)
    sns.barplot(x=r_repetitive + r_error, y='Task_Short', data=df_top, label='Rủi ro sai sót', color='#3498db', ax=ax)
    sns.barplot(x=r_repetitive, y='Task_Short', data=df_top, label='Tính lặp lại', color='#2ecc71', ax=ax)

    ax.set_title(f'TOP {top_n} TASK IT KHAO KHÁT GIẢI PHÓNG BẰNG AI', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('Trọng số động lực tích hợp', fontsize=8)
    ax.set_ylabel('', fontsize=8)

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    # ĐƯA CHÚ THÍCH RA NGOÀI BIÊN BÊN PHẢI (bbox_to_anchor x=1.02)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', title='Lý do', borderaxespad=0, fontsize=7, title_fontsize=8)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
    
    return fig

def draw_human_constraints_heatmap(df_data, selected_occupation=None):
    if selected_occupation and selected_occupation != "Tất cả các ngành IT":
        df_plot = df_data[df_data['Occupation (O*NET-SOC Title)'] == selected_occupation]
    else:
        df_plot = df_data

    if df_plot.empty:
        fig, ax = plt.subplots(figsize=(7.5, 3.5))
        ax.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
        return fig

    human_factors = {
        'Physical Action Requirement': 'Vật lý',
        'Interpersonal Communication Requirement': 'Giao tiếp',
        'Involved Uncertainty': 'Bất định',
        'Domain Expertise Requirement': 'Chuyên môn'
    }

    top_skills = df_plot['Skill (O*NET Work Activity)'].value_counts().head(6).index
    df_skills = df_plot[df_plot['Skill (O*NET Work Activity)'].isin(top_skills)]
    
    heatmap_data = df_skills.groupby('Skill (O*NET Work Activity)')[list(human_factors.keys())].mean()
    heatmap_data.rename(columns=human_factors, inplace=True)
    heatmap_data.index = [idx[:30] + '...' if len(idx) > 30 else idx for idx in heatmap_data.index]

    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    
    sns.heatmap(
        heatmap_data, 
        annot=True, 
        fmt=".2f", 
        cmap="YlOrRd", 
        linewidths=.4, 
        cbar_kws={'label': 'Điểm số', 'shrink': 0.6}, 
        ax=ax
    )

    ax.set_title('BẢN ĐỒ NHIỆT RÀNG BUỘC VAI TRÒ CON NGƯỜI', fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel('', fontsize=8)
    ax.set_ylabel('', fontsize=8)

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    
    plt.xticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    return fig

def draw_gap_analysis_dumbbell(df_data, selected_occupation=None):
    if selected_occupation and selected_occupation != "Tất cả các ngành IT":
        df_plot = df_data[df_data['Occupation (O*NET-SOC Title)'] == selected_occupation]
    else:
        df_plot = df_data

    if df_plot.empty:
        fig, ax = plt.subplots(figsize=(7.5, 4))
        ax.text(0.5, 0.5, 'Không có dữ liệu', ha='center', va='center')
        return fig

    top_skills = df_plot['Skill (O*NET Work Activity)'].value_counts().head(8).index
    df_skills = df_plot[df_plot['Skill (O*NET Work Activity)'].isin(top_skills)]
    
    gap_data = df_skills.groupby('Skill (O*NET Work Activity)').agg({
        'Automation Capacity Rating': 'mean',
        'Automation Desire Rating': 'mean'
    }).reset_index()

    gap_data['Skill_Short'] = gap_data['Skill (O*NET Work Activity)'].apply(lambda x: x[:30] + '...' if len(x) > 30 else x)
    gap_data = gap_data.sort_values(by='Automation Capacity Rating', ascending=True)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.set_theme(style="whitegrid")

    ax.hlines(y=gap_data['Skill_Short'], xmin=gap_data['Automation Capacity Rating'], 
              xmax=gap_data['Automation Desire Rating'], color='#b2bec3', linewidth=1.2, alpha=0.5)

    # ĐÃ ĐỔI: Ép cứng chỉ định mã màu HEX chuẩn Cam (#e17055) và Xanh Dương (#0984e3)
    ax.scatter(gap_data['Automation Capacity Rating'], gap_data['Skill_Short'], 
               color='#0984e3', s=60, label='Chuyên gia (AI Capacity)', zorder=3)

    ax.scatter(gap_data['Automation Desire Rating'], gap_data['Skill_Short'], 
               color="#6A1591", s=60, label='Nhân sự IT (Worker Desire)', zorder=3)

    ax.set_title('PHÂN TÍCH LỆCH PHA (GAP ANALYSIS) GIỮA KỸ THUẬT & CON NGƯỜI', fontsize=10, fontweight='bold', pad=12)
    ax.set_xlabel('Thang điểm [1 -> 5]', fontsize=8)
    ax.set_ylabel('', fontsize=8)

    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)

    ax.set_xlim(1.0, 5.0)

    
    # Đẩy Legend lên phía trên cùng của đồ thị
    ax.legend(title='Nhóm đánh giá', bbox_to_anchor=(1.02, 1), loc='upper left', ncol=1, frameon=True, fontsize=7, title_fontsize=8)
    sns.despine(left=True, bottom=True)
    plt.tight_layout()
      
    return fig


