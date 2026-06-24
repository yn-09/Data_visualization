import streamlit as st
import pandas as pd
from dashboard import (
    process_it_data, 
    draw_recommendation_matrix, 
    draw_top_desire_tasks, 
    draw_human_constraints_heatmap, 
    draw_gap_analysis_dumbbell
)

st.set_page_config(
    page_title="AI Agent Analysis",
    page_icon="🤖",
    layout="wide"
)

st.title("Phân tích & Khuyến nghị Ứng dụng AI Agent trong ngành IT")
st.markdown("Hỗ trợ ra quyết định chiến lược tích hợp AI Agent dựa trên góc nhìn Chuyên gia và Người lao động.")
st.write("---")

@st.cache_data
def load_and_preprocess_all_data():
    df1 = pd.read_csv('domain_worker_desires.csv')
    df2 = pd.read_csv('expert_rated_technological_capability.csv')
    df3 = pd.read_csv('task_statement_with_metadata.csv')
    df4 = pd.read_csv('domain_worker_metadata.csv')
    
    for df in [df1, df2, df3, df4]:
        df.columns = df.columns.str.strip().str.replace('\n', ' ', regex=True)
        df.drop_duplicates(inplace=True)
        
    df_it_final = process_it_data(df1, df2, df3, df4)
    return df_it_final

try:
    df_it_final = load_and_preprocess_all_data()
except Exception as e:
    st.error(f"Lỗi đọc dữ liệu: {e}")
    st.stop()

# SIDEBAR LỌC CHUNG
st.sidebar.header("Bộ Lọc Tổng Thể")
occupation_options = ["Tất cả các ngành IT"] + list(df_it_final['Occupation (O*NET-SOC Title)'].unique())
selected_occupation = st.sidebar.selectbox("Chọn vị trí công việc:", options=occupation_options)

# KPI Metrics hiển thị nhanh
col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Tổng số Task IT", value=df_it_final.shape[0])
with col2:
    current_cnt = df_it_final[df_it_final['Occupation (O*NET-SOC Title)'] == selected_occupation].shape[0] if selected_occupation != "Tất cả các ngành IT" else df_it_final.shape[0]
    st.metric(label="Số lượng Task đang chọn", value=current_cnt)
with col3:
    quick_wins_cnt = df_it_final[(df_it_final['Automation Capacity Rating'] >= 3.0) & (df_it_final['Automation Desire Rating'] >= 3.0)].shape[0]
    st.metric(label="Số Task Quick Wins", value=quick_wins_cnt)

st.write("---")

# ==============================================================================
# KHỐI 1: BẢNG TRA CỨU CHI TIẾT
# ==============================================================================
st.subheader("📋 Tra cứu chi tiết thông tin các Task công việc")
display_cols = ['Occupation (O*NET-SOC Title)', 'Task', 'Skill (O*NET Work Activity)', 'Automation Capacity Rating', 'Automation Desire Rating']

# Tái cấu trúc phân bổ bảng nguồn dữ liệu trước
# ĐÃ SỬA: Đưa slider xuống nằm ngay bên dưới bảng hiển thị và căn lề giữa bằng columns
df_table_source = df_it_final[df_it_final['Occupation (O*NET-SOC Title)'] == selected_occupation if selected_occupation != "Tất cả các ngành IT" else slice(None)]

# Định vị mặc định hoặc qua tương tác biến trạng thái session để giữ trạng thái lọc gọn gàng
if 'min_desire_val' not in st.session_state:
    st.session_state.min_desire_val = 1.0

df_filtered_view = df_table_source[df_table_source['Automation Desire Rating'] >= st.session_state.min_desire_val]
st.dataframe(df_filtered_view[display_cols].sort_values(by='Automation Desire Rating', ascending=False), use_container_width=True)

# ĐẨY SLICER XUỐNG DƯỚI CÙNG VÀ CĂN GIỮA
col_l1, col_center1, col_r1 = st.columns([1, 1.5, 1])
with col_center1:
    st.session_state.min_desire_val = st.slider("Bộ lọc: Mức mong muốn tối thiểu (Tra cứu)", 1.0, 5.0, st.session_state.min_desire_val, 0.1)

st.write("---")

# ==============================================================================
# KHỐI 2: MA TRẬN KHUYẾN NGHỊ
# ==============================================================================
st.subheader("🎯 Ma Trận Khuyến Nghị Tích Hợp AI Agent")
st.pyplot(draw_recommendation_matrix(df_it_final, selected_occupation))

st.markdown("#### 📝 Nhận xét phân tích chiến lược tích hợp:")
st.info(
    "**1. Vùng Quick Wins:** AI làm tốt và nhân sự muốn giao việc. Triển khai ngay **AI Agent tự động hoàn toàn**.\n\n"
    "**2. Vùng AI Assisted / Copilot:** AI xử lý tốt nhưng nhân sự còn e ngại. Khuyến nghị cấu hình dạng **Trợ lý (Copilot)**.\n\n"
    "**3. Vùng Tech Challenges:** Nhân sự khao khát ứng dụng nhưng công nghệ chưa đáp ứng đủ. Cần đầu tư chuyên sâu."
)
st.write("---")

# ==============================================================================
# KHỐI 3: BIỂU ĐỒ TOP TASK KHAO KHÁT (Slicer nằm dưới Chart và căn giữa)
# ==============================================================================
st.subheader("🔥 Phân Tích Sâu: Điểm Nóng Quá Tải của Kỹ sư IT")

if 'top_n_val' not in st.session_state:
    st.session_state.top_n_val = 10

st.pyplot(draw_top_desire_tasks(df_it_final, selected_occupation, top_n=st.session_state.top_n_val))

# ĐẨY SLICER XUỐNG DƯỚI BIỂU ĐỒ VÀ CĂN GIỮA BẰNG COLUMNS
col_l2, col_center2, col_r2 = st.columns([1, 1.5, 1])
with col_center2:
    st.session_state.top_n_val = st.slider("Bộ lọc: Thay đổi số lượng tác vụ hiển thị trên biểu đồ", min_value=5, max_value=20, value=st.session_state.top_n_val, step=1)

st.markdown("#### 📝 Nhận xét phát triển tính năng cho AI Agent:")
col_ins1, col_ins2, col_ins3 = st.columns(3)
with col_ins1:
    st.success("**Thanh Xanh Lá (Repetitive) cao:**\n\n-> Thiết kế dạng **Automation Workflow / Scripting** (Ví dụ: tạo boilerplate, tự sinh tài liệu).")
with col_ins2:
    st.info("**Thanh Xanh Dương (Human Error) cao:**\n\n-> Định hình AI dạng **Gatekeeper / Lớp kiểm định** (Ví dụ: rà quét mã nguồn, review logic PR).")
with col_ins3:
    st.warning("**Thanh Vàng Cam (Stress) cao:**\n\n-> Ưu tiên build hệ thống phản ứng khẩn cấp cứu hộ **Auto-Incident Responder** đêm.")

st.write("---")

# ==============================================================================
# KHỐI 4: BẢN ĐỒ NHIỆT RÀNG BUỘC
# ==============================================================================
st.subheader("🚧 Khảo Sát Điểm Nghẽn & Ràng Buộc Vai Trò Con Người")
st.pyplot(draw_human_constraints_heatmap(df_it_final, selected_occupation))

st.markdown("#### 📝 Nhận xét cấu hình ràng buộc hệ thống:")
col_str1, col_str2 = st.columns(2)
with col_str1:
    st.error("🚨 **VÙNG ĐỎ ĐẬM (> 3.5):** Ràng buộc cực lớn về độ bất định/chuyên môn. Nghiêm cấm Auto-approve, bắt buộc áp dụng cơ chế **Human-in-the-loop**.")
with col_str2:
    st.success("🟢 **VÙNG VÀNG NHẠT (< 2.5):** Không có rào cản giao tiếp/vật lý. Cơ hội vàng triển khai **Autonomous AI Agent** tự trị qua Webhook/API.")
st.write("---")

# ==============================================================================
# KHỐI 5: BIỂU ĐỒ TẠ ĐÔI (Màu sắc chuẩn Cam - Xanh cố định)
# ==============================================================================
st.subheader("⚖️ Phân Tích Điểm Mù (Gap Analysis) Giữa Kỹ Thuật & Con Người")
st.pyplot(draw_gap_analysis_dumbbell(df_it_final, selected_occupation))

st.markdown("#### 📝 Nhận xét giải mã khoảng cách & Chiến lược giải pháp:")
st.info("🔵 **Điểm Xanh Dương nằm bên PHẢI điểm Cam:** AI làm được nhưng worker không muốn. Lý do là e ngại hoặc UX tệ. Cần tập trung cải thiện UI/UX và truyền thông giá trị trợ lực.")
st.warning("🟠 **Điểm Cam nằm bên PHẢI điểm Xanh Dương:** Worker rất thèm khát nhưng công nghệ chưa tới. Tuyệt đối không dùng LLM phổ thông đại trà vì dễ sinh lỗi hệ thống, cần build mô hình Fine-tuned độc quyền.")