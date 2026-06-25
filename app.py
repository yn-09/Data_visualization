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

# ==============================================================================
# THANH SIDEBAR: BỘ LỌC & MỤC LỤC ĐIỀU HƯỚNG TỰ ĐỘNG TRƯỢT (ANCHOR LINK)
# ==============================================================================
st.sidebar.header("Bộ Lọc Tổng Thể")
occupation_options = ["Tất cả các ngành IT"] + list(df_it_final['Occupation (O*NET-SOC Title)'].unique())
selected_occupation = st.sidebar.selectbox("Chọn vị trí công việc:", options=occupation_options)

st.sidebar.write("---")
st.sidebar.header("📌 Mục lục báo cáo")

st.sidebar.markdown("""
* [1. Thống kê tổng quan (KPIs)](#kpis)
* [2. Tra cứu chi tiết Task công việc](#khoi1)
* [3. Ma Trận Khuyến Nghị Tích Hợp AI](#khoi2)
* [4. Điểm Nóng Quá Tải của Kỹ sư IT](#khoi3)
* [5. Điểm Nghẽn & Ràng Buộc Con Người](#khoi4)
* [6. Phân Tích Điểm Mù (Gap Analysis)](#khoi5)
""", unsafe_allow_html=True)


# ==============================================================================
# KHỐI TỔNG QUAN (KPIs)
# ==============================================================================
st.markdown("<div id='kpis'></div>", unsafe_allow_html=True)
st.subheader("Thống kê tổng quan (KPIs)")

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
st.markdown("<div id='khoi1'></div>", unsafe_allow_html=True)
st.subheader("Tra cứu chi tiết thông tin các Task công việc")

display_cols = ['Occupation (O*NET-SOC Title)', 'Task', 'Skill (O*NET Work Activity)', 'Automation Capacity Rating', 'Automation Desire Rating']
df_table_source = df_it_final[df_it_final['Occupation (O*NET-SOC Title)'] == selected_occupation if selected_occupation != "Tất cả các ngành IT" else slice(None)]

if 'min_desire_val' not in st.session_state:
    st.session_state.min_desire_val = 1.0

df_filtered_view = df_table_source[df_table_source['Automation Desire Rating'] >= st.session_state.min_desire_val]
st.dataframe(df_filtered_view[display_cols].sort_values(by='Automation Desire Rating', ascending=False), width='stretch')

col_l1, col_center1, col_r1 = st.columns([1, 1.5, 1])
with col_center1:
    st.session_state.min_desire_val = st.slider("Bộ lọc: Mức mong muốn tối thiểu (Tra cứu)", 1.0, 5.0, st.session_state.min_desire_val, 0.1)

st.write("---")

# ==============================================================================
# KHỐI 2: MA TRẬN KHUYẾN NGHỊ
# ==============================================================================
st.markdown("<div id='khoi2'></div>", unsafe_allow_html=True)
st.subheader("Ma Trận Khuyến Nghị Tích Hợp AI Agent")
st.pyplot(draw_recommendation_matrix(df_it_final, selected_occupation))

st.markdown("### Nhận xét nhanh theo 4 vùng phân vị")
st.markdown(
    "**Cơ hội triển khai (Góc trên - phải):** Vùng 'thắng nhanh' (quick-wins). "
    "Công nghệ đã sẵn sàng và nhân sự rất hào hứng đón nhận. Cần triển khai ngay để tối ưu hiệu suất và tạo động lực chuyển đổi số.\n\n"
    "**AI trợ lý hỗ trợ (Góc dưới - phải):** Nghịch lý 'công nghệ chờ người'. "
    "Khả năng tự động hóa rất cao nhưng nhân sự lại e dè, ngại sử dụng (lo bị thay thế hoặc giao diện khó dùng).\n\n"
    "**Thách thức công nghệ (Góc trên - trái):** Nhân sự rất mong muốn có AI hỗ trợ "
    "nhưng quy trình hoặc dữ liệu hiện tại còn quá phức tạp, chưa thể đóng gói tự động hóa ngay.\n\n"
    "**Vai trò cốt lõi con người (Góc dưới - trái):** Vùng an toàn tuyệt đối. "
    "Cả chuyên gia lẫn nhân sự đều đồng thuận giữ nguyên yếu tố con người (tư duy chiến lược, kỹ năng mềm, cảm xúc)."
)

st.markdown("### Đề xuất phương pháp cải thiện tình trạng hiện tại")
st.markdown("Để tối ưu hóa được điều này, chúng ta cần thực hiện hai chiến lược dịch chuyển cốt lõi:")

st.markdown("**1. Thúc đẩy nhân sự (Dịch chuyển từ Dưới lên Trên tại vùng 'AI trợ lý')**")
st.markdown(
    "* **Quản trị sự thay đổi:** Đào tạo lại nhận thức, định vị AI là 'trợ lý' giúp giảm tải việc nhàm chán chứ không phải để thay thế con người.\n"
    "* **Phương pháp Co-design:** Cho nhân sự tham gia thử nghiệm và góp ý hoàn thiện giao diện (UI/UX) để tăng độ thân thiện và tính cá nhân hóa.\n"
    "* **Cơ chế thi đua:** Đưa tỷ lệ áp dụng công cụ vào KPI hoặc có phần thưởng ngắn hạn để kích thích tinh thần chủ động sử dụng."
)

st.markdown("**2. Nâng cấp kỹ thuật (Dịch chuyển từ Trái sang Phải tại vùng 'Thách thức công nghệ')**")
st.markdown(
    "* **Chuẩn hóa quy trình (SOP):** Rà soát, cấu trúc lại các bước làm việc một cách logic, loại bỏ các yếu tố cảm tính trước khi đưa AI vào.\n"
    "* **Xây dựng cơ sở tri thức:** Thu thập tài liệu, kinh nghiệm của chuyên gia để làm dữ liệu huấn luyện (RAG/Fine-tuning) cho AI.\n"
    "* **Chia nhỏ bài toán:** Bóc tách quy trình lớn thành các tác vụ nhỏ, ưu tiên tự động hóa những phần dễ trước thay vì làm dàn trải."
)

st.write("---")

# ==============================================================================
# KHỐI 3: BIỂU ĐỒ TOP TASK KHAO KHÁT
# ==============================================================================
st.markdown("<div id='khoi3'></div>", unsafe_allow_html=True)
st.subheader("Phân Tích Sâu: Điểm Nóng Quá Tải của Kỹ sư IT")

if 'top_n_val' not in st.session_state:
    st.session_state.top_n_val = 10

st.pyplot(draw_top_desire_tasks(df_it_final, selected_occupation, top_n=st.session_state.top_n_val))

col_l2, col_center2, col_r2 = st.columns([1, 1.5, 1])
with col_center2:
    st.session_state.top_n_val = st.slider("Bộ lọc: Thay đổi số lượng tác vụ hiển thị trên biểu đồ", min_value=5, max_value=20, value=st.session_state.top_n_val, step=1)

st.markdown("### Nhận xét biểu đồ")
st.markdown(
    "Biểu đồ cho thấy áp lực của kỹ sư IT đến từ ba nguyên nhân chính: Tính lặp lại, Rủi ro sai sót, và Áp lực tinh thần. Dữ liệu phân hóa rõ thành các nhóm:\n\n"
    "**Vùng quá tải nghiêm trọng (Trọng số > 2.0):** Tác vụ Giám sát vận hành hệ thống (*Monitor system operation...*) và "
    "Cài đặt/Cấu hình máy chủ web (*Install or configure Web server...*) đứng đầu bảng. Đây là những công việc vừa tẻ nhạt (lặp lại cao), "
    "vừa cực kỳ căng thẳng vì chỉ một lỗi cấu hình nhỏ (Human Error) cũng có thể làm sập hệ thống.\n\n"
    "**Vùng rủi ro sai sót cao:** Các công việc liên quan đến dữ liệu và báo cáo như Báo cáo phân tích thống kê (*Report results...*) "
    "hay Đánh giá kế hoạch dự án (*Evaluate project plans...*) có tỷ lệ xanh dương chiếm chủ đạo. Việc xử lý số liệu thủ công rất dễ dẫn đến nhầm lẫn.\n\n"
    "**Vùng lặp lại nhàm chán:** Các tác vụ như Ghi chép dữ liệu hàng ngày (*Maintain records...*) hay Sao lưu dữ liệu (*Back up files...*) "
    "hầu như chỉ thuần tính lặp lại (thanh xanh lá kéo dài). Chúng ít gây stress ngay lập tức nhưng lại tiêu tốn nhiều thời gian vô ích."
)

st.markdown("### Đề xuất giải pháp")

st.markdown("**1. Mô hình \"AI canh gác - Con người xử lý\" (Giải quyết vùng quá tải)**")
st.markdown(
    "Đối với việc giám sát hệ thống, thay đổi quy trình từ \"con người ngồi canh màn hình\" sang giao cho công cụ quét lỗi tự động 24/7. "
    "Khi và chỉ khi hệ thống có sự cố vượt ngưỡng an toàn, máy mới gửi cảnh báo để kỹ sư nhảy vào xử lý. Cách này giúp triệt tiêu hoàn toàn sự căng thẳng (Stress) túc trực."
)

st.markdown("**2. Quy trình \"Con người phác thảo - AI rà soát\" (Giải quyết vùng sai sót)**")
st.markdown(
    "Đối với các tác vụ làm báo cáo hoặc đánh giá kế hoạch, áp dụng nguyên lý kiểm tra chéo (Double-check). Con người đưa ra các phân tích, "
    "định hướng ban đầu, sau đó nạp dữ liệu vào AI để nó đóng vai trò \"Kỹ thuật viên rà soát\" nhằm phát hiện nhanh các lỗi logic hoặc sai số."
)

st.markdown("**3. Tự động hóa lịch trình tuyệt đối (Giải quyết vùng lặp lại)**")
st.markdown(
    "Đối với việc ghi chép và sao lưu dữ liệu, đề xuất đóng gói toàn bộ các bước thành một quy trình chuẩn (quy định rõ định dạng file, "
    "khung giờ chạy, nơi lưu trữ). Sau đó cài đặt cho máy tính tự động kích hoạt thực hiện theo lịch (Automation) mà không cần con người can thiệp thủ công hàng ngày nữa."
)

st.write("---")

# ==============================================================================
# KHỐI 4: BẢN ĐỒ NHIỆT RÀNG BUỘC
# ==============================================================================
st.markdown("<div id='khoi4'></div>", unsafe_allow_html=True)
st.subheader("Khảo Sát Điểm Nghẽn & Ràng Buộc Vai Trò Con Người")
st.pyplot(draw_human_constraints_heatmap(df_it_final, selected_occupation))

st.markdown("### Nhận xét biểu đồ")
st.markdown(
    "Biểu đồ nhiệt thể hiện mức độ ràng buộc của con người đối với các hoạt động dựa trên 4 khía cạnh tiêu chí. Dữ liệu phân hóa rõ rệt thành các vùng trọng điểm sau:\n\n"
    "**Điểm nghẽn Chuyên môn (Cột đỏ đậm nhất):** Đây là rào cản lớn nhất đối với mọi hoạt động. "
    "Đặc biệt, hai kỹ năng đòi hỏi tư duy cao là Tư duy sáng tạo (*Thinking Creatively - 3.42*) và Phân tích dữ liệu (*Analyzing Data... - 3.32*), "
    "đi kèm với khả năng Làm việc với máy tính (*Working with Computers - 3.20*) có điểm số cao nhất. Điều này chứng tỏ công việc phụ thuộc "
    "cực kỳ lớn vào trình độ và tri thức chuyên sâu của nhân sự, rất khó để thay thế hoàn toàn.\n\n"
    "**Yếu tố Bất định và Giao tiếp (Vùng cam trung bình):** Khía cạnh Bất định gây ràng buộc mạnh lên tác vụ Giám sát quy trình "
    "(*Monitoring Processes... - 2.52*) và Phân tích dữ liệu (*2.45*) vì môi trường biến động liên tục buộc con người phải ra quyết định. "
    "Trong khi đó, Giao tiếp phân bổ đều và gây áp lực vừa phải (từ 1.63 đến 2.27) lên tất cả các tác vụ.\n\n"
    "**Vùng ràng buộc Vật lý thấp (Cột màu vàng nhạt):** Tất cả các hoạt động đều có điểm số rất thấp ở khía cạnh Vật lý (đều dưới 1.6). "
    "Cho thấy đây thuần túy là các công việc văn phòng, thao tác trí óc, ít bị giới hạn bởi không gian hay sức lực cơ bắp."
)

st.markdown("### Đề xuất giải pháp (Ý tưởng hệ thống trong tầm tư duy sinh viên)")

st.markdown("**1. Hạ nhiệt điểm nghẽn Chuyên môn**")
st.markdown(
    "Đối với các tác vụ đòi hỏi tư duy cao như phân tích dữ liệu hay sáng tạo, đề xuất giao cho AI Agent đảm nhận phần việc thô như thu thập thông tin "
    "và gợi ý ý tưởng nền tảng. Con người sẽ đóng vai trò kiểm duyệt, dùng kinh nghiệm thực tế để chọn lọc và đưa ra quyết định sáng tạo cuối cùng."
)

st.markdown("**2. Giảm áp lực khía cạnh Bất định**")
st.markdown(
    "Để hỗ trợ con người xử lý sự biến động trong quy trình giám sát, đề xuất xây dựng một bộ kịch bản ứng phó mẫu chuẩn hóa (SOP). "
    "Khi tình huống bất ngờ phát sinh, nhân sự chỉ cần đối chiếu nhanh theo bộ quy tắc có sẵn để ra quyết định xử lý một cách chủ động."
)

st.markdown("**3. Tự động hóa ghi chép thông tin**")
st.markdown(
    "Tác vụ lưu trữ thông tin nên được tối ưu hóa bằng quy trình số hóa thông qua các biểu mẫu tự động điền dữ liệu. "
    "Khi con người hoàn thành việc giao tiếp hoặc kiểm tra, hệ thống sẽ tự động ghi nhận báo cáo mà không cần nhập liệu thủ công."
)

st.write("---")

# ==============================================================================
# KHỐI 5: BIỂU ĐỒ TẠ ĐÔI
# ==============================================================================
st.markdown("<div id='khoi5'></div>", unsafe_allow_html=True)
st.subheader("Phân Tích Điểm Mù (Gap Analysis) Giữa Kỹ Thuật & Con Người")
st.pyplot(draw_gap_analysis_dumbbell(df_it_final, selected_occupation))

st.markdown("### Nhận xét biểu đồ")
st.markdown(
    "Biểu đồ thể hiện khoảng cách giữa Khả năng của AI và Mong muốn của nhân sự trên thang điểm từ 1 đến 5. Điểm cốt lõi được rút ra như sau:\n\n"
    "**Lệch pha lớn nhất ở nhóm tích hợp kỹ năng:** Tác vụ hỗn hợp đầu tiên (*Thinking Creatively, Working with Computers...*) "
    "có khoảng cách xa nhất khi điểm kỹ thuật của AI vượt mức 4.0 nhưng mong muốn của nhân sự lại dưới 3.0. Điều này cho thấy "
    "công nghệ đã rất sẵn sàng để tự động hóa chuỗi việc này, nhưng nhân viên lại cực kỳ e ngại hoặc chưa sẵn sàng phối hợp.\n\n"
    "**Khoảng cách trung bình ở nhóm tác vụ văn phòng/vận hành:** Các tác vụ như Ghi chép thông tin (*Documenting...*), "
    "Kiểm tra thiết bị (*Inspecting...*) và Làm việc với máy tính (*Working with Computers*) đều có độ lệch pha rõ rệt từ 0.5 đến 0.7 điểm. "
    "Công nghệ AI đang đi trước một bước so với nhu cầu tiếp nhận thực tế của người lao động.\n\n"
    "**Đồng pha ở nhóm tác vụ tư duy và giám sát:** Các hoạt động như Giám sát quy trình (*Monitoring...*), "
    "Phân tích dữ liệu (*Analyzing Data...*) và Tư duy sáng tạo (*Thinking Creatively*) có hai chấm gần như chạm vào nhau "
    "(đều quanh mốc 3.0 đến 3.5). Đây là sự đồng thuận lý tưởng khi năng lực của AI đáp ứng vừa vặn với mong đợi và kỳ vọng hỗ trợ từ nhân sự IT."
)

st.markdown("### Đề xuất giải pháp")

st.markdown("**1. Thu hẹp khoảng cách ở vùng lệch pha lớn**")
st.markdown(
    "Đối với các tác vụ có độ lệch pha cực lớn ở đầu bảng, đề xuất tổ chức các chương trình đào tạo nhận thức và thay đổi tư duy cho nhân sự. "
    "Việc giúp họ hiểu rõ AI đóng vai trò hỗ trợ tăng năng suất chứ không thay thế vị trí làm việc sẽ kích thích mức độ mong muốn sử dụng tăng lên."
)

st.markdown("**2. Triển khai ở vùng đồng pha hoàn hảo**")
st.markdown(
    "Các tác vụ có sự đồng thuận cao như phân tích dữ liệu hay giám sát quy trình cần được ưu tiên đóng gói và đưa vào ứng dụng thực tế ngay lập tức. "
    "Đây là những vùng \"thắng nhanh\" vì cả công nghệ và tâm lý con người đã sẵn sàng phối hợp để tạo ra hiệu quả công việc mà không gặp rào cản."
)

st.markdown("**3. Chuẩn hóa tài liệu để giảm lệch pha tác vụ văn phòng**")
st.markdown(
    "Đối với nhóm tác vụ ghi chép và lưu trữ thông tin đang có độ lệch pha trung bình, đề xuất xây dựng các biểu mẫu và quy trình nhập liệu tự động hóa hoàn toàn. "
    "Việc tối giản hóa thao tác thủ công sẽ giúp nhân sự IT dễ dàng làm quen và nhanh chóng thu hẹp khoảng cách tiếp cận công nghệ."
)