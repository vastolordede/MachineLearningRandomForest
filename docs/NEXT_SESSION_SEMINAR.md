# PROMPT BÀN GIAO CHO SESSION MỚI — SEMINAR CHAPTER 10

> Dán tài liệu này kèm PDF **Chapter 10: Infrastructure and Tooling for MLOps** và sheet tab **Seminar**. Tránh coi bản nháp là slide đã hoàn thành.

## 1. Môn học, đề tài, yêu cầu
- Môn Machine Learning, nhóm 4 người; seminar **Chapter 10 — Infrastructure and Tooling for MLOps** trong sách *Designing Machine Learning Systems* của Chip Huyen (O'Reilly).
- Theo ảnh phân công giảng viên: seminar từ tuần 8 theo lịch trình, thời gian trình bày khoảng **20–25 phút**, cần **PPTX**, references và chia sẻ nguồn/video hữu ích; tài liệu nội dung có thể dùng Google Docs.
- Sheet seminar theo giai đoạn: chuẩn bị đọc/chia phạm vi, nghiên cứu và soạn nội dung, slide, review/rehearsal, trình bày. Checklist IDs `S01–S31`.
- Project cuối kỳ là **một tab khác**, không được nhầm ID với seminar.

## 2. Chia phần Chapter 10
- TV1 **Thảo**: Introduction, Storage & Compute, Public Cloud vs Private Data Centers. Research `S03`, content `S08–S09`.
- TV2 **Sang**: Development Environment, standardization, Containers, Dev→Prod. Research `S04`, content `S10–S11`.
- TV3 **Đặng (tôi)**: Resource Management, Cron/Scheduler/Orchestrator, Data Science Workflow Management, DAG, tools. Research `S05`, content `S12–S13`, sơ đồ `S20`, video `S22`.
- TV4 **Danh**: ML Platform, Model Deployment, Model Store, Feature Store, Build vs Buy, Conclusion. Research `S06`, content `S14–S16`.
- `S07` nghiên cứu ví dụ tools là việc chung. `S17–S19` ghép/thống nhất slide; `S21` references; `S23` mini-demo tùy chọn; `S24–S31` review, rehearsal, Q&A, trình bày.

## 3. Nội dung Đặng đã có ở session trước
- Bản nghiên cứu `S05` khá chi tiết về Resource Management (CPU/GPU/RAM/I/O, cloud resource/cost), tính lặp lại và dependencies trong ML workflows, Cron vs Scheduler vs Orchestrator, DAG/node/edge, Airflow, Prefect, Argo, Kubeflow Pipelines, Metaflow.
- `S12`: đề xuất 2 slide: (1) Why Resource Management Matters, (2) Cron vs Scheduler vs Orchestrator.
- `S13`: đề xuất 2 slide: (3) ML Workflow & DAG, (4) Workflow Tools, nhấn Airflow. Tổng phần Đặng **4 slide ~5–6 phút**.
- `S20`: đã phác thảo 2 sơ đồ: MLOps Infrastructure Stack (bốn lớp Storage & Compute / Resource Management / ML Platform / Development Environment) và workflow DAG từ Load → Preprocess → Train A + Train B → Evaluate/Compare → Deploy; có sơ đồ phụ Scheduler → Execution/Infrastructure → Workflow Tasks. Chưa dựng hình final.
- `S22`: 3 video ứng viên: **Apache Airflow Explained Quick and Simple** (TechTalkSourav, 2023), **What is a DAG in Apache Airflow?** (TechTrapture, 2025), **Kubernetes Explained in 100 Seconds** (Fireship, 2020). **Chưa xem/chốt timestamp; chưa xác nhận dùng trong slide**. Nên tìm lại link chính xác nếu thiếu lịch sử chat, không bịa timestamp.
- Bảng nên giữ trong slide: Cron = trigger theo lịch, Scheduler = quyết định task đủ điều kiện chạy/điều phối job, Orchestrator = quản lý hạ tầng/workload. Trên thực tế có chồng lấn chức năng, không cứng nhắc.
- Có lưu ý **sách 2022** nên vài nhận xét về giới hạn Airflow cũ hơn tài liệu Airflow hiện tại; phân biệt nguồn gốc với cập nhật công cụ, không lặp lại như sự thật 2026.

## 4. Lộ trình nội dung cả chương
- Storage & Compute: lớp nền tài nguyên, FLOPS, memory, utilization; Public vs Private Data Center, cloud repatriation và multicloud.
- Development Environment: IDE, notebooks, versioning, reproducibility, standardizing dependencies; Dockerfile vs Docker image vs container; Kubernetes ở mức khái quát.
- Resource Management: repetitiveness/dependency; Cron, Scheduler, Orchestrator, DAG, Workflow Management và các công cụ.
- ML Platform: Model Deployment (online/batch), Model Store (quản lý artifacts/model/version), Feature Store (feature management/computation/consistency), Build vs Buy, kết luận.
- Khung slide đề xuất cả nhóm ~20 slide: Thảo 5, Sang 4, Đặng 4, Danh 5, summary/references 2. Chỉ là gợi ý, không cứng.

## 5. Tình trạng thực tế chưa hoàn tất
Đã có bản nháp nghiên cứu/ý tưởng từ ChatGPT, **không có bằng chứng** slide đã dựng, nội dung đã được giảng viên duyệt, clip đã xem, references đã kiểm tra, rehearsal đã chạy hoặc task đã tick. Nhiệm vụ còn: lọc ý, hoàn thiện nội dung ngắn gọn mỗi slide, thiết kế sơ đồ, chọn video có timestamp hoặc bỏ, kiểm tra nguồn, ghép deck, review chéo, tập 20–25 phút, Q&A và xuất PPTX/PDF dự phòng.

## 6. Nguồn gốc cần giữ
- Nguồn chính: Chip Huyen, *Designing Machine Learning Systems*, Chapter 10. Khi tóm tắt từ PDF phải dựa vào file nguồn được user cung cấp; đừng suy đoán số trang nếu edition khác.
- Tham khảo chính thức: https://airflow.apache.org/docs/ , https://kubernetes.io/docs/ , https://argo-workflows.readthedocs.io/ , https://www.kubeflow.org/docs/components/pipelines/ , https://docs.prefect.io/ , https://docs.metaflow.org/ .
- Có thể tham khảo book club Chapter 10: https://amazingelearning.com/book-club-designing-machine-learning-systems-chapter-10/ ; chỉ làm nguồn bổ sung, kiểm tra lại link và nội dung.

## 7. Yêu cầu cho trợ lý session mới
"Tôi đã có bản nghiên cứu về TV3 (Đặng) và checklist S05/S12/S13/S20/S22. Hãy giúp tôi lọc thành 4 slide đúng thời lượng, giữ nội dung không đứt mạch, viết speaker notes và gợi ý hình sơ đồ/video đáng tin cậy. Khi tài liệu gốc được upload, hãy đối chiếu trích dẫn trang chính xác. Không tự tick S17–S31 hay nói seminar đã hoàn thành. Khi làm PPTX hãy chờ tôi xác nhận nội dung cuối cùng." 
