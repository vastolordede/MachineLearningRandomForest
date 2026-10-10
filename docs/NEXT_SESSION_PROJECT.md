# PROMPT BÀN GIAO CHO SESSION MỚI — PROJECT CUỐI KỲ MACHINE LEARNING

> Dán toàn bộ file này vào session mới, kèm **Google Sheet xuất PDF mới nhất**, `README.md`, `PATCH_MANIFEST.json`, `git log --oneline` và nếu cần **ZIP/source repo**. Không suy đoán code đã merge chỉ dựa vào mô tả này.

## 1. Mục tiêu và bối cảnh
Tôi học môn Machine Learning, nhóm bốn người thực hiện đề tài **Dự đoán giá xe cũ bằng Decision Tree Regression và Random Forest Regression**. Tên nghiên cứu dự kiến (chưa chính thức): **From One Tree to a Forest: A Comparative Study of Decision Tree and Random Forest Regression for Used-Car Price Prediction**. Bài toán supervised regression, target `price`, so sánh Dummy baseline/Decision Tree/Custom Random Forest bằng MAE, RMSE, R²; bổ sung OOB, cross-validation và feature importance khi khả thi.

Repo GitHub: https://github.com/vastolordede/MachineLearningRandomForest

Dataset đang lấy làm **schema khởi tạo tạm thời** là DS04 `imbalet/used_cars` trên Hugging Face, gần 432k dòng, khoảng 21 trường; link: https://huggingface.co/datasets/imbalet/used_cars . Các bộ ứng viên từng khảo sát: DS01 CarDekho https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho ; DS02 Kaggle S4E9 https://www.kaggle.com/competitions/playground-series-s4e9/data ; DS03 Craigslist EDA https://huggingface.co/datasets/Yoad22/craigslist-used-cars-eda ; DS04 như trên. **D10/D12 chưa được nghiệm thu**, currency và dtypes phải audit; không bịa sạch/đã chốt khi chưa có bằng chứng.

## 2. Người phụ trách (đúng Google Sheet)
- **Thảo — TV1**: raw dataset, numeric cleaning, missing/outlier/duplicates, data quality, numeric EDA.
- **Sang — TV2**: categorical cleaning, feature engineering, encoding, train-fitted preprocessing pipeline.
- **Đặng — TV3 (tôi)**: Model Engineering + Testing + ML Ops, tổ chức Git patches.
- **Danh — TV4**: chạy experiment, tuning, cross validation, đánh giá, biểu đồ/so sánh.

Cố gắng chỉ hoàn thành công việc của Đặng; không tự đánh dấu hoàn thành dữ liệu chưa do Thảo/Sang bàn giao. Khi code cần giả dữ liệu, dùng synthetic fixtures, nói rõ không phải real metrics.

## 3. Kiến trúc đã thống nhất
- GitHub quản lý **source, cấu hình, tests, notebook Kaggle**; không commit datasets/logs/weights/credentials.
- Kaggle Notebook (ưu tiên CPU miễn phí) làm môi trường chạy; đặt input bằng Kaggle Add Input và output dưới `/kaggle/working`, Save Version sau run.
- Persistent tracking/artifact store **W&B mới là ứng viên**, chưa xác nhận đã tích hợp. Trước mắt có bản output cục bộ/Save Kaggle Version và manifest run_id/config/dataset/git commit/metrics. Không tự báo đã upload W&B.
- Đầu vào từ TV1+TV2: cleaned **Parquet** (giữ schema/column names), `schema.json`, `cleaning_report.json` và code `preprocessing.py`. Không trao đổi NumPy array thiếu tên feature như contract chính.
- TV3 thích phương án “dùng gạch xây nhà”: **tự ghép nhiều `sklearn.tree.DecisionTreeRegressor` thành custom Random Forest**, tự triển khai bootstrap, per-tree config/random seeds, average prediction, OOB, interface/tests; không viết CART splitter ở mức thấp và không chỉ gọi một dòng `RandomForestRegressor` cho bài chính. Sklearn RF chỉ reference kiểm chứng.
- Data leakage: split train/dev/test trước mọi transformer có `fit`, fit imputer/encoder trên train rồi transform dev/test, test khóa đến bước đánh giá cuối; kiểm tra duplicate/group leakage.

## 4. Workflow ZIP Patch/Git BẮT BUỘC
Repo local Windows, VSCode; root có `work/apply_patch.py`, thư mục `work/` bị .gitignore. GPT tạo **ZIP Patch**: `PATCH_MANIFEST.json`, `README_PATCH.txt`, `payload/<relative root path>`. Tuyệt đối không giả vờ sửa repo hoặc push Git; tôi sẽ chạy lệnh tại máy.

Vòng thường: `git switch dev` → `git pull --ff-only origin dev` → kiểm tra `git status` sạch → `git switch -c feature/<cluster>` → đặt ZIP vào `work/` → `python work\apply_patch.py <ZIP> --dry-run` → kiểm tra danh sách CREATE/REPLACE → `python work\apply_patch.py <ZIP>` → tests → selective `git add <paths>` → `git diff --cached --check` và `git diff --cached` → `git commit` → `git log --oneline -1` → `git push -u origin <branch>` → PR vào `dev` → merge → ghi commit hash theo **ID đúng trong sheet**.

**Ngoại lệ được user cho phép vì ít thời gian:** Một ZIP lớn, nhiều cụm logic, **nhiều commits riêng theo từng cụm trên CÙNG một feature branch**, push branch một lần, một PR chứa nhiều commits. Không stage tất cả bằng `git add .`; không tick task chưa pass/merge. ZIP có backup và `--dry-run` trước apply.

## 5. Lịch sử và trạng thái đã biết
- Loop 01 scaffold gồm 17 file, ZIP `MLRF_01_Project_Scaffold_Patch.zip`, được user apply và commit `f4baf13` trên `feature/project-scaffold`, message `chore: scaffold ML project and Kaggle notebook` (ảnh màn hình); **chưa có xác nhận PR merge** trong chat.
- Mapping scaffold: `O01` notebook skeleton, `O02` centralized config, `O03` seed chung **một phần** vì chưa thực thi model/split. Không gán hash hoàn thành `D10`, `D12`, `D13`, `O08`.
- Patch gộp code cá nhân (được tạo ở session này): `MLRF_02_TV3_All_In_One_Patch.zip` với model custom forest, metrics, safeguards, experiment artifacts, Kaggle smoke notebook, tests, docs. Nếu chưa có log merge, coi là **đang chờ user apply/test/Git**, không khẳng định đã push.
- PDF source checklist mới nhất: **Sheet Checklist Việc Machine Learning - Đồ án cuối kì.pdf** gồm 4 trang. Trên PDF các ID `O01–O09`, `T01–T15`, `D01–D14`, `C01–C28`, `E11–E14`, `R01–R13` có hiện; PDF export **không thấy đầy đủ dòng M01–M15 và E01–E10**, nên yêu cầu kiểm tra Google Sheet thực tế trước khi tick/model ID mapping.

## 6. Các commit logic của Patch 02
1. **Model components**: `src/mlrf/models/`, `src/mlrf/training/`, `src/mlrf/evaluation.py`, `configs/model_search.json`. Liên quan M01-M15 (chỉ phần triển khai phù hợp, không tick experiment).
2. **Testing + Artifact Infrastructure**: `src/mlrf/testing/`, `src/mlrf/ops/`, `tests/test_models.py`, `tests/test_ops_and_splits.py`. Liên quan T01-T05/T07/T08 mức synthetic, O04/O05 một phần, O08 metadata support.
3. **Kaggle Notebook + Documentation**: `notebooks/01_training_smoke.ipynb`, `docs/MODEL_WORKFLOW.md`, `docs/REPO_FILE_GUIDE.md`, 2 handoff docs. O07 chưa hoàn toàn đến khi Kaggle Run All thật; T09 chưa nghiệm thu thực tế.

Luôn map hash cho từng commit theo **dòng đã thực sự đáp ứng tiêu chí**, có thể ghi hash một phần kèm trạng thái Đang Làm. Không đổi owner của Thảo/Sang/Danh.

## 7. Chỗ cần giúp khi quay lại
- Chỉ dẫn lệnh Windows từng bước tương ứng trạng thái Git thật (đừng giả sử PR trước đã merge nếu chưa có chứng cứ).
- Debug unittest/import/schema mismatch của patch và lỗi notebook Kaggle thật.
- Tích hợp Data Contract sau khi TV1/TV2 bàn giao Parquet; đối chiếu `price`/dtype/metadata/duplicate/tuổi xe; không tự suy luận currency.
- Thử Kaggle Add Input source package + Parquet + Save Run, đo RAM/runtime; tránh chạy 432k rows × 100 trees ngay.
- Nếu triển khai W&B, làm patch riêng, secrets qua Kaggle Secrets, test quota và quyền riêng tư; không commit API key.
- Hỗ trợ giải thích từng file và đọc hiểu thuật toán, không chỉ viết code.

## 8. Mở session nên gửi kèm
1. File này.
2. File PDF xuất lại từ Google Sheet mới nhất, có rõ ID/Commit Github.
3. Log `git status -sb`, `git log --oneline -8`, URL PR đã merge.
4. `src/`, `configs/`, `tests/`, `notebooks/` zip từ repo nếu muốn ChatGPT review code chính xác.
5. Kết quả chạy `python -m unittest discover -s tests -v` và notebook Kaggle nếu có lỗi.

**Đầu bài cho trợ lý:** "Hãy kiểm tra trạng thái repo và commit/hash theo sheet, tiếp tục từ phần chưa hoàn thành. Không giả định code đã chạy trên dữ liệu thật. Luôn chuẩn bị ZIP patch và hướng dẫn selective commits, push/PR; ưu tiên unit-test trước rồi Kaggle end-to-end." 
