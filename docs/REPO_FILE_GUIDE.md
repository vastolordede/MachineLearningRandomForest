# ĐỌC HIỂU CODE PROJECT — MachineLearningRandomForest (bản cập nhật sau gói cá nhân)

## 1. Bài toán
Input là thuộc tính xe cũ; target `price` là giá dạng số liên tục. Đây là **supervised regression**, không phải image classification/CNN. Hai mô hình chính: CART Decision Tree, custom Random Forest dùng nhiều CART của sklearn. Không có ReLU, Softmax, optimizer epoch, backprop hay CNN weights. `criterion` quyết định cách đo chất lượng phép chia nút; `MAE/RMSE/R²` đánh giá dự đoán. Đây là hai vai trò khác nhau.

## 2. Tổng quan luồng chạy
`Parquet do TV1/TV2 bàn giao` → `kiểm tra schema` → `split train/dev/test (chặn leakage)` → `fit preprocessing trên train` → `transform dev/test bằng cùng preprocessing` → `train Dummy/Tree/Forest trên train` → `TV4 chạy/so sánh trên dev` → `freeze best configuration` → `đánh giá test một lần` → `ghi artifacts, manifest, report`.

**Ranh giới trách nhiệm:** Thảo quản lý thu thập, numeric, duplicate, outlier; Sang categorical, feature engineering, fitted preprocessing; Đặng model + code guards + tests + ML Ops; Danh experiments, hyperparameter tuning, evaluation, plots và kết luận. Repo scaffold không làm thay D01–D12/C01–C28.

## 3. Đọc theo thứ tự (tránh bị ngợp)
1. `configs/project.json`: dataset/target/split/seed/provisional trạng thái. Seed 42 và split 70/15/15 là cấu hình thử, chưa nghiệm thu D13.
2. `configs/ds04_schema.json`: danh sách cột **raw** dự kiến cho DS04 (21 cột); dtype thực tế và currency còn cần xác minh.
3. `src/mlrf/config.py`: đọc central config và validate tỷ lệ chia dữ liệu; `repository_root()` tìm root từ đường dẫn package.
4. `src/mlrf/data/contract.py` và `io.py`: xác minh tên cột raw, đọc cleaned Parquet. Không tự clean hay fit transformer.
5. `src/mlrf/models/decision_tree.py`: builder trả `sklearn.tree.DecisionTreeRegressor` với criterion/max_depth/min_samples*. Đây là **viên gạch CART**.
6. `src/mlrf/models/forest.py`: lớp `CustomRandomForestRegressor`; xem phương thức `__init__`, `_sample_size`, `fit`, `predict`.
7. `src/mlrf/training/interface.py`: nhận tên model/config, tạo estimator, fit và predict. `sklearn_reference_rf` dùng để đối chiếu, không thay custom forest.
8. `src/mlrf/testing/guards.py`: split reproducible, optional group-aware, check không trùng row IDs; không thay audit leakage về mặt ngữ nghĩa của feature.
9. `src/mlrf/evaluation.py`: các hàm tính MAE, RMSE, R²; TV4 chịu trách nhiệm phân tích metrics thực tế.
10. `src/mlrf/ops/artifacts.py`: tạo run folder, ghi `config.json`, `metrics.json`, `predictions.csv`, `manifest.json`; hỗ trợ save/load pipeline.
11. `notebooks/00_environment_check.ipynb`: notebook môi trường của scaffold cũ.
12. `notebooks/01_training_smoke.ipynb`: notebook mới, dùng synthetic data mặc định; chạy DS04 thật chỉ sau khi dữ liệu được nghiệm thu.
13. `tests/test_bootstrap.py`, `tests/test_models.py`, `tests/test_ops_and_splits.py`: bộ unit và integration tests trên dữ liệu tổng hợp.

## 4. Đọc `CustomRandomForestRegressor` từ trên xuống
- `n_estimators`: bao nhiêu cây.
- `max_depth`, `min_samples_split`, `min_samples_leaf`, `criterion`: cách xây từng cây.
- `max_features`: tập feature được xem tại **mỗi split** của CART; ví dụ 0.7 => khoảng 70% feature (sklearn quyết định làm tròn).
- `bootstrap=True`: mỗi cây chọn các dòng train có hoàn lại. Một số dòng không được chọn là OOB của cây đó.
- `random_state`: generator tạo bootstrap và random seed từng cây, giúp tái lập nếu dữ liệu, phiên bản thư viện và môi trường giữ nguyên.
- `fit`: lấy `X, y`, validate, lặp qua từng cây: sample indices → khởi tạo CART → fit → lưu tree; nếu bật OOB, cộng prediction cho các dòng chưa thấy.
- `feature_importances_`: trung bình impurity-based importances của các cây; cẩn thận thiên lệch cho feature nhiều giá trị.
- `predict`: gọi predict từng tree và trung bình dạng running mean để tránh mảng dự đoán n_trees × n_rows.
- `oob_prediction_`: dự đoán cho hàng có ít nhất một cây OOB; các hàng không được dự đoán là NaN. `oob_score_` dùng R² các hàng có OOB prediction, không phải 36.8% error rate.

## 5. Chỗ dễ hiểu nhầm
- **36.8%** là tỷ lệ *sample không được lấy* trong bootstrap cỡ n với n lượt lấy, không phải tỷ lệ cây học sai.
- Custom forest không có `activation function` hay gradient. `criterion='squared_error'` là cách đánh giá split, không phải chức năng loss của mạng CNN.
- Random Forest có thể vẫn overfit và không mặc định luôn tốt hơn một cây; cần kiểm chứng bằng kết quả dev/test.
- OOB chỉ hữu ích khi bootstrap=True và đủ cây. OOB không thay thế mọi mục đích của held-out Test.
- Bản custom hiện chạy **tuần tự (single process)**, không hỗ trợ mọi tối ưu của sklearn native. Trên ~432k hàng, bắt đầu sample nhỏ rồi đo chi phí thật, không chạy full một lần khi chưa kiểm tra runtime.
- Kiểm thử tự động trên **synthetic fixtures**, không chứng minh DS04 sạch hoặc project đạt metric nào.
- Thứ tự split/fit: các bước tính statistics từ dữ liệu phải học từ Train; test không tham gia tuning.

## 6. Cách đọc code thực hành (30-60 phút)
- Chạy `python -m unittest discover -s tests -v` rồi đọc `tests/test_models.py`.
- Chạy `build_model('decision_tree',...)`, quan sát `.tree_.node_count` và `.get_depth()`.
- Chạy custom forest `n_estimators=3`, xem `.bootstrap_indices_`, `.estimators_`, các predict riêng và trung bình cuối.
- Bật `oob_score=True`, xem `.oob_counts_`, `.oob_prediction_`, `.oob_score_`.
- Chạy `tests/test_ops_and_splits.py` để hiểu vì sao pipeline chỉ `fit` preprocessing trên train.
- Mở notebook 01, bắt đầu ở chế độ synthetic; xem các file `manifest.json` được ghi.

## 7. Nhánh/Git/ZIP
Workflow gốc: `dev` sạch → `feature/<cluster>` → ZIP `payload/` dry-run → apply → test → selective staging → commit theo cụm → `git log --oneline -1` → push → PR vào `dev`. Một ZIP lớn có thể chứa nhiều cụm riêng nhưng phải stage đúng tệp từng commit. `work/` bị ignore và không commit; không commit dataset Parquet/log/joblib/API token. Đừng tick các ID phụ thuộc dữ liệu thật chỉ vì unit tests mock đạt.

## 8. Trạng thái/công việc thiếu
- Chưa có clean DS04 thực tế do Thảo/Sang bàn giao; schema raw là tạm, currency chưa xác minh; model không có kết quả dự đoán thực tế.
- Cần chốt D10/D12/D13/D14, rồi kiểm chứng TV2 preprocessing trên dữ liệu thật.
- Chưa có W&B integration đã xác minh hoặc artifact store ngoài Kaggle; output hiện là local/Kaggle-working, cần Save Version và external sync.
- TV4 chưa chạy tuning/so sánh cuối; chưa có final best model, paper, biểu đồ giá xe thực tế.
- Commit hash cho patch lớn sẽ do người dùng tạo trên cùng branch và ghi vào các dòng checklist thực sự nghiệm thu sau PR merge.
