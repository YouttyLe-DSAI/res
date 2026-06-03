# So sánh repo hiện tại với MandyMo/ECCV_Countformer

Tài liệu này so sánh repo hiện tại `khangkaka066/CountFormer_CityStreet` với repo gốc [`MandyMo/ECCV_Countformer`](https://github.com/MandyMo/ECCV_Countformer).

- Upstream được so sánh: `MandyMo/ECCV_Countformer@27e5abd0ebc3bfa0431b4bb4e3f5f14389964317`
- Repo hiện tại: `khangkaka066/CountFormer_CityStreet@6ba1ad780d4b4e8cdcb7570a71cd06470aae6e0a`
- Lệnh kiểm tra chính:

```bash
git diff --stat 27e5abd0ebc3bfa0431b4bb4e3f5f14389964317..HEAD
git diff --name-status 27e5abd0ebc3bfa0431b4bb4e3f5f14389964317..HEAD
```

## Tóm tắt nhanh

Repo hiện tại vẫn giữ ý tưởng chính của CountFormer gốc:

- Multi-view image input.
- Image backbone + feature pyramid.
- Cross-view deformable attention để đưa đặc trưng ảnh vào không gian voxel/BEV.
- Voxel density prediction để đếm người trong scene.
- Image density head phụ trợ.

Nhưng repo đã được chỉnh mạnh để chạy được trên Kaggle với GPU đơn và tài nguyên hạn chế. Bản Kaggle hiện tại là một phiên bản nhỏ hơn, không phải cấu hình paper gốc.

Thay đổi lớn nhất:

| Hạng mục | Repo gốc | Repo hiện tại |
|---|---|---|
| Dataset chính trong script train | Cross-view config `cscv_voxel.py` | CityStreet config |
| Config Kaggle | Không có | Có `citystreet_voxel_kaggle_tiny.py` |
| Backbone bản Kaggle | Swin-T trong config gốc/paper-style | ResNet50 |
| Scale ảnh bản Kaggle | Lớn hơn, gần full config | `0.5` |
| Voxel bản Kaggle | `256x256x16` ở config full | `64x64x8` |
| Transformer layers | `[3, 5, 7, 9]` ở config full | `[1, 1, 2, 2]` |
| Feature dims | `[64, 128, 192, 256]` ở config full | `[32, 64, 128, 128]` |
| Custom CUDA MSDA | Bắt buộc compile extension | Có fallback PyTorch bằng `USE_MSDA_PYTORCH=1` |
| Kaggle scripts | Không có | Có train/test/visualize scripts |
| TensorBoard | Luôn import/use | Có thể tắt bằng `DISABLE_TENSORBOARD=1` |
| Eval trong train | Theo checkpoint period gốc | In kết quả sau mỗi `KAGGLE_EVAL_PERIOD` epoch |

## File thay đổi

So với upstream, repo hiện tại thay đổi 13 file:

```text
M  .gitignore
A  projects/configs/citystreet/citystreet_voxel.py
A  projects/configs/citystreet/citystreet_voxel_kaggle_tiny.py
M  projects/dataset/BaseDataset.py
M  projects/modules/attentions/ms_deform_atten/ms_deform_atten.py
M  projects/modules/counters/vox_counter/voxcounter.py
A  requirements_kaggle.txt
A  tools/do_test_kaggle.sh
M  tools/do_train.sh
A  tools/do_train_kaggle.sh
A  tools/do_visualize_kaggle_3d.sh
M  tools/train.py
A  tools/visualize_kaggle_3d.py
```

Tổng diff:

```text
13 files changed, 1158 insertions(+), 36 deletions(-)
```

## 1. Thêm config CityStreet

### `projects/configs/citystreet/citystreet_voxel.py`

Đây là config CityStreet đầy đủ hơn, gần với cấu hình lớn/paper-style hơn bản Kaggle tiny.

Mục đích:

- Tạo config riêng cho dataset CityStreet.
- Dùng refined BEV labels.
- Đặt `root='CityStreet'`.
- Dùng `exp_name='citystreet/vox/...'`.
- Dùng voxel resolution lớn hơn bản Kaggle tiny.

Config này phù hợp hơn nếu train trên GPU mạnh hơn, ví dụ A100/V100 nhiều VRAM. Với Kaggle GPU đơn, config này nặng.

### `projects/configs/citystreet/citystreet_voxel_kaggle_tiny.py`

Đây là config chính đang dùng trên Kaggle.

Mục tiêu của file này là giảm chi phí GPU nhưng vẫn giữ pipeline CountFormer:

- Vẫn dùng multi-view CityStreet 3 camera.
- Vẫn sinh image density map và voxel density map.
- Vẫn dùng cross-view attention.
- Vẫn predict density trong không gian voxel.
- Vẫn evaluate bằng `mae_bev`, `nae_bev`, `mae_image`, `nae_image`.

Các thay đổi giảm tải chính:

| Thông số | Giá trị trong Kaggle tiny | Ý nghĩa |
|---|---:|---|
| `scale` | `0.5` | Giảm kích thước ảnh đầu vào còn một nửa |
| `batch_size` | `1` | Phù hợp GPU đơn |
| `num_workers` | `2` train/val, `1` test | Nhẹ hơn cho Kaggle |
| backbone | `resnet50` | Mạnh hơn ResNet18, vẫn thực dụng hơn Swin-T trên Kaggle |
| `image_bkb_channels` | `[256, 512, 1024, 2048]` | Channel đúng cho 4 stage ResNet50 Bottleneck |
| `_dim_` | `[32, 64, 128, 128]` | Giảm channel trong voxel transformer |
| `vox_h/vox_w/vox_z` | `[64,32,16,8]`, `[64,32,16,8]`, `[8,4,2,1]` | Giảm số voxel cần xử lý |
| `_num_layers_` | `[1,1,2,2]` | Ít transformer layer hơn |
| optimizer | AdamW `lr=1e-5` | Learning rate nhỏ, an toàn hơn khi fine-tune |
| epochs | `KAGGLE_EPOCHS`, default `80` | Điều khiển bằng biến môi trường |
| eval period | `KAGGLE_EVAL_PERIOD`, default `5` | In kết quả mỗi 5 epoch |

Điểm quan trọng: checkpoint train bằng config tiny chỉ tương thích với config tiny. Không thể load trực tiếp checkpoint này vào config gốc hoặc config full nếu kiến trúc khác nhau.

## 2. Script train/test/visualize cho Kaggle

### `tools/do_train_kaggle.sh`

Script train mới cho Kaggle.

Nó thêm các biến môi trường quan trọng:

```bash
PRETRAINED=${PRETRAINED:-1}
KAGGLE_EPOCHS=${KAGGLE_EPOCHS:-80}
KAGGLE_EVAL_PERIOD=${KAGGLE_EVAL_PERIOD:-5}
USE_MSDA_PYTORCH=${USE_MSDA_PYTORCH:-1}
DISABLE_TENSORBOARD=${DISABLE_TENSORBOARD:-1}
```

Ý nghĩa:

- `PRETRAINED=1`: dùng pretrained ResNet50 nếu môi trường tải được weights.
- `KAGGLE_EPOCHS`: số epoch train.
- `KAGGLE_EVAL_PERIOD`: chu kỳ evaluate và save checkpoint.
- `USE_MSDA_PYTORCH=1`: không compile custom CUDA extension, dùng fallback PyTorch.
- `DISABLE_TENSORBOARD=1`: tránh lỗi/log nặng từ TensorBoard trên Kaggle.

Ví dụ train:

```bash
CITYSTREET_ROOT=/kaggle/input/datasets/nguyenvohoangkhang/citystreet/CityStreet \
PRETRAINED=0 \
KAGGLE_EPOCHS=80 \
KAGGLE_EVAL_PERIOD=5 \
USE_MSDA_PYTORCH=1 \
bash tools/do_train_kaggle.sh
```

### `tools/do_test_kaggle.sh`

Script test/evaluate mới cho Kaggle.

Mặc định:

- Dùng config tiny.
- Tự tìm run mới nhất trong `workdirs/citystreet/vox/kaggle_tiny`.
- Ưu tiên checkpoint `best_mae_bev.pth`.

Ví dụ:

```bash
CITYSTREET_ROOT=/kaggle/input/datasets/nguyenvohoangkhang/citystreet/CityStreet \
PRETRAINED=0 \
USE_MSDA_PYTORCH=1 \
bash tools/do_test_kaggle.sh
```

### `tools/visualize_kaggle_3d.py` và `tools/do_visualize_kaggle_3d.sh`

Đây là phần repo gốc không có.

Mục đích:

- Load checkpoint đã train.
- Lấy một sample CityStreet.
- Chạy model.
- Xuất voxel density prediction thành file HTML 3D tương tác bằng Plotly.

Output mặc định:

```text
workdirs/kaggle_3d_visualization.html
```

Script đã được chỉnh để:

- Dùng đúng `collate_fn` từ dataloader.
- Tự tìm checkpoint `best_mae_bev.pth`.
- Nhúng Plotly vào HTML bằng `include_plotlyjs=True`, giúp file mở được offline.

## 3. Sửa attention để chạy được trên Kaggle

File:

```text
projects/modules/attentions/ms_deform_atten/ms_deform_atten.py
```

Repo gốc luôn compile custom CUDA extension:

```python
torch.utils.cpp_extension.load(...)
```

Điều này dễ lỗi trên Kaggle vì:

- CUDA/PyTorch version không khớp.
- GPU architecture không khớp.
- Không đủ quyền hoặc môi trường compile không ổn định.

Repo hiện tại thêm:

```python
USE_MSDA_PYTORCH = os.environ.get('USE_MSDA_PYTORCH', '0') == '1'
```

Khi chạy:

```bash
USE_MSDA_PYTORCH=1
```

model dùng implementation PyTorch fallback thay vì custom CUDA extension.

Đánh đổi:

- Dễ chạy hơn trên Kaggle.
- Ít lỗi compile hơn.
- Có thể chậm hơn custom CUDA.
- Kết quả training vẫn đúng về mặt pipeline, nhưng tốc độ/hiệu năng có thể khác repo gốc.

File này cũng đổi API AMP cũ:

```python
torch.cuda.amp.custom_fwd/custom_bwd
```

sang:

```python
torch.amp.custom_fwd/custom_bwd(device_type='cuda')
```

để giảm warning trên PyTorch mới.

## 4. Sửa training loop

File:

```text
tools/train.py
```

Các thay đổi chính:

### Tắt TensorBoard khi cần

Thêm `NullSummaryWriter` khi:

```bash
DISABLE_TENSORBOARD=1
```

Điều này giúp Kaggle không import TensorFlow/TensorBoard quá nhiều và giảm log nhiễu.

### Dùng AMP API mới

Đổi:

```python
torch.cuda.amp.GradScaler
torch.cuda.amp.autocast
```

sang:

```python
torch.amp.GradScaler('cuda')
torch.amp.autocast('cuda')
```

Mục đích là giảm `FutureWarning` trên PyTorch mới.

### Detach tensor khi log scalar

Repo gốc convert loss tensor trực tiếp sang float, gây warning:

```text
Converting a tensor with requires_grad=True to a scalar may lead to unexpected behavior
```

Repo hiện tại dùng `.detach()` trước khi log.

### In kết quả evaluate sau mỗi N epoch

Repo hiện tại thêm:

```python
print(f'[eval][epoch {current_epoch}] {eval_msg}', flush=True)
```

Kết quả sẽ hiện trong log Kaggle sau mỗi `KAGGLE_EVAL_PERIOD` epoch, ví dụ:

```text
[eval][epoch 5] {'nae_bev': ..., 'mae_bev': ..., 'nae_image': ..., 'mae_image': ...}
```

## 5. Sửa dataset để không crash khi label ảnh rỗng

File:

```text
projects/dataset/BaseDataset.py
```

Repo gốc gọi:

```python
cv2.undistortPoints(points, ...)
```

ngay cả khi `points` rỗng.

Một số frame/camera trong CityStreet có thể không có điểm annotation ảnh. Khi `points` rỗng, OpenCV có thể crash.

Repo hiện tại thêm check:

```python
if points.size == 0:
    undistorted_points = points.reshape(0, 2).astype(np.float32)
else:
    undistorted_points = cv2.undistortPoints(...)
```

Kết quả: validation/test ổn hơn trên CityStreet.

## 6. Sửa `VOXCounter` AMP

File:

```text
projects/modules/counters/vox_counter/voxcounter.py
```

Đổi:

```python
torch.cuda.amp.autocast
```

sang:

```python
torch.amp.autocast('cuda', enabled=... and torch.cuda.is_available())
```

Mục đích:

- Giảm warning.
- Tránh bật CUDA AMP khi không có CUDA.

## 7. Thêm dependency Kaggle

File:

```text
requirements_kaggle.txt
```

Repo gốc không có file dependency riêng cho Kaggle.

File mới gom các package cần thiết:

```text
opencv-python-headless
h5py
easydict
scipy
einops
addict
yapf
tensorboard
tqdm
ninja
mmcv-lite
```

Trên Kaggle có thể chạy:

```bash
pip install -q -r requirements_kaggle.txt
```

## 8. `.gitignore`

Repo hiện tại ignore thêm các file/folder không nên đưa lên GitHub:

```text
data/
workdirs/*
.DS_Store
.venv/
__pycache__/
*.py[cod]
2407.02047v1.pdf
.history/
```

Điều này tránh commit dataset, checkpoint, cache Python, PDF paper, và file tạm.

## 9. Khác biệt quan trọng về kết quả paper

Bản Kaggle tiny không nên được so trực tiếp như một reproduction đầy đủ của paper.

Lý do:

- Backbone đổi từ Swin-T/paper-style sang ResNet50.
- Ảnh bị giảm scale còn `0.5`.
- Voxel resolution giảm rất mạnh.
- Số layer attention giảm.
- Feature dims giảm.
- Số epoch thường thấp hơn config gốc.
- Dùng PyTorch fallback cho MSDA thay vì custom CUDA.

Vì vậy, kết quả từ `citystreet_voxel_kaggle_tiny.py` nên được hiểu là:

> Phiên bản thử nghiệm nhỏ để chạy được trên Kaggle, giữ ý tưởng cốt lõi của CountFormer, nhưng đánh đổi độ chính xác.

Nếu muốn so với paper công bằng hơn, cần train config lớn hơn:

```text
projects/configs/citystreet/citystreet_voxel.py
```

hoặc cấu hình gần repo gốc/paper hơn, trên GPU mạnh hơn và nhiều epoch hơn.

## 10. Checkpoint và tái sử dụng

Checkpoint Kaggle được lưu ở:

```text
workdirs/citystreet/vox/kaggle_tiny/[timestamp]/checkpoints/
```

Các file chính:

```text
best_mae_bev.pth
best_nae_bev.pth
best_mae_image.pth
best_nae_image.pth
```

Nên dùng:

```text
best_mae_bev.pth
```

vì metric paper chủ yếu so bằng BEV/scene-level MAE và NAE.

Lưu ý: checkpoint tiny phải dùng lại với config tiny:

```text
projects/configs/citystreet/citystreet_voxel_kaggle_tiny.py
```

Không load checkpoint tiny vào config full/gốc vì shape model khác.

## Kết luận

Repo hiện tại là một bản chuyển hướng thực dụng từ CountFormer gốc sang CityStreet + Kaggle:

- Có thể train/test/visualize trên Kaggle GPU đơn.
- Có fallback tránh lỗi custom CUDA extension.
- Có config nhỏ hơn để chạy trong giới hạn VRAM.
- Có logging/evaluation tiện hơn cho notebook.

Đổi lại, đây không còn là cấu hình paper gốc. Nó giữ ý tưởng cốt lõi của CountFormer, nhưng đã giảm mô hình và độ phân giải đáng kể, nên kết quả thấp hơn paper là bình thường.
