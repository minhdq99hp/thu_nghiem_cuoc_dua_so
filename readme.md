# Thử nghiệm Cuộc đua số

## Công việc: 
Thử nghiệm một số thuật toán cho Cuộc đua số

**Hướng đi hiện tại**: Dùng xử lý ảnh và machine learning để có thể nhận diện vật thể và đưa ra quyết định.

**Keyword**: Python, Computer Vision, Machine Learning, Depth Camera

## Phần cứng
**Orbbec Astra**: sử dụng 2 kênh Color và Depth có thông số 640x320, 30FPS

**Jetson TX2**: Hệ thống xử lý của con xe.

## Cấu trúc project:

### cameramanager.py
Cài đặt OpenCV và OpenNI2 để dùng camera Astra [hướng dẫn tại đây](https://astra-wiki.readthedocs.io/)

#### Lấy depth_frame và color_frame
Dùng numpy để lấy dữ liệu từ camera. Tuy nhiên do sử dụng registration để khớp ảnh depth với ảnh color nên sẽ sinh ra cropped_frame để tiện xử lý. 

Quá trình về sau chủ yếu xử lý trên cropped_depth_frame (dtype=np.uint16) và cropped_color_frame(dtype=np.uint8) với size 550x420.

#### Chọn Region of Interest
Có 3 regions:
- left_roi: vùng có khả năng xuất hiện biển báo ở bên trái
- right_roi: vùng có khả năng xuất hiện biển báo ở bên phải
- main_roi: vùng chứa đường và làn đường, có khả năng xuất hiện các vật thể cản trở.

![]('/data/demo_1.png')

Xem demo ở cuối file

### objectsegmentation.py

#### Lọc nền đường (dùng cho main_roi)
Nhận thấy nếu camera cố định trên xe thì khoảng cách camera đo được đến nền đường là cố định.

-> Cần xác định ma trận nền đường đó. 

##### Xác định depth_ground
Đầu tiên, ta sẽ xây dựng ground_data (hiện tại đã có ground_data.txt trong folder data/).

Xét các điểm tạo thành đường thằng ở giữa ảnh (từ cuối frame đến giữa frame). Nhận thấy các điểm này đi qua một hàm (x+a)/(x+b). Ta xấp xỉ hàm đó bằng hàm `scipy.optimize.curve_fit()` và được tham số a=1.94926806e+06, b=-1.90139644e+02.

-> Có được độ depth xấp xỉ tại center line.

Tiếp theo sẽ tính toán ma trận depth_ground dựa vào kết quả trên. Xem chi tiết tại hàm `compute_depth_ground()` và `get_depth_of_center_row()` 

##### Lọc frame dựa trên depth_ground
Numpy hỗ trợ tính toán ma trận rất tốt. 

Ta lọc các điểm có khả năng là nền dựa trên khoảng cách của điểm đó với depth_ground, tùy chỉnh độ nhạy bằng cách thay đổi biến `ground_offset`. 

Đọc chi tiết trong hàm `remove_ground_2()` 

#### Lọc nền ở left_roi và right_roi
Phần này khá đơn giản. Lọc tất cả các điểm lớn hơn một giá trị threshold là được. 


Toàn bộ quá trình xem ở hàm `test_ground_removal()` cuối file 





