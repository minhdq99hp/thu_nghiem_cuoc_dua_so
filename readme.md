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







