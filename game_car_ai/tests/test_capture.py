import cv2
import os

# Tạo thư mục lưu ảnh nếu chưa có
output_dir = "game_car_ai/assets/samples"
os.makedirs(output_dir, exist_ok=True)

# Mở camera ảo (video10)
cap = cv2.VideoCapture(10)  # vì là /dev/video10

if not cap.isOpened():
    raise RuntimeError("Không thể mở camera ảo /dev/video10. Hãy chắc chắn scrcpy đang chạy.")

count = 0
while count < 10:  # Lưu 10 ảnh
    ret, frame = cap.read()
    if not ret:
        print("Không đọc được frame từ camera ảo.")
        break

    filename = os.path.join(output_dir, f"frame_{count}.jpg")
    cv2.imwrite(filename, frame)
    print(f"Đã lưu {filename}")
    count += 1

cap.release()
print("Hoàn thành lưu ảnh.")
