import numpy as np

class DrivingAgent:
    def __init__(self, lane_center_x, safe_distance=100):
        """
        :param lane_center_x: Tọa độ X (pixel) của trung tâm làn đường xe bạn.
        :param safe_distance: Khoảng cách an toàn (pixel) để tránh va chạm.
        """
        self.lane_center_x = lane_center_x
        self.safe_distance = safe_distance

    def decide_action(self, detections, frame_width):
        """
        Quyết định hành động dựa vào các xe NPC đã phát hiện.
        :param detections: DataFrame từ YOLO (results)
        :param frame_width: chiều rộng khung hình
        :return: 'left', 'right', hoặc 'none'
        """
        if detections is None or len(detections) == 0:
            return "none"  # Không có xe nào, giữ nguyên

        # Lấy các xe gần nhất (lọc theo khoảng cách dọc trục Y)
        nearest_cars = []
        for _, row in detections.iterrows():
            x_center = (row['xmin'] + row['xmax']) / 2
            y_center = (row['ymin'] + row['ymax']) / 2
            distance = frame_width - y_center  # đơn giản hóa: khoảng cách pixel
            nearest_cars.append((x_center, y_center, distance))

        # Xe gần nhất (Y nhỏ nhất ~ gần phía dưới màn hình)
        nearest_cars.sort(key=lambda c: c[1], reverse=True)  # lớn nhất Y -> gần hơn
        nearest_x, nearest_y, nearest_dist = nearest_cars[0]

        # Nếu xe quá gần, chọn rẽ tránh
        if nearest_dist < self.safe_distance:
            if nearest_x < self.lane_center_x:
                return "right"  # Xe bên trái -> rẽ phải
            else:
                return "left"   # Xe bên phải -> rẽ trái

        return "none"