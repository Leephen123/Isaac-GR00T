"""
使用指定设备路径显示多个摄像头
不进行枚举扫描

例如：
/dev/video0
/dev/video2
/dev/video4
"""

import glob
import cv2


class MultiCameraViewer:
    def __init__(self, camera_paths):
        self.camera_paths = camera_paths
        self.caps = []

    def open_cameras(self):
        for path in self.camera_paths:
            cap = cv2.VideoCapture(path)

            if not cap.isOpened():
                print(f"[×] 打开失败: {path}")
                continue

            # 可选：设置分辨率/FPS
            # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            # cap.set(cv2.CAP_PROP_FPS, 60)

            ret, frame = cap.read()

            if not ret:
                print(f"[×] 无法读取: {path}")
                cap.release()
                continue

            h, w = frame.shape[:2]

            print(f"[✓] 已打开: {path} ({w}x{h})")

            cv2.namedWindow(path, cv2.WINDOW_NORMAL)
            self.caps.append((path, cap))

        return len(self.caps) > 0

    def run(self):
        while True:
            for path, cap in self.caps:
                ret, frame = cap.read()

                if not ret:
                    continue

                cv2.imshow(path, frame)

            key = cv2.waitKey(1)

            if key & 0xFF == ord("q"):
                break

        self.close()

    def close(self):
        for _, cap in self.caps:
            cap.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    viewer = MultiCameraViewer(glob.glob("/dev/v4l/by-path/*"))

    if viewer.open_cameras():
        viewer.run()
