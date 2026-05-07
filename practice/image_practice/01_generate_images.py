import os
import cv2
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def create_single_image(save_path):
    img = np.full((480, 640, 3), 255, dtype=np.uint8)

    cv2.rectangle(img, (50, 80), (220, 260), (0, 255, 0), -1)
    cv2.circle(img, (400, 180), 70, (255, 0, 0), -1)
    cv2.line(img, (100, 400), (540, 400), (0, 0, 255), 5)

    cv2.putText(
        img,
        "OpenCV Demo",
        (170, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 0),
        2
    )

    cv2.imwrite(save_path, img)
    print(f"已保存单张图片: {save_path}")


def create_frame_sequence(save_dir, num_frames=60):
    ensure_dir(save_dir)

    width, height = 640, 480

    for i in range(num_frames):
        img = np.full((height, width, 3), 255, dtype=np.uint8)

        rect_x = 30 + i * 6
        circle_y = 80 + i * 4

        cv2.rectangle(img, (rect_x, 120), (rect_x + 120, 240), (0, 200, 0), -1)
        cv2.circle(img, (450, circle_y), 45, (255, 0, 0), -1)

        cv2.putText(
            img,
            f"Frame {i:03d}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 0),
            2
        )

        cv2.putText(
            img,
            "Moving Shapes",
            (360, 440),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (50, 50, 50),
            2
        )

        frame_path = os.path.join(save_dir, f"frame_{i:03d}.png")
        cv2.imwrite(frame_path, img)

    print(f"已生成连续帧: {save_dir}，共 {num_frames} 帧")


def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(current_dir, "outputs")
    image_dir = os.path.join(base_dir, "source_images")
    frame_dir = os.path.join(base_dir, "source_frames")

    ensure_dir(image_dir)
    ensure_dir(frame_dir)

    create_single_image(os.path.join(image_dir, "test_image.png"))
    create_frame_sequence(frame_dir, num_frames=60)


if __name__ == "__main__":
    main()