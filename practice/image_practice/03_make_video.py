import os
import cv2


def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    frame_dir = os.path.join(current_dir, "outputs", "source_frames")
    video_dir = os.path.join(current_dir, "outputs", "videos")
    os.makedirs(video_dir, exist_ok=True)

    frame_files = sorted(
        [f for f in os.listdir(frame_dir) if f.endswith(".png")]
    )

    if not frame_files:
        print("没有找到帧图片，请先运行 01_generate_images.py")
        return

    first_frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    if first_frame is None:
        print("第一帧读取失败")
        return

    height, width = first_frame.shape[:2]
    fps = 20

    output_path = os.path.join(video_dir, "demo_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for file_name in frame_files:
        frame_path = os.path.join(frame_dir, file_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"跳过损坏帧: {frame_path}")
            continue
        writer.write(frame)

    writer.release()
    print(f"视频生成完成: {output_path}")


if __name__ == "__main__":
    main()