import os
import cv2


def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_video = os.path.join(current_dir, "outputs", "videos", "demo_video.mp4")
    output_dir = os.path.join(current_dir, "outputs", "video_processed")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"视频打开失败: {input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    color_output = os.path.join(output_dir, "annotated_video.mp4")
    gray_output = os.path.join(output_dir, "gray_video.mp4")

    writer_color = cv2.VideoWriter(color_output, fourcc, fps, (width, height))
    writer_gray = cv2.VideoWriter(gray_output, fourcc, fps, (width, height), False)

    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()
        cv2.putText(
            annotated,
            f"Frame: {frame_id}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        writer_color.write(annotated)
        writer_gray.write(gray)

        cv2.imshow("annotated", annotated)
        cv2.imshow("gray", gray)

        frame_id += 1

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    writer_color.release()
    writer_gray.release()
    cv2.destroyAllWindows()

    print("视频处理完成")
    print("输出文件:")
    print(color_output)
    print(gray_output)


if __name__ == "__main__":
    main()