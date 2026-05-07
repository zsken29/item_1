import os
import cv2


def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_video = os.path.join(current_dir, "outputs", "videos", "demo_video.mp4")
    output_dir = os.path.join(current_dir, "outputs", "motion_detection")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"视频打开失败: {input_video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_path = os.path.join(output_dir, "motion_detection.mp4")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=100,
        varThreshold=25,
        detectShadows=False
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = bg_subtractor.apply(frame)

        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result = frame.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(
                result,
                "Moving Object",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        writer.write(result)

        cv2.imshow("motion_result", result)
        cv2.imshow("foreground_mask", thresh)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"运动检测完成: {output_path}")


if __name__ == "__main__":
    main()