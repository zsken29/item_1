import os
import cv2


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def main():
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(current_dir, "outputs", "source_images", "test_image.png")
    output_dir = os.path.join(current_dir, "outputs", "processed_images")
    ensure_dir(output_dir)

    img = cv2.imread(input_path)
    if img is None:
        print(f"读取失败: {input_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 160)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = img.copy()
    cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)

    cv2.imwrite(os.path.join(output_dir, "gray.png"), gray)
    cv2.imwrite(os.path.join(output_dir, "blur.png"), blur)
    cv2.imwrite(os.path.join(output_dir, "edges.png"), edges)
    cv2.imwrite(os.path.join(output_dir, "contours.png"), contour_img)

    print("图像处理完成，结果已保存到:", output_dir)
    print("检测到轮廓数量:", len(contours))

    cv2.imshow("original", img)
    cv2.imshow("gray", gray)
    cv2.imshow("edges", edges)
    cv2.imshow("contours", contour_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()