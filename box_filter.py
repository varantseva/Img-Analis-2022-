import sys, os.path, cv2, numpy as np
from skimage.transform import integral_image


def box_filter(img: np.ndarray, w: int, h: int) -> np.ndarray:
    if (h - 1) % 2 == 0:
        border_gorisont = int((h - 1) / 2)
    else:
        border_gorisont = int(h / 2)
    if (w - 1) % 2 == 0:
        border_vert = int((w - 1) / 2)
    else:
        border_vert = int(w / 2)
    size = w * h
    pad_img = cv2.copyMakeBorder(img, border_vert, border_vert, border_gorisont, border_gorisont, cv2.BORDER_REPLICATE)

    sums = integral_image(pad_img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            A = sums[i + w - 1][j + h - 1]
            if (i - 1 >= 0 and j - 1 >= 0):
                B = sums[i - 1][j - 1]
                C = sums[i + w - 1][j - 1]
                D = sums[i - 1][j + h - 1]
            elif (i - 1 >= 0):
                B = 0
                C = 0
                D = sums[i - 1][j + h - 1]
            elif (j - 1 >= 0):
                B = 0
                C = sums[i + w - 1][j - 1]
                D = 0
            else:
                B, C, D = 0, 0, 0

            img[i][j] = round((A + B - C - D) / size)
    return img.astype(np.uint8)

def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    w, h = int(sys.argv[3]), int(sys.argv[4])
    assert w > 0
    assert h > 0

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = box_filter(img, w, h)
    cv2.imwrite(dst_path, result)

if __name__ == '__main__':
    main()
