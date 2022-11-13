import sys, os.path, cv2, numpy as np


def autocontrast(img: np.ndarray, white_percent: float, black_percent: float) -> np.ndarray:

    img1 = np.ravel(img)
    sort_img = np.sort(img1)
    len1 = len(img1)

    Lmin = 0
    Lmax = 255
    Imax = sort_img[-1]
    Imin = sort_img[0]

    white = sort_img[int(len1 * white_percent)]
    black = sort_img[int(len1 * white_percent)]

    c = ((Lmax - Lmin) / (Imax - Imin))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] <= white:
                img[i][j] = 0
            elif img[i][j] >= black:
                img[i][j] = 255
            else:
                img[i][j] = (img[i][j] - Imin) * c

    return img.astype(np.uint8)

def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    white_percent, black_percent = float(sys.argv[3]), float(sys.argv[4])
    assert 0 <= white_percent < 1
    assert 0 <= black_percent < 1

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = autocontrast(img, white_percent, black_percent)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
