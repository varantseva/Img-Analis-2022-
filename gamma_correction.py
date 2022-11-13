import sys, os.path, cv2, numpy as np


# def gamma_correction(img: np.ndarray, a: float, b: float) -> np.ndarray:
#     img_normalize = np.power(img / 255, b ) * a * 255
#     img = np.clip(np.around(img_normalize), a_min = 0.0, a_max = 255.0)
#     return img.astype(np.uint8)

def gamma_correction(img: np.ndarray, a: float, b: float) -> np.ndarray:
    uniq_px = np.array([np.power(i / 255, b) * a * 255 for i in range(256)])
    gamma_px = np.clip(np.around(uniq_px), a_min=0.0, a_max=255.0)

    copy_img = img.copy()

    for i in range(256):
        img[copy_img == i] = gamma_px[i]

    return img.astype(np.uint8)

def main():
    assert len(sys.argv) == 5
    src_path, dst_path = sys.argv[1], sys.argv[2]
    a, b = float(sys.argv[3]), float(sys.argv[4])

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = gamma_correction(img, a, b)
    cv2.imwrite(dst_path, result)

if __name__ == '__main__':
    main()
