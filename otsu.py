import sys, os.path, cv2, numpy as np


def otsu(img: np.ndarray) -> np.ndarray:
    t_optimal = 0
    sigma_min = 0
    hist, bins = np.histogram(img, np.arange(0, 257))

    n = sum(hist)
    nt = sum(bins[:-1] * hist)
    q1 = 0

    mu1 = (bins[:-1] * hist)
    for i in range(1, len(mu1)):
        mu1[i] += mu1[i - 1]

    for t in range(0, 256):
        q1 += hist[t]
        q2 = n - q1
        if q1 == 0 or q2 == 0:
            continue
        mu2 = (nt - mu1[t]) / q2
        sigma = q1 * q2 * (mu1[t]/q1 - mu2) ** 2
        if sigma > sigma_min:
            sigma_min = sigma
            t_optimal = t

    for i in range(img.shape[0]):
        img[i] = np.where( img[i]<=t_optimal, 0, 255)

    return img


def main():
    assert len(sys.argv) == 3
    src_path, dst_path = sys.argv[1], sys.argv[2]

    assert os.path.exists(src_path)
    img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    result = otsu(img)
    cv2.imwrite(dst_path, result)


if __name__ == '__main__':
    main()
