import cv2 as cv
import numpy as np

# 高斯濾波核大小
blur_ksize = 5

# Canny邊緣檢測高低閾值
#1, 2, 3
#canny_lth = 50
#canny_hth = 150
#4
#canny_lth = 100
#canny_hth = 150
#5
canny_lth = 150
canny_hth = 250

# 霍夫變換參數
rho = 1
theta = np.pi / 180
threshold = 15
#1, 2, 3
#min_line_len = 40
#max_line_gap = 20
#4
#min_line_len = 130
#max_line_gap = 33
#5
min_line_len = 60
max_line_gap = 90


def process_an_image(img):
    # 1. 灰度化、濾波和Canny
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    blur_gray = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 1)
    edges = cv.Canny(blur_gray, canny_lth, canny_hth)
    cv.imshow("edges", edges)
    
    # 2. 標記四個坐標點用於ROI截取
    rows, cols = edges.shape
    points = np.array([[(0, rows), (200, 170), (300, 170), (cols, rows)]])
    roi_edges = roi_mask(edges, points)
    cv.imshow("roi_edges", roi_edges)
    
    # 3. 霍夫直線提取
    drawing, lines = hough_lines(roi_edges, rho, theta,
                                 threshold, min_line_len, max_line_gap)

    # 4. 車道擬合計算
    draw_lanes(drawing, lines)

    # 5. 最終將結果合在原圖上
    result = cv.addWeighted(img, 1, drawing, 2, 0)
    # cv.imshow("result", result)
    return result


def roi_mask(img, corner_points):
    # 創建掩膜
    mask = np.zeros_like(img)
    cv.fillPoly(mask, corner_points, 255)

    masked_img = cv.bitwise_and(img, mask)
    cv.imshow("masked_img", masked_img)  
    return masked_img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # 統計概率霍夫直線變換
    lines = cv.HoughLinesP(img, rho, theta, threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)

    # 新建一副空白畫布
    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # 畫出直線檢測結果
    draw_lines(drawing, lines)
    # print(len(lines))
    cv.imshow("drawing", drawing)
    return drawing, lines


def draw_lines(img, lines, color=[0, 0, 255], thickness=10):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    # a. 划分左右車道
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return

    # b. 清理異常數據
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)

    # c. 得到左右車道線點的集合，擬合直線
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2)
                                 for line in left_lines for x1, y1, x2, y2 in line]

    right_points = [(x1, y1)
                    for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + \
        [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]

    left_results = least_squares_fit(left_points, 325, img.shape[0])
    right_results = least_squares_fit(right_points, 325, img.shape[0])

    # 注意這里點的順序
    vtxs = np.array(
        [[left_results[1], left_results[0], right_results[0], right_results[1]]])
    # d.填充車道區域
    cv.fillPoly(img, vtxs, (0, 255, 0))

    # 或者只畫車道線
    # cv.line(img, left_results[0], left_results[1], (0, 255, 0), thickness)
    # cv.line(img, right_results[0], right_results[1], (0, 255, 0), thickness)


def clean_lines(lines, threshold):
    # 迭代計算斜率均值，排除掉與差值差異較大的數據
    slope = [(y2 - y1) / (x2 - x1)
             for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def least_squares_fit(point_list, ymin, ymax):
    # 最小二乘法擬合
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]

    # polyfit第三個參數為擬合多項式的階數，所以1代表線性
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)  # 獲取擬合的結果

    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))

    return [(xmin, ymin), (xmax, ymax)]


if __name__ == "__main__":
    img = cv.imread('test_images/5.jpg')
    cv.imshow("img", img)
    result = process_an_image(img)
    cv.imshow("lane", result)
    cv.waitKey(0)
