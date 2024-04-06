import cv2
import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pictPath = "haar_carplate.xml"                          # 哈爾特徵檔路徑
img = cv2.imread("testCar/cartest1.jpg")                # 讀辨識的影像
car_cascade = cv2.CascadeClassifier(pictPath)           # 讀哈爾特徵檔
# 執行辨識
plates = car_cascade.detectMultiScale(img, scaleFactor=1.05, minNeighbors=3,
         minSize=(20,20),maxSize=(155,50))  
if len(plates) > 0 :                                    # 有偵測到車牌
    for (x, y, w, h) in plates:                         # 標記車牌  
        carplate = img[y:y+h, x:x+w]                    # 車牌影像        
else:
    print("偵測車牌失敗")

resized = cv2.resize(carplate, (150,50), interpolation= cv2.INTER_LINEAR)
cv2.imshow('resized', resized)                          # 顯示所讀取的車輛
ret, dst = cv2.threshold(resized, 120, 255, cv2.THRESH_BINARY)  # 二值化
cv2.imshow('Car binary', dst)                           # 顯示二值化車牌

# 加上白色邊框
dst1 = cv2.copyMakeBorder(dst, 20, 20, 20, 20, cv2.BORDER_CONSTANT,value=(255,255,255));
cv2.imshow('border', dst1)

cv2.imwrite("car_plate.jpg", dst1)                     # 寫入儲存

#----------------------------------------

# 以灰度模式读取图片
gray = cv2.imread('car_plate.jpg', 0)  #車牌檔案
result = cv2.imread('T0.jpg', 0)        #一張空圖，方便串接

# 二值化
ret,thresh1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh1, 2, 2)
flag = 1
letter_image_regions = []  #文字圖形串列
for cnt in contours:
    # 最小的外接矩形
    x, y, w, h = cv2.boundingRect(cnt)
    letter_image_regions.append((x, y, w, h))  #輪廓資料加入串列
letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])  #按X坐標排序
# print(letter_image_regions)
    
i=1
for letter_bounding_box in letter_image_regions:  #依序處理輪廓資料
    x, y, w, h = letter_bounding_box
    if w >= 7 and h >= 30 and w*h >= 200 and w*h <= 2000:  #長度*高度正確才是文字
        letter_image = gray[y:y+h, x:x+w]  #擷取圖形
        resized = cv2.resize(letter_image, (w,40), interpolation= cv2.INTER_LINEAR)
        # 加上白色邊框
        dst = cv2.copyMakeBorder(resized, 10, 10, 9, 9, cv2.BORDER_CONSTANT,value=(255,255,255));

        cv2.imwrite('T{}.jpg'.format(i), dst)  #存檔
        i += 1
        result = np.concatenate((result,dst), axis = 1)      

ret,result1 = cv2.threshold(result, 200, 255, cv2.THRESH_BINARY)
cv2.imshow("result1",result1)
text = pytesseract.image_to_string(result1) # 執行辨識
print(f"車號是 : {text}")

#----------------------------------------

cv2.waitKey(0)
cv2.destroyAllWindows()
