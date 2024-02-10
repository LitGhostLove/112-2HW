import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Microsoft JhengHei"]

def zmMinFilterGray(src, r=7):
    '''最小值滤波，r是滤波器半径'''
    return cv2.erode(src, np.ones((2 * r + 1, 2 * r + 1)))

def guidedfilter(I, p, r, eps):
    height, width = I.shape
    m_I = cv2.boxFilter(I, -1, (r, r))
    m_p = cv2.boxFilter(p, -1, (r, r))
    m_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = m_Ip - m_I * m_p
    m_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = m_II - m_I * m_I

    a = cov_Ip / (var_I + eps)
    b = m_p - a * m_I
    m_a = cv2.boxFilter(a, -1, (r, r))
    m_b = cv2.boxFilter(b, -1, (r, r))
    return m_a * I + m_b

def Defog(m, r, eps, w, maxV1):                 #输入rgb图像，值范围[0,1]
    '''计算大气遮罩图像V1和光照值A, V1 = 1-t/A'''
    V1 = np.min(m, 2)                           #得到暗通道图像
    Dark_Channel = zmMinFilterGray(V1, 5)
#    cv2.imshow('wu_Dark',Dark_Channel)    #查看暗通道
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()

    V1 = guidedfilter(V1, Dark_Channel, r, eps)  #使用引导滤波优化
    bins = 2000
    ht = np.histogram(V1, bins)                  #计算大气光照A
    d = np.cumsum(ht[0]) / float(V1.size)
    for lmax in range(bins - 1, 0, -1):
        if d[lmax] <= 0.999:
            break
    A = np.mean(m, 2)[V1 >= ht[1][lmax]].max()
    V1 = np.minimum(V1 * w, maxV1)               #对值范围进行限制
    return V1, A

def deHaze(m, r=81, eps=0.001, w=0.95, maxV1=0.80, bGamma=False):
    Y = np.zeros(m.shape)
    Mask_img, A = Defog(m, r, eps, w, maxV1)             #得到遮罩图像和大气光照

    for k in range(3):
        Y[:,:,k] = (m[:,:,k] - Mask_img)/(1-Mask_img/A)  #颜色校正
    Y = np.clip(Y, 0, 1)
    if bGamma:
        Y = Y ** (np.log(0.5) / np.log(Y.mean()))       #gamma校正,默认不进行该操作
    return Y

if __name__ == '__main__':
    HW2 = cv2.imread('HW2.png')
    HW2_1_1 = deHaze(cv2.imread('HW2.png') / 255.0) * 255
    cv2.imwrite('HW2_1_1.png', HW2_1_1)      #結果存檔
    
HW2_1_1 = cv2.imread('HW2_1_1.png')   
temp = cv2.GaussianBlur(HW2_1_1, (5, 5), 1)
HW2_1_2 = cv2.addWeighted(HW2_1_1, 2, temp, -1, 0)
cv2.imwrite('HW2_1_2.png', HW2_1_2)

HW2_1_2 = cv2.imread('HW2_1_2.png')  
HW2_1_3 = cv2.Canny(HW2_1_2, 50, 100)             # minVal=50, maxVal=100
cv2.imwrite('HW2_1_3.png', HW2_1_3)

HW2 = cv2.imread('HW2.png') 
HW2_1_1 = cv2.imread('HW2_1_1.png') 
HW2_1_2 = cv2.imread('HW2_1_2.png')
HW2_1_3 = cv2.imread('HW2_1_3.png')
HW2 = cv2.cvtColor(HW2,cv2.COLOR_BGR2RGB)
HW2_1_1 = cv2.cvtColor(HW2_1_1,cv2.COLOR_BGR2RGB)
HW2_1_2 = cv2.cvtColor(HW2_1_2,cv2.COLOR_BGR2RGB)
HW2_1_3 = cv2.cvtColor(HW2_1_3,cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.title("原始影像")
plt.imshow(HW2)
plt.axis('off')
plt.subplot(222)
plt.title("暗通道去霧")
plt.imshow(HW2_1_1)
plt.axis('off')
plt.subplot(223)
plt.title("銳化")
plt.imshow(HW2_1_2)
plt.axis('off')
plt.subplot(224)
plt.title("Canny邊緣檢測")
plt.imshow(HW2_1_3)
plt.axis('off')
plt.show()


'''-------------------------------------------------------------------------'''


HW2 = cv2.imread('HW2.png') 
kerne = np.ones((5,5),np.uint8)        # 建立5x5內核
HW2_2_1 = cv2.erode(HW2, kerne)           # 腐蝕操作
cv2.imwrite('HW2_2_1.png', HW2_2_1)

HW2_2_1 = cv2.imread('HW2_2_1.png')  
ret, mask = cv2.threshold(HW2_2_1, 250, 255, cv2.THRESH_BINARY)   
# 遮罩處理, 適度增加要處理的表面
kernal = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
mask = cv2.dilate(mask, kernal)
HW2_2_2 = cv2.inpaint(HW2_2_1, mask[:, :, -1], 5, cv2.INPAINT_NS)
cv2.imwrite('HW2_2_2.png', HW2_2_2)


HW2 = cv2.imread('HW2.png') 
HW2_2_1 = cv2.imread('HW2_2_1.png') 
HW2_2_2 = cv2.imread('HW2_2_2.png') 
HW2 = cv2.cvtColor(HW2,cv2.COLOR_BGR2RGB)
HW2_2_1 = cv2.cvtColor(HW2_2_1,cv2.COLOR_BGR2RGB)
HW2_2_2 = cv2.cvtColor(HW2_2_2,cv2.COLOR_BGR2RGB)
plt.subplot(131)
plt.title("原始影像")
plt.imshow(HW2)
plt.axis('off')
plt.subplot(132)
plt.title("腐蝕")
plt.imshow(HW2_2_1)
plt.axis('off')
plt.subplot(133)
plt.title("影像修復")
plt.imshow(HW2_2_2)
plt.axis('off')
plt.show()
