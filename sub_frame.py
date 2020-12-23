# 帧差法
import cv2

# 时间平均法
import glob
import numpy as np


def method2():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    N = 40      # select the first 40 frame for modeling
    mu = np.zeros((h, w), np.uint8)
    for i in range(N):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(frame, (5, 5), 0)
        mu = (mu * i + gaussian) / (i+1)

    mu = np.array(mu, dtype=np.uint8)

    cnt = 0
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian = cv2.GaussianBlur(frame, (5, 5), 0)
        mu = mu / i * (i - 1) + gaussian / i   # update mean image
        mu = np.array(mu, dtype=np.uint8)
        frame = np.array(frame, dtype=np.uint8)
        result = cv2.absdiff(frame, mu)
        ret, thresh = cv2.threshold(
            result, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow('thresh', thresh)
        if cv2.waitKey(30) == 27:
            break


def method1():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    while True:
        ret, frame = cap.read()
        ret, frame_next_origin = cap.read()
        if ret == False:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_next = cv2.cvtColor(frame_next_origin, cv2.COLOR_BGR2GRAY)
        frame_delta = cv2.absdiff(frame_next, frame)
        # result = cv2.Laplacian(frame_delta, cv2.CV_8U, ksize=5)
        _, result = cv2.threshold(frame_delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.morphologyEx(result, cv2.MORPH_OPEN, (5, 5), iterations=3)
        _, contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            contour_area = cv2.contourArea(contour)
            if (contour_area > 500):
                cv2.drawContours(frame_next_origin, contours, i, (255, 0, 0), 3)
        # ret, thresh = cv2.threshold(
        #     frame_delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('thresh', result)
        cv2.imshow('frame', frame_next_origin)
        if cv2.waitKey(30) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def method3():
    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    mu = cv2.GaussianBlur(img, (5,5), 0)
    mu = cv2.cvtColor(mu, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    cov = np.zeros((h, w))
    pro = np.zeros((h, w))
    sav_mu = mu
    a = 0.01
    N = 40
    for i in range(1,N):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(frame, (5,5), 0)
        mu = (blur+(i-1)*sav_mu)/i
        cov = ((blur- mu)**2 + (i - 1) * cov)/ i + (mu - sav_mu)**2
        sav_mu = mu
    cov = cov + 0.1
    cnt = 0
    while True:
        cnt += 1
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurImg = cv2.GaussianBlur(frame, (5,5), 0)
        T = 1e-8
        pro = (2 * np.pi)**(-0.5) * np.exp(-0.5 * (blurImg - mu)**2 / cov) / np.sqrt(cov)
        ret, pro = cv2.threshold(pro, T, 255, cv2.THRESH_BINARY)
        mu = mu +a*(1-pro)*(blurImg - mu)
        cov = cov + a*(1-pro)*((blurImg - mu)**2-cov)
        cv2.imshow('cov', cov)
        if cv2.waitKey(10) == 27:
            break

method1()
