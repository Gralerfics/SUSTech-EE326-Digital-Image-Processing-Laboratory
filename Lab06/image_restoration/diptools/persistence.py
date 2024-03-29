import cv2


def load_gray(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def save_gray(filepath, img):
    if filepath[-4:] == ".jpg":
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
    else:
        cv2.imwrite(filepath, img)



def show(img, win_name='img_grayscale', delay=0):
    cv2.imshow(win_name, img)
    cv2.waitKey(delay)

