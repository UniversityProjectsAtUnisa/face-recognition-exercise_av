import cv2

class SharpeningFilter:
    def __init__(self, kernel_size=None, th1=100, th2=200):
        if kernel_size == None:
            kernel_size = (5, 5)
        self.kernel_size = kernel_size
        self.th1 = th1
        self.th2 = th2

    def __call__(self, image):
        smooth = cv2.GaussianBlur(image, self.kernel_size, 0)
        edges = cv2.Canny(smooth, self.th1, self.th2)
        return image + edges 