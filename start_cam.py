import cv2


class StartCam:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)

    def get_frame(self):
        return self.cam.read()

    def cam_test(self):
        while True:
            check, frame = self.get_frame()
            cv2.imshow('video', frame)

            key = cv2.waitKey(1)

            # esc key
            if key == 27:
                break

        self.stop()

    def stop(self):
        self.cam.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    cam = StartCam()
    cam.cam_test()
