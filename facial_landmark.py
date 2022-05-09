import dlib
import cv2
import start_cam
import threading
import numpy as np
import math

eye = cv2.imread('image/test_face.png')


class FacialDetect:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.using_dot = [1, 5, 9, 13, 17, 28, 31, 34, 32, 36,
                          37, 38, 39, 40, 41, 42, 43, 44, 45,
                          46, 47, 48, 61, 62, 63, 64, 65, 66,
                          67, 68]

        self.cam = start_cam.StartCam()
        self.shape = None

    def distance_two_points(self, a, b):
        return math.sqrt(
            (a[0] - b[0])**2 + (a[1] - b[1])
        )

    def frame_landmarking(self, frame):
        img = cv2.flip(frame, 1)

        r = 350. / img.shape[1]
        dim = (350, int(img.shape[0] * r))
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        rects = self.detector(resized, 1)
        for i, rect in enumerate(rects):
            # l = rect.left()
            # r = rect.right()
            # t = rect.top()
            # b = rect.bottom()

            self.shape = self.predictor(resized, rect)

            for j in self.using_dot:
                x, y = self.shape.part(j - 1).x, self.shape.part(j - 1).y
                cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)

            # cv2.rectangle(resized, (l, t), (r, b), (0, 255, 0), 2)

        return resized

    def test_landmarking(self):
        h, w, chn = eye.shape
        pre_tilt = 0
        pre_pan = 0

        while True:

            check, frame = self.cam.get_frame()
            marked_frame = self.frame_landmarking(frame)

            if self.shape is not None:
                pan = (self.shape.part(36).x - self.shape.part(0).x) - \
                        (self.shape.part(16).x - self.shape.part(45).x)
                if pan > 35:
                    pan = 35
                elif pan < -35:
                    pan = -35

                tilt = -int(
                    4*(self.shape.part(36).y + self.shape.part(46).y)/2 -
                    4*(self.shape.part(16).y + self.shape.part(0).y)/2
                    )
                if tilt > 35:
                    tilt = 35
                elif tilt < -35:
                    tilt = -35

            else:
                pan = 0
                tilt = 0

            marked_frame = cv2.putText(marked_frame, 'tilt: {}'.format(tilt), (10, 230),
                                       cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1, cv2.LINE_AA
                                       )
            marked_frame = cv2.putText(marked_frame, 'pan: {}'.format(pan), (10, 250),
                                       cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 1, cv2.LINE_AA
                                       )

            pts1 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])
            pts2 = np.float32([[0, 0], [0, h], [w, 0], [w, h]])

            if abs(pan - pre_pan) > 6:
                if pan >= 0:
                    pts2[2] = pts2[2] + [-pan*1.5, pan * 2]
                    pts2[3] = pts2[3] + [-pan*1.5, -pan * 2]

                    # pts2 = np.float32([[0, 0], [0, h], [w-pan*1.5, pan * 2], [w-pan*1.5, h-pan * 2]])
                else:
                    pts2[0] = pts2[0] + [-pan*1.5, -pan * 2]
                    pts2[1] = pts2[1] + [-pan*1.5, pan * 2]
                    # pts2 = np.float32([[-pan*1.5, -pan * 2], [-pan*1.5, h + pan * 2], [w, 0], [w, h]])
            else:
                if pre_pan >= 0:
                    pts2[2] = pts2[2] + [-pre_pan*1.5, pre_pan * 2]
                    pts2[3] = pts2[3] + [-pre_pan*1.5, -pre_pan * 2]

                    # pts2 = np.float32([[0, 0], [0, h], [w-pan*1.5, pan * 2], [w-pan*1.5, h-pan * 2]])
                else:
                    pts2[0] = pts2[0] + [-pre_pan*1.5, -pre_pan * 2]
                    pts2[1] = pts2[1] + [-pre_pan*1.5, pre_pan * 2]
                    # pts2 = np.float32([[-pan*1.5, -pan * 2], [-pan*1.5, h + pan * 2], [w, 0], [w, h]])

            if abs(tilt - pre_tilt) > 6:
                if tilt >= 0:
                    pts2[0] = pts2[0] + [tilt*2, tilt*1.5]
                    pts2[2] = pts2[2] + [-tilt*2, tilt*1.5]
                else:
                    pts2[1] = pts2[1] + [-tilt*2, tilt*1.5]
                    pts2[3] = pts2[3] + [tilt*2, tilt*1.5]
            else:
                if pre_tilt >= 0:
                    pts2[0] = pts2[0] + [pre_tilt*2, pre_tilt*1.5]
                    pts2[2] = pts2[2] + [-pre_tilt*2, pre_tilt*1.5]
                else:
                    pts2[1] = pts2[1] + [-pre_tilt*2, pre_tilt*1.5]
                    pts2[3] = pts2[3] + [pre_tilt*2, pre_tilt*1.5]

            mtrx = cv2.getPerspectiveTransform(pts1, pts2)
            eye_fix = cv2.warpPerspective(eye, mtrx, (h, w))

            # cv2.imshow('eye', eye_fix)
            # cv2.imshow('test', marked_frame)
            added = cv2.hconcat([marked_frame, cv2.resize(eye_fix, (350, 262), interpolation=cv2.INTER_AREA)])
            cv2.imshow('test', added)

            pre_tilt = tilt
            pre_pan = pan

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    marking = FacialDetect()
    t = threading.Thread(target=marking.test_landmarking)
    t.start()
    # marking.test_landmarking()
