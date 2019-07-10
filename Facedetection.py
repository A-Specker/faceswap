import numpy as np
import os
import logging
import cv2
import dlib
from detector import FaceDetector
from detector import Tools

'''
Lets try different face detectors
https://www.learnopencv.com/face-detection-opencv-dlib-and-deep-learning-c-python/
'''

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)



def load_img():
    # im = cv2.imread('files/girl.png', cv2.IMREAD_COLOR)
    # im = cv2.imread('files/kevin.png')
    im = cv2.imread('files/quadruppel_kevin.png')

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if im is None:
        print('No image found')
        quit()
    return im


def show_img(im):
    cv2.imshow('image', cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)


def crop_img_to_face(im, faces_coord):
    # x1, y1, x2, y2 = faces_coord
    # face_area = im[x1:x2, y1:y2]
    image = im.copy()
    face_area = image[faces_coord[0]:faces_coord[2], faces_coord[1]:faces_coord[3]]
    return face_area


def pose_estimation(im, coords, predictor):
    # http://dlib.net/face_landmark_detection.py.html
    shape = []
    for j in range(coords.shape[0]):
        rect = dlib.rectangle(coords[j][0], coords[j][1], coords[j][2], coords[j][3])
        shape.append(predictor(im, rect))
    return np.asarray(shape)

def load_pose_estimator():
    return dlib.shape_predictor('files/shape_predictor_68_face_landmarks.dat')

def predictor_shape_to_coord_list(shp):
    faces = []
    for j in range(shp.shape[0]):
        pts = []
        for i in range(68):
            pts.append([shp[j].part(i).x, shp[j].part(i).y])
        faces.append(pts)
    return np.asarray(faces)


def add_marks_to_img(im, pts):
    col = [255, 0, 0]
    for j in range(pts.shape[0]):
        for k in range(pts.shape[1]):
            # because for loops are fucking nice
            cv2.circle(im, (pts[j][k][0], pts[j][k][1]), 2, col, -1)
    return im


def draw_box(im, b, c):
    for j in range(b.shape[0]):
        bo = cv2.rectangle(im, (b[j][0], b[j][1]), (b[j][2], b[j][3]), c, 2)
    return bo


def get_conv_hull(p):
    hulls = []
    ps = np.reshape(p, (p.shape[0], p.shape[1], 1, p.shape[2]))
    for j in range(p.shape[0]):
        hulls.append(cv2.convexHull(ps[j], returnPoints=True))
    return np.asarray(hulls)


if __name__ == '__main__':

    logging.info('Start.')

    img = load_img()
    det = FaceDetector('haar') # haar, nn, hog
    pose_estimator = load_pose_estimator()
    logging.info('Detector loaded.')

    face_coords = det.detect(img)
    img1 = draw_box(img.copy(), face_coords, (0, 255, 0))
    logging.info('Face detected.')
    
    shape = pose_estimation(img, face_coords, pose_estimator)
    marks = predictor_shape_to_coord_list(shape)
    logging.info('Landmarks calculated.')

    hulls = get_conv_hull(marks)
    logging.info('Hull calculated.')



    rect = (face_coords[0][0], face_coords[0][1], face_coords[0][2],face_coords[0][3])
    delaunay = Tools.calculateDelaunayTriangles(rect, hulls[0])  


    


    # img1 = add_marks_to_img(img1, marks)
    # logging.warning('Landmarks drawn.')

    # show_img(img1)

    # hull1 = get_conv_hull((np.asarray(marks1)))

    # logging.warning('Hull Done.')

    # print(hull1.shape)

    logging.warning('Done.')

    cv2.destroyAllWindows()



