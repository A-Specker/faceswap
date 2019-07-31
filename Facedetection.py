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

#logging.basicConfig(format='%(asctime)s %(message)s')#, level=logging.DEBUG)
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)



def load_img():
    #im = cv2.imread('files/girl.png', cv2.IMREAD_COLOR)
    #im = cv2.imread('files/kevin.png')
    # im= cv2.imread('files/quadruppel_kevin.png')
    #im = cv2.imread('files/tutorial.png')
    im = cv2.imread('files/beide.png', cv2.IMREAD_COLOR)
#   im = cv2.resize(im, (im.shape*2))
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
            cv2.circle(im, (pts[j][k][0], pts[j][k][1]), 2, (0, 0, (k*255//68)), -1)
    return im


def draw_box(im, b, c):
    for j in range(b.shape[0]):
        bo = cv2.rectangle(im, (b[j][0], b[j][1]), (b[j][2], b[j][3]), c, 2)
    return bo

# lets asume the faces are more or less looking in the same direction
# --> calculate one hull, return index instead of points and use same index
#       for both faces
def get_conv_hull(p):
    #hulls = []
    #ps = np.reshape(p, (p.shape[0], p.shape[1], 1, p.shape[2]))
    #for j in range(p.shape[0]):
    #    hulls.append(cv2.convexHull(ps[j], returnPoints=True))
    #return np.asarray(hulls)
    hulls = []
    ps = np.reshape(p, (p.shape[0], p.shape[1], 1, p.shape[2]))
    res = cv2.convexHull(ps[0], returnPoints=False)
    # convert idx to coords
    for j in range(len(res)):
        for k in range(p.shape[0]):
            hulls.append(p[k][res])
    return np.asarray(hulls)

def draw_delaunay(im, tris):
    col = (255, 255, 0)
    for j in tris:
        i = tuple(map(tuple, j))
        cv2.line(im, i[0], i[1], col, 1)
        cv2.line(im, i[1], i[2], col, 1)
        cv2.line(im, i[2], i[0], col, 1)
    return im


def delIxs2coords(points, delTris):
    coords = []
    for i in delTris:
        coords.append((points[i[0]], points[i[1]], points[i[2]]))
    return coords

def calc_delaunay(idx1, idx2, face_coords, hulls):
    rect = (face_coords[idx1][0], face_coords[idx1][1], face_coords[idx1][2], face_coords[idx1][3])
    delaun_tris = Tools.calculateDelaunayTriangles(rect, hulls[idx1])
    delaunay_1 = delIxs2coords(hulls[idx1], delaun_tris)
    delaunay_2 = delIxs2coords(hulls[idx2], delaun_tris)
    return delaunay_1, delaunay_2


def draw_delaunay(img1, delaunay_1, delaunay_2):
    for i in range(len(delaunay_1)):
        triangle_cnt = np.array(delaunay_1[i])
        triangle_cnt2 = np.array(delaunay_2[i])
        tmpIM = img1.copy()
        cv2.drawContours(tmpIM, [triangle_cnt], 0, (0, 255, 0), -1)
        cv2.drawContours(tmpIM, [triangle_cnt2], 0, (255, 0, 0), -1)
        show_img(tmpIM)


def warp_tris_to_face(source, target, del_1, del_2):
    for i in range(len(del_1)):
        d1 = []
        d2 = []
        for j in range(len(del_1[i])):
            d1.append(tuple(del_1[i][j][0]))
            d2.append(tuple(del_2[i][j][0]))
        Tools.warpTriangle(source, target, d1, d2)
        Tools.warpTriangle(source, target, d2, d1)
    return target

def smooth_in_face(hull, img, warped_im, shape, ty):
    hullMask = []
    for i in range(len(hull)):
        hullMask.append(hull[i])
    mask = np.zeros(shape, dtype=ty)
    cv2.fillConvexPoly(mask, np.int32(hullMask), (255, 255, 255))
    r = cv2.boundingRect(np.float32(hull))
    center = (r[0] + r[2]//2, r[1]+r[3]//2)
    ret = cv2.seamlessClone(np.uint8(warped_im), img, mask, center, 1)
    return ret


if __name__ == '__main__':

    logging.info('Start.')

    img = load_img()
    det = FaceDetector('nn') # haar, nn, hog
    pose_estimator = load_pose_estimator()
    logging.info('Detector loaded.')




    face_coords = det.detect(img.copy())
    logging.info('{} Faces detected.'.format(str(face_coords.shape[0])))
    # print(face_coords[0].shape)
    #   print(face_coords[0].shape)

    face_shape = pose_estimation(img, face_coords, pose_estimator)
    marks = predictor_shape_to_coord_list(face_shape)
    np.save("fcs_crds.npy", face_coords)

    np.save("fcs_shp.npy", face_shape)
    logging.info('Landmarks calculated.')

    hulls = get_conv_hull(marks)
    logging.info('Hull calculated.')

    # currently, two face from all detected faces are used, maybe change this
    faceIdx0 = 0
    faceIdx1 = 1

    delaunay_1, delaunay_2 = calc_delaunay(faceIdx0, faceIdx1, face_coords, hulls)
    logging.info('Delaunay calculated')

    warped_img = warp_tris_to_face(img, img.copy(), delaunay_2, delaunay_1)
    logging.info('Triangles warped.')

    swaped_img = smooth_in_face(hulls[faceIdx0], img, warped_img, img.shape, img.dtype)
    swaped_img = smooth_in_face(hulls[faceIdx1], swaped_img, warped_img, img.shape, img.dtype)
    logging.info('Faces swaped.')
    # img1 = add_marks_to_img(img1, marks)
    # show_img(img)
    # show_img(img1)
    # show_img(warped_img)
    # show_img(swaped_img)

    logging.info('Done.')
    cv2.destroyAllWindows()

