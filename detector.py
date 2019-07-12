import numpy as np
import cv2

class FaceDetector(object):
    def __init__(self, detection):
        self.detection = detection
        self.detector = None
        self.assign_detector()
    
    def assign_detector(self):
        if self.detection is 'haar':
            self.detector = cv2.CascadeClassifier("files/haarcascade_frontalface_default.xml")

        if self.detection is 'hog':
            import dlib
            self.detector = dlib.get_frontal_face_detector()

        if self.detection is 'nn':
            self.tf_assign()

    def tf_assign(self):
        import tensorflow as tf
        self.face_corrector = FaceCorrector()

        # load model
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session()
            
            with tf.gfile.GFile('files/face_yolo.pb', "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="") # If not, name is appended in op name

            self.img = tf.get_default_graph().get_tensor_by_name("img:0")
            self.training = tf.get_default_graph().get_tensor_by_name("training:0")
            self.prob = tf.get_default_graph().get_tensor_by_name("prob:0")
            self.x_center = tf.get_default_graph().get_tensor_by_name("x_center:0")
            self.y_center = tf.get_default_graph().get_tensor_by_name("y_center:0")
            self.w = tf.get_default_graph().get_tensor_by_name("w:0")
            self.h = tf.get_default_graph().get_tensor_by_name("h:0")
        # load aux vars
        cols = np.zeros(shape=[1, 9])
        for i in range(1, 9):
            cols = np.concatenate((cols, np.full((1, 9), i)), axis=0)

        self.cols = cols
        self.rows = cols.T

    def detect(self, im):
        if self.detection is 'haar':
            if im.shape[2] > 1:
                im_sw = cv2.cvtColor(im.copy(), cv2.COLOR_BGR2GRAY)
            else:
                im_sw = im.copy()

            im_sw = cv2.equalizeHist(im_sw)
            faces_coords = self.detector.detectMultiScale(im_sw, scaleFactor=1.2, minNeighbors=3, minSize=(30,30), flags=cv2.CASCADE_SCALE_IMAGE)
            faces_coords[:, 2] = faces_coords[:, 0] + faces_coords[:, 2]
            faces_coords[:, 3] = faces_coords[:, 1] + faces_coords[:, 3]
            return faces_coords

        if self.detection is 'hog':
            det = self.detector(im, 1)
            ret = []
            for d in det:
                ret.append([d.left(), d.top(), d.right(), d.bottom()])
            return np.asarray(ret)

        if self.detection is 'nn':

            def _absolute_bboxes(pred, frame, thresh):
                img_h, img_w, _ = frame.shape
                p, x, y, w, h = pred

                mask = p > thresh

                x += self.cols
                y += self.rows

                p, x, y, w, h = p[mask], x[mask], y[mask], w[mask], h[mask]

                ret = []

                for j in range(x.shape[0]):
                    xc, yc = int((x[j]/9)*img_w), int((y[j]/9)*img_h)
                    wi, he = int(w[j]*img_w), int(h[j]*img_h)
                    ret.append((xc, yc, wi, he, p[j]))

                return ret

            def _nonmax_supression(bboxes, thresh=0.2):
                SUPPRESSED = 1
                NON_SUPPRESSED = 2

                N = len(bboxes)
                status = [None] * N
                for i in range(N):
                    if status[i] is not None:
                        continue

                    curr_max_p = bboxes[i][-1]
                    curr_max_index = i

                    for j in range(i+1, N):
                        if status[j] is not None:
                            continue

                        metric = _iou(bboxes[i], bboxes[j])
                        if metric > thresh:
                            if bboxes[j][-1] > curr_max_p:
                                status[curr_max_index] = SUPPRESSED
                                curr_max_p = bboxes[j][-1]
                                curr_max_index = j
                            else:
                                status[j] = SUPPRESSED

                    status[curr_max_index] = NON_SUPPRESSED

                return [bboxes[i] for i in range(N) if status[i] == NON_SUPPRESSED]

            def _iou(bbox1, bbox2):
                # determine the (x, y)-coordinates of the intersection rectangle
                boxA = bbox1[0] - bbox1[2]/2, bbox1[1] - bbox1[3]/2, bbox1[0] + bbox1[2]/2, bbox1[1] + bbox1[3]/2
                boxB = bbox2[0] - bbox2[2]/2, bbox2[1] - bbox2[3]/2, bbox2[0] + bbox2[2]/2, bbox2[1] + bbox2[3]/2

                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])

                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
                boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

                ret = interArea / float(boxAArea + boxBArea - interArea)

                return ret

            def _correct(frame, bboxes):
                N = len(bboxes)
                ret = []

                img_h, img_w, _ = frame.shape
                for i in range(N):
                    x, y, w, h, p = bboxes[i]

                    MARGIN = 0.5
                    # Add margin
                    xmin = int(max(0, x - w/2 - MARGIN*w))
                    xmax = int(min(img_w, x + w/2 + MARGIN*w))
                    ymin = int(max(0, y - h/2 - MARGIN*h))
                    ymax = int(min(img_h, y + h/2 + MARGIN*h))

                    face = frame[ymin:ymax, xmin:xmax, :]
                    x, y, w, h = self.face_corrector.predict(face)

                    ret.append((x + xmin, y + ymin, w, h, p))

                return ret

            input_img = cv2.resize(im.copy(), (288, 288)) / 255.
            input_img = np.expand_dims(input_img, axis=0)

            pred = self.sess.run([self.prob, self.x_center, self.y_center, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

            # change _absolute_bboxes for more faces
            bboxes = _absolute_bboxes(pred, im, 0.5)
            bboxes = _correct(im, bboxes)
            bboxes = _nonmax_supression(bboxes)

            ret = []
            for x,y,w,h,_ in bboxes:
                ret.append([int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)])
            return np.asarray(ret)


class FaceCorrector(object):
    
    def __init__(self):
        self.load_model('files/face_corrector.pb')

    def load_model(self, corrector_model, from_pb=True):
        import tensorflow as tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            if from_pb:
                with tf.gfile.GFile(corrector_model, "rb") as f:
                    graph_def = tf.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, name="") # If not, name is appended in op name

            else:
                ckpt_path = tf.train.latest_checkpoint(corrector_model)
                saver = tf.train.import_meta_graph('{}.meta'.format(ckpt_path))
                saver.restore(self.sess, ckpt_path)

            self.img = tf.get_default_graph().get_tensor_by_name("img:0")
            self.training = tf.get_default_graph().get_tensor_by_name("training:0")
            self.x = tf.get_default_graph().get_tensor_by_name("X:0")
            self.y = tf.get_default_graph().get_tensor_by_name("Y:0")
            self.w = tf.get_default_graph().get_tensor_by_name("W:0")
            self.h = tf.get_default_graph().get_tensor_by_name("H:0")

    def predict(self, frame):
        # Preprocess
        input_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (50, 50)) / 255.
        input_img = np.reshape(input_img, [1, 50, 50, 3])

        x, y, w, h = self.sess.run([self.x, self.y, self.w, self.h], feed_dict={self.training: False, self.img: input_img})

        img_h, img_w, _ = frame.shape

        x = int(x*img_w)
        w = int(w*img_w)

        y = int(y*img_h)
        h = int(h*img_h)

        return x, y, w, h

class Tools:
    
    def calculateDelaunayTriangles(rect, points):
        def rectContains(rect, point):
            if point[0] < rect[0]:
                return False
            elif point[1] < rect[1]:
                return False
            elif point[0] > rect[0] + rect[2]:
                return False
            elif point[1] > rect[1] + rect[3]:
                return False
            return True
        #create subdiv
        subdiv = cv2.Subdiv2D(rect)

        # Insert points into subdiv
        points = tuple(map(tuple, np.squeeze(points)))
        for p in points:
            subdiv.insert(p)

        triangleList = subdiv.getTriangleList()
        delaunayTri = []
        
        pt = []    
            
        for t in triangleList:       
            pt.append((t[0], t[1]))
            pt.append((t[2], t[3]))
            pt.append((t[4], t[5]))
            
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])        
            
            if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
                ind = []
                #Get face-points (from 68 face detector) by coordinates
                for j in range(0, 3):
                    for k in range(0, len(points)):
                        if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                            ind.append(k)    
                # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph
                if len(ind) == 3:
                    delaunayTri.append((ind[0], ind[1], ind[2]))
            
            pt = []        
        # by now we got a tripple with the index of the coords of the triangle,
        # but we want a rtiple consisting of tuple coords
        return delaunayTri
        

    def warpTriangle(img, t1, t2):
        def applyAffineTransform(src, srcTri, dstTri, size) :
        
            # Given a pair of triangles, find the affine transform.
            warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
            # Apply the Affine Transform just found to the src image
            dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
            return dst

        # Find bounding rectangle for each triangle
        r1 = cv2.boundingRect(np.float32([t1]))
        r2 = cv2.boundingRect(np.float32([t2]))

        # Offset points by left top corner of the respective rectangles
        t1Rect = [] 
        t2Rect = []
        t2RectInt = []

        for i in range(0, 3):
            t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
            t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
            t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


        # Get mask by filling triangle
        mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
        cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

        # Apply warpImage to small rectangular patches
        img1Rect = img[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
        #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
        size = (r2[2], r2[3])

        img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
        img2Rect = img2Rect * mask

        # Copy triangular region of the rectangular patch to the output image
        img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask)
        img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect
