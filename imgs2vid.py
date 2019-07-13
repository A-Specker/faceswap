import cv2
import os

image_folder = 'files/vidder'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape


# video = cv2.VideoWriter(video_name, 0, 1, (width, height))
video = cv2.VideoWriter()
codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
fps = 25.0

video.open(video_name, codec, fps, (width//2, height))

for image in images:
    im = cv2.imread(os.path.join(image_folder, image))
    im = im[0:height, 0:width//2]
    # cv2.imshow('image', im)
    # cv2.waitKey(1)
    video.write(im)

cv2.destroyAllWindows()
video.release()
