import cv2

video_capture = cv2.VideoCapture('green_vid.MXF')
success, image = video_capture.read()
count = 0
while success:
    cv2.imwrite("suhyun_frame%d.jpg" % count, image)  # save frame as JPEG file
    success, image = video_capture.read()
    print('Read a new frame: ', success)
    count += 1
