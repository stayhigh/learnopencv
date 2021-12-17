# Import required modules
import cv2 as cv
import math
import time
import argparse
import sys
from line_profiler import LineProfiler
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from imutils.video import FPS
import logging

# set logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ENABLE_AGE = True
ENABLE_GENDER = True


##由于textput不能使用汉字，使用此函数添加中文
def cv2AddChineseText(img, text, position, textColor=(0, 255, 0), textSize=30):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text(position, text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input', help='Path to input image or video file. Skip this argument to capture frames from a camera.')
parser.add_argument("--device", default="cpu", help="Device to inference on")

args = parser.parse_args()


args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
#genderList = ['男', '女']

# Load network
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)


if args.device == "cpu":
    ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

    genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    
    faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

    print("Using CPU device")
elif args.device == "gpu":
    ageNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    ageNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    genderNet.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    genderNet.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")


def main():
    # Open a video file or an image file or a camera stream
    cap = cv.VideoCapture(args.input if args.input else 0)
    padding = 20
    frame_num = 0
    fps = FPS().start()
    fps.stop()
    while cv.waitKey(1) < 0 and fps._numFrames <= float("inf"):
        # Read frame
        t = time.time()
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv.waitKey()
            break

        frameFace, bboxes = getFaceBox(faceNet, frame)
        if not bboxes:
            print("No face Detected, Checking next frame")
            cv.imshow("Age Gender Demo", frameFace)
            # topmost
            window_name = 'Age Gender Demo'
            cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            continue

        for bbox in bboxes:
            # print(bbox)
            face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

            blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]
            # print("Gender Output : {}".format(genderPreds))
            print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            print("Age Output : {}".format(agePreds))
            print("Age : {}, conf = {:.3f}".format(age, agePreds[0].max()))

            if not ENABLE_GENDER:
                gender = ""
            if not ENABLE_AGE:
                age = ""
            label = "{},{}".format(gender, age)
            print(label)
            cv.putText(frameFace, label, (bbox[0], bbox[1]-10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv.LINE_AA)
            #cv2AddChineseText(frameFace, label, (bbox[0], bbox[1]-10), (0, 255, 255), 30)

            # cv.imwrite("age-gender-out-{}".format(args.input),frameFace)
        # update the frame
        fps.stop()
        fps.update()

        # FPS info
        logger.info("approx. FPS/elasped_time/#frames: {:.2f}/{:.2f}/{}".format(fps.fps(), fps.elapsed(), fps._numFrames))
        fps_info_xpadding = 20
        fps_info_ypadding = 20
        cv.putText(frameFace,
                    f"FPS/elapsed time: {fps.fps():.4}/{fps.elapsed():.4}",
                    (0+fps_info_xpadding, 0+fps_info_ypadding),
                    fontFace=cv.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(255, 0, 0),
                    thickness=2)
        # topmost display
        cv.imshow("Age Gender Demo", frameFace)
        window_name = 'Age Gender Demo'
        cv.setWindowProperty(window_name, cv.WND_PROP_TOPMOST, 1)
        cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        print("time : {:.3f}".format(time.time() - t))

if __name__ == '__main__':
    # add function to be profiled
    lprofiler = LineProfiler()
    lprofiler.add_function(main)

    # set wrapper
    lp_wrapper = lprofiler(main)
    lp_wrapper()

    # save the output of profiling
    statfile = "{}.lprof".format(sys.argv[0])
    lprofiler.dump_stats(statfile)


     
# cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=~/opencv_gpu -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DOPENCV_ENABLE_NONFREE=ON -DOPENCV_EXTRA_MODULES_PATH=~/cv2_gpu/opencv_contrib/modules -DPYTHON_EXECUTABLE=~/env/bin/python3 -DBUILD_EXAMPLES=ON -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON  -DENABLE_FAST_MATH=ON -DCUDA_FAST_MATH=ON  -DWITH_CUBLAS=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.2 -DOpenCL_LIBRARY=/usr/local/cuda-10.2/lib64/libOpenCL.so -DOpenCL_INCLUDE_DIR=/usr/local/cuda-10.2/include/ ..
