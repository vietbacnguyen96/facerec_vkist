from flask import Flask, render_template, Response, jsonify
import cv2
import requests
import base64
import json
import time
import threading
import os
from datetime import datetime
import time
from threading import Thread


from utils.service.TFLiteFaceAlignment import *
from utils.service.TFLiteFaceDetector import *
from utils.functions import *

app = Flask(__name__)

fd_0 = UltraLightFaceDetecion(
    "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fd_1 = UltraLightFaceDetecion(
    "utils/service/weights/RFB-320.tflite", conf_threshold=0.98)
fa = CoordinateAlignmentModel("utils/service/weights/coor_2d106.tflite")

url = 'http://123.16.55.212:85/'

# IP cam thang
# rtsp_stream_link = "rtsp://admin:vkist123@192.168.2.130:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"
rtsp_stream_link = "rtsp://admin:vkist123@123.16.55.212:61055/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"

# IP cam tru
# rtsp_stream_link = "rtsp://admin:vkist123@192.168.2.131:554/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"
# rtsp_stream_link = "rtsp://admin:vkist123@123.16.55.212:61056/Streaming/Channels/101?transportmode=unicast&profile=Profile_1"

# rtsp_stream_link = "Video_2023-02-24_135459_Hoa.wmv"

api_list = [url + 'facerec', url + 'FaceRec_DREAM',
            url + 'FaceRec_3DFaceModeling', url + 'check_pickup']
request_times = [20, 10, 10]
api_index = 0
extend_pixel = 100
crop_image_size = 100

# vkist_3 123456789
secret_key = "bb04d080-372b-45d6-9111-4e5ed99a15ea"

predict_labels = []

# video_dst_dir = 'videos/'
# if not os.path.exists(video_dst_dir):
#     os.makedirs(video_dst_dir)

# record_time = datetime.fromtimestamp(time.time())
# year = '20' + record_time.strftime('%y')
# month = record_time.strftime('%m')
# date = record_time.strftime('%d')
# record_time = str(record_time).replace(' ', '_').replace(':', '_')


class VideoScreenshot(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Take screenshot every x seconds
        self.screenshot_interval = 1

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        self.size = (self.frame_width, self.frame_height)
        # self.record_screen = cv2.VideoWriter(video_dst_dir + 'record_' + record_time + '.avi', cv2.VideoWriter_fourcc(*'MJPG'),10, self.size)

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

        self.queue = []
        self.count = 0
        self.temp_resized_boxes = []
        self.temp_resized_marks = []

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
                # self.frame = cv2.resize(self.frame, (self.frame_width, self.frame_height))

    def show_frame(self):
        # Display frames in main program
        if self.status:
            self.temp_resized_boxes, _ = fd_0.inference(self.frame)
            # if len(temp_resized_boxes) > 0:
            #     print('Found {} face(s)'.format(len(temp_resized_boxes)))
            #     # Saving the image
            #     for bbI in temp_resized_boxes:

            #         cv2.imwrite(output_folder + str(time.time()) + '.jpg', self.frame[int(bbI[1]):int(bbI[3]), int(bbI[0]):int(bbI[2])])
            #     # Saving the video
            #     self.record_screen.write(self.frame)

            self.temp_resized_marks = fa.get_landmarks(
                self.frame, self.temp_resized_boxes)
            # Draw landmarks of each face
            # for bbox_I, landmark_I in zip(self.temp_resized_boxes, self.temp_resized_marks):
            #     # landmark_I = landmark_I * (1 / scale_ratio)
            #     draw_landmark(self.frame, landmark_I, color=(125, 255, 125))

            for bbox_I, landmark_I in zip(self.temp_resized_boxes, self.temp_resized_marks):
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(
                    bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= int(extend_pixel)
                xmax += int(extend_pixel)
                ymin -= int(extend_pixel)
                ymax += int(extend_pixel)

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                xmax = self.frame_width if xmax >= self.frame_width else xmax
                ymax = self.frame_height if ymax >= self.frame_height else ymax

                resized_face_I = self.frame[ymin:ymax, xmin:xmax]
                rotated_resized_face_I = align_face(
                    resized_face_I, landmark_I[34], landmark_I[88])

                # Show rotated resized face image
                # cv2.imshow('Rotated resized face image', rotated_resized_face_I)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                self.queue = [t for t in self.queue if t.is_alive()]
                if len(self.queue) < 3:
                    # cv2.imwrite('rotated_faces/' + str(time.time()) + '.jpg', rotated_resized_face_I)
                    self.queue.append(threading.Thread(
                        target=face_recognize, args=(rotated_resized_face_I,)))
                    self.queue[-1].start()
            draw_box(self.frame, self.temp_resized_boxes,
                     color=(125, 255, 125))
            return self.frame

    def save_frame(self):
        # Save obtained frame periodically
        self.frame_count = 0

        def save_frame_thread():
            while True:
                try:
                    cv2.imwrite('frame_{}.png'.format(
                        self.frame_count), self.frame)
                    self.frame_count += 1
                    time.sleep(self.screenshot_interval)
                except AttributeError:
                    pass
        Thread(target=save_frame_thread, args=()).start()


def face_recognize(frame):
    global api_index

    _, encimg = cv2.imencode(".jpg", frame)
    img_byte = encimg.tobytes()
    img_str = base64.b64encode(img_byte).decode('utf-8')
    new_img_str = "data:image/jpeg;base64," + img_str
    headers = {'Content-type': 'application/json',
               'Accept': 'text/plain', 'charset': 'utf-8'}

    payload = json.dumps(
        {"secret_key": secret_key, "img": new_img_str, 'local_register': False})

    response = requests.post(
        api_list[api_index], data=payload, headers=headers, timeout=100)

    try:
        for id, name, profileID, timestamp in zip(
            response.json()['result']['id'],
            response.json()['result']['identities'],
            response.json()['result']['profilefaceIDs'],
            response.json()['result']['timelines']
        ):
            print('Server response', response.json()['result']['identities'])
            if id != -1:
                cur_profile_face = None

                if profileID is not None:
                    cur_url = url + 'images/' + secret_key + '/' + profileID
                    cur_profile_face = np.array(Image.open(
                        requests.get(cur_url, stream=True).raw))
                    cur_profile_face = cv2.cvtColor(
                        cur_profile_face, cv2.COLOR_BGR2RGB)

                    _, encimg = cv2.imencode(".jpg", cur_profile_face)
                    img_byte = encimg.tobytes()
                    img_str = base64.b64encode(img_byte).decode('utf-8')
                    cur_profile_face = "data:image/jpeg;base64," + img_str

                frame = cv2.resize(frame, (crop_image_size, crop_image_size))
                _, encimg = cv2.imencode(".jpg", frame)
                img_byte = encimg.tobytes()
                img_str = base64.b64encode(img_byte).decode('utf-8')
                new_img_str = "data:image/jpeg;base64," + img_str

                predict_labels.append(
                    [id, name, new_img_str, cur_profile_face, timestamp])

    except requests.exceptions.RequestException:
        print(response.text)


def get_webcam_frame():
    # Open the webcam stream
    webcam_0 = cv2.VideoCapture(0)
    # if not webcam_0.isOpened():
    #     return True
    frame_width = int(webcam_0.get(3))
    frame_height = int(webcam_0.get(4))

    prev_frame_time = 0
    new_frame_time = 0
    queue = []
    count = 0

    while True:
        count += 1
        # Read a frame from the stream
        ret, orig_image = webcam_0.read()
        # orig_image = cv2.flip(orig_image, 1)

        final_frame = orig_image.copy()
        scale_ratio = 1/1
        new_height, new_width = int(
            frame_height * scale_ratio), int(frame_width * scale_ratio)

        resized_image = cv2.resize(
            orig_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        temp_resized_boxes, _ = fd_0.inference(resized_image)

        temp_boxes = temp_resized_boxes * (1 / scale_ratio)

        # Draw boundary boxes around faces
        draw_box(final_frame, temp_boxes, color=(125, 255, 125))

        # Find landmarks of each face
        temp_resized_marks = fa.get_landmarks(
            resized_image, temp_resized_boxes)

        # # Draw landmarks of each face
        # for bbox_I, landmark_I in zip(temp_boxes, temp_resized_marks):
        #     landmark_I = landmark_I * (1 / scale_ratio)
        #     draw_landmark(final_frame, landmark_I, color=(125, 255, 125))

        #     # Show rotated raw face image
        #     xmin, ymin, xmax, ymax = int(bbox_I[0]), int(bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])
        #     xmin -= extend_pixel
        #     xmax += extend_pixel
        #     ymin -= 2 * extend_pixel
        #     ymax += extend_pixel

        #     xmin = 0 if xmin < 0 else xmin
        #     ymin = 0 if ymin < 0 else ymin
        #     xmax = frame_width if xmax >= frame_width else xmax
        #     ymax = frame_height if ymax >= frame_height else ymax

        #     face_I = orig_image[ymin:ymax, xmin:xmax]
        #     face_I = align_face(face_I, landmark_I[34], landmark_I[88])

        #     cv2.imshow('Rotated raw face image', face_I)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

        if (count % request_times[api_index]) == 0:
            for bbox_I, landmark_I in zip(temp_resized_boxes, temp_resized_marks):
                landmark_I = landmark_I * (1 / scale_ratio)
                xmin, ymin, xmax, ymax = int(bbox_I[0]), int(
                    bbox_I[1]), int(bbox_I[2]), int(bbox_I[3])

                xmin -= int(extend_pixel * scale_ratio)
                xmax += int(extend_pixel * scale_ratio)
                ymin -= int(extend_pixel * scale_ratio)
                ymax += int(extend_pixel * scale_ratio)

                xmin = 0 if xmin < 0 else xmin
                ymin = 0 if ymin < 0 else ymin
                # xmax = frame_width if xmax >= frame_width else xmax
                # ymax = frame_height if ymax >= frame_height else ymax
                xmax = new_width if xmax >= new_width else xmax
                ymax = new_height if ymax >= new_height else ymax

                resized_face_I = resized_image[ymin:ymax, xmin:xmax]
                rotated_resized_face_I = align_face(
                    resized_face_I, landmark_I[34], landmark_I[88])

                # Show rotated resized face image
                # cv2.imshow('Rotated resized face image', rotated_resized_face_I)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                count = 0
                queue = [t for t in queue if t.is_alive()]
                if len(queue) < 3:
                    # cv2.imwrite('rotated_faces/' + str(time.time()) + '.jpg', rotated_resized_face_I)
                    queue.append(threading.Thread(
                        target=face_recognize, args=(rotated_resized_face_I,)))
                    queue[-1].start()

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps = str(int(fps))

        cv2.putText(final_frame, '{0} fps'.format(
            fps), (20, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (100, 255, 0), 1, cv2.LINE_AA)

        # Convert the frame to a jpeg image
        ret, jpeg = cv2.imencode('.jpg', final_frame)

        # Return the image as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

def get_ip_frame():

    video_stream_widget = VideoScreenshot(rtsp_stream_link)
    while True:
        try:
            video_stream_widget.count += 1
            if (video_stream_widget.count % 25) == 0:
                video_stream_widget.count = 0
                frame = video_stream_widget.show_frame()
                # Convert the frame to a jpeg image
                ret, jpeg = cv2.imencode('.jpg', frame)
                # Return the image as bytes
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
        except AttributeError:
            pass

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/webcam')
def webcam():
    return Response(get_webcam_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/ip_cam')
def ip_cam():
    return Response(get_ip_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/data')
def data():
    global predict_labels
    if len(predict_labels) > 3:
        predict_labels = predict_labels[-3:]
    newest_data = list(reversed(predict_labels))
    return jsonify({'info': newest_data})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    # app.run(debug=True, host='127.0.0.1')
