import os
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import json
import cv2
import time
import sys
import calendar
import datetime
from win10toast import ToastNotifier
from HGGS_detector import *
from Vest_detector import *

app = Flask(__name__)

COLORS = [(255, 255, 0), (0, 255, 0), (0, 255, 255)]


def is_wearing_items(person_boxes, item_boxes):
    left_p, top_p, width_p, height_p = person_boxes
    left_i, top_i, width_i, height_i = item_boxes
    # Check if the item's bounding box is entirely within the person's bounding box
    if (
            left_p <= left_i <= left_p + width_p and top_p <= top_i <= top_p + height_p and left_i + width_i <= left_p + width_p and top_i + height_i <= top_p + height_p):
        return True
    else:
        return False


@app.route('/callback', methods=['POST', 'GET'])
def callback():
    return render_template('index.html')


@app.route('/')
def upload_page():
    return render_template('upload.html')


# Main function
@app.route('/ppe_detection', methods=['POST'])
def ppe_detection():
    video_file = request.files['video']
    # Access the callback URL from the form submission
    callback_url = request.form['callback_url']
    is_cuda = request.form.get('cuda', 'Not Cuda')
    video_path = r'video_save\uploaded_video.mp4'
    video_file.save(video_path)

    # Check if the video file and callback URL are provided
    if not video_file:
        return jsonify({'error': 'No video file provided'}), 400

    if not callback_url:
        return jsonify({'error': 'No callback URL provided'}), 400

    # Now you have access to the video file and callback URL, and you can proceed with processing
    # For example, you can save the video file and print the callback URL
    print(f'Callback URL: {callback_url}')
    # # if __name__== "__main__":
    # parameters = request.get_json()
    # print(parameters.get("isCuda"))
    #
    # is_cuda = len(parameters) > 1 and parameters.get("isCuda") == "cuda"

    capture = cv2.VideoCapture(video_path)
    # if __name__== "__main__":
    # parameters = request.get_json()
    # print(parameters.get("isCuda"))

    # is_cuda = len(parameters) > 1 and parameters.get("isCuda") == "cuda"

    HGGS_detector = HGGS_Detector(YOLO_V5, is_cuda)
    # HGGS_detector = YoloDetector(YOLO_V8, is_cuda)

    Vest_detector = Vest_Detector(YOLO_V5, is_cuda)

    classes = ["vest", "shoes", "helmet", "gloves", "glasses"]
    # capture = cv2.VideoCapture(parameters[0])
    # capture = cv2.VideoCapture("input_files/" + parameters.get("input_source"))
    total_frame = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened():
        print("Cannot open video file")
        sys.exit()

    ok, frame = capture.read()
    if not ok:
        print("Cannot read video file")
        sys.exit()

    height, width, layers = frame.shape
    size = (width, height)
    date = datetime.datetime.utcnow()
    utc_time = calendar.timegm(date.utctimetuple())
    output_file_name = "output_" + str(utc_time) + ".mp4"
    output = cv2.VideoWriter(output_file_name, cv2.VideoWriter_fourcc(*'MP4V'), 20, size)
    # toaster = ToastNotifier()
    # Initialize calculating FPS
    # start = time.time_ns()
    frame_count = 0
    frame_cnt = 0
    # fps = -1
    cnt = 0
    sec = 0
    final_json = []
    alert_image = os.getcwd() + os.sep + 'Alert'
    os.makedirs(alert_image, exist_ok=True)
    while True:
        # Read a new frame
        ok, frame = capture.read()
        if not ok:
            break

        if frame is None:
            break

        class_ids, class_names, confidences, boxes = HGGS_detector.apply(frame)
        vest_ids, vest_names, vest_confidences, vest_boxes = Vest_detector.apply(frame)

        frame_count += 1
        frame_cnt += 1
        other_class = {"vest": [], "shoes": [], "helmet": [], "gloves": [], "glasses": []}
        person_class = []
        frame_data = []
        print("Frame count: ", frame_cnt)
        for (class_id, class_name, confidence, box) in zip(vest_ids, vest_names, vest_confidences, vest_boxes):
            color = COLORS[int(class_id) % len(COLORS)]
            if class_name == "vest" and confidence > 0.6:
                other_class['vest'].append(box)
                label = "%s" % "Vest"
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        for (class_id, class_name, confidence, box) in zip(class_ids, class_names, confidences, boxes):
            color = COLORS[int(class_id) % len(COLORS)]
            if class_name == "Person":
                label = "%s" % "Worker"
                person_class.append(box)
            if class_name == "Sneakers" or class_name == "Other Shoes" or class_name == "Leather Shoes" or class_name == "Boots":
                label = "%s" % "Shoes"
                other_class['shoes'].append(box)
            if class_name == "Hat" or class_name == "Helmet":
                label = "%s" % "Helmet"
                other_class['helmet'].append(box)
            if class_name == "Glasses":
                label = "%s" % "Goggles"
                other_class['glasses'].append(box)
            if class_name == "Gloves":
                label = "%s" % "Gloves"
                other_class['gloves'].append(box)

            if class_name == "Person" or class_name == "Sneakers" or class_name == "Other Shoes" or class_name == "Leather Shoes" or class_name == "Boots" or class_name == "Hat" or class_name == "Helmet" or class_name == "Glasses" or class_name == "Gloves":
                cv2.rectangle(frame, box, color, 2)
                cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0))

        if frame_count % 5 == 0:
            cnt += 1
            alert_image_path = alert_image + os.sep + str(cnt) + '.jpg'
            cv2.imwrite(alert_image_path, frame)
            print(frame_cnt)
            detected_items = [k for k, v in other_class.items() if len(v) > 0]
            not_Detected = [i for i in classes if i not in detected_items]
            if not_Detected:
                try:
                    alert_message = 'ALERT'
                    response = requests.post(callback_url, data={'alert_message': alert_message})
                    if response.status_code == 200:
                        # If the response status code is 200, redirect to another URL in the browser
                        # return render_template('index.html', redirect_url='https://example.com/success')
                        return render_template('index.html', redirect_url='http://127.0.0.1:5000/callback')
                    else:
                        return jsonify({'error': 'Failed to send POST request to the target URL'}), 500
                    # response.raise_for_status()
                    # redirect(url_for('index.html', alert_message='Alert'))
                    # response.raise_for_status()
                except requests.exceptions.RequestException as e:
                    return f"Error sending callback: {str(e)}"
                # toaster.show_toast(title='Alert', msg=f"Following elements is not detected:- {', '.join(not_Detected)}", duration=3)
            print(detected_items)
            print(not_Detected)

        result = {}
        person_data = []
        for p_index, person in enumerate(person_class, 1):
            class_set = set()
            person[0] = person[0] - 5
            person[1] = person[1] - 5
            person[2] = person[2] + 10
            person[3] = person[3] + 10
            cv2.rectangle(frame, (person[0] - 10, person[1] - 10), ((person[0] + person[2]), (person[1] + person[3])),
                          (0, 0, 0), 2)
            for key, value in other_class.items():
                for v in value:
                    if is_wearing_items(person, v):
                        class_set.add(key)
            if len(class_set) > 0:
                if p_index in result.keys():
                    result[p_index].append(class_set)
                else:
                    result[p_index] = class_set

            detected = list(class_set)
            not_detected = [i for i in other_class.keys() if i not in class_set]

            person_data.append({"person_id": p_index,
                                "Detected": detected,
                                "Not Detected": not_detected
                                })
        person_key = person_data
        frame_data.append(
            {
                "no": frame_cnt,
                "person": person_key
            }
        )
        final_json.append(frame_data)

        if frame_count >= 30:
            sec += 1
            print("Second: ", sec)
            # end = time.time_ns()

            # fps = 1000000000 * frame_count / (end - start)
            frame_count = 0
            # start = time.time_ns()

        # if fps > 0:
        #     fps_label = "FPS: %.2f" % fps
        #     cv2.putText(frame, fps_label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # cv2.imshow("Frame", frame)
        # output_img = rf"E:\ppe-kit-elements-detection-PPE-detection-demo\output_frame1\frame_{frame_cnt}.jpg"
        # cv2.imwrite(output_img, frame)
        output.write(frame)

        key = cv2.waitKey(30)
        if key == 27:
            break
    print(final_json)
    capture.release()
    output.release()
    cv2.destroyAllWindows()

    return_value = {
        "output_file_name": output_file_name,
        "output_json": final_json
    }
    return return_value


if __name__ == "__main__":
    start = time.time()
    app.run(debug=True)
    end = time.time()
    print(end - start)