import cv2
import os
from os import listdir
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import face_recognition
from PIL import Image, ImageDraw
import warnings
import Model.Person as mp
import DatabaseQueries.db as db
import datetime
import pandas as pd
import psycopg2
from dotenv import load_dotenv
import numpy as np


load_dotenv()

app = FastAPI()
networkip = '192.168.29.244'
networkport = 8000
# 'rtsp://192.168.29.244:8080/h264_ulaw.sdp'
# cmaeraip = 'rtsp://192.168.29.244:8000/h264_ulaw.sdp'
# cmaeraip1 = 'rtsp://192.168.29.244:8080/h264_ulaw.sdp'
# cmaeraip2 = 'rtsp://192.168.29.244/h264_ulaw.sdp'


templates = Jinja2Templates(directory="templates")


@app.get('/')
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# ---------------------------Camera-----------------------------------------


def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            cv2.imwrite('test_images/new.jpg', frame)
            test_image = face_recognition.load_image_file('test_images/new.jpg')
            face_locations = face_recognition.face_locations(test_image, model='cnn')
            face_encodings = face_recognition.face_encodings(test_image, face_locations)
            face_names = []
            pil_image = Image.fromarray(test_image)
            draw = ImageDraw.Draw(pil_image)
            warnings.filterwarnings("ignore")
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
                name = "Unknown Person"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = know_face_names[best_match_index]

                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = know_face_names[first_match_index]
                    current_time = datetime.datetime.now()
                    time = str(current_time.hour) + ':' + str(current_time.minute) + ':' + str(current_time.second)
                    day = str(current_time.day) + '/' + str(current_time.month) + '/' + str(current_time.year)
                    res = person_object.person_details(status=status, person=mp.Person(name=name, time=time, day=day),
                                                       check=0)
                    if res == 0:
                        if status == 'CHECK_IN':
                            person_object.add(status=status, person=mp.Person(name=name, time=time, day=day))
                            # person_object.add(status='CHECK_OUT', person=mp.Person(name=name, time='0', day=day))
                        else:
                            if status == 'CHECK_OUT':
                                res = person_object.person_details(status=status,
                                                                   person=mp.Person(name=name, time=time, day=day),
                                                                   check=1)
                                if res == "0":
                                    person_object.update(status=status, person=mp.Person(name=name, time=time, day=day))
                face_names.append(name)
                draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
                text_width, text_height = draw.textsize(name)
                draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 0), outline=(0, 0, 0))
                draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
            del draw

            rgb_im = pil_image.convert('RGB')
            rgb_im.save('rgb.jpg')
            img = cv2.imread('rgb.jpg')
            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.get('/video_feed')
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get('/get_data')
def get_data():
    cursor.execute(f'''
                SELECT * FROM CHECK_IN
                    ''')
    name = []
    time = []
    day = []

    for row in cursor.fetchall():
        name.append(row[0])
        time.append(row[1])
        day.append(row[2])
    conn.commit()

    cursor.execute(f'''
                    SELECT * FROM CHECK_OUT
                        ''')
    timeout = []
    for row in cursor.fetchall():
        timeout.append(row[1])
    conn.commit()
    d = {'Name': name, 'Day': day, 'Time_In': time, 'Time_Out': timeout}
    print(d)
    df = pd.DataFrame(data=d)
    current_time = datetime.datetime.now()
    time = str(current_time.hour) + str(current_time.minute) + str(current_time.second)
    day = str(current_time.day) + str(current_time.month) + str(current_time.year)
    df.to_csv(f'{time}{day}.csv')
    return "File Saved Successfully..."


if __name__ == '__main__':
    DATABASE_USER = os.environ['DATABASE_USER']
    DATABASE_PASSWORD = os.environ['DATABASE_PASSWORD']
    DATABASE_HOST = os.environ['DATABASE_HOST']
    DATABASE_PORT = os.environ['DATABASE_PORT']
    DATABASE_NAME = os.environ['DATABASE_NAME']

    # print(DATABASE_USER)
    # print(DATABASE_PASSWORD)
    # print(DATABASE_HOST)
    # print(DATABASE_PORT)
    # print(DATABASE_NAME)
    conn = psycopg2.connect(user=DATABASE_USER, password=DATABASE_PASSWORD, host=DATABASE_HOST, port=DATABASE_PORT, database=DATABASE_NAME)

    cursor = conn.cursor()
    person_object = db.Persondb(cursor, conn)
    known_face_encodings = []
    face_names = []
    know_face_names = []


    def loadImages(path):
        imagesList = listdir(path)
        for img in imagesList:
            image = face_recognition.load_image_file(path + img)
            image_encoding = face_recognition.face_encodings(image, num_jitters=100)[0]
            known_face_encodings.append(image_encoding)
            know_face_names.append(img.split('.')[0])


    path = "./images/"

    loadImages(path)
    # CHECK_IN
    # CHECK_OUT
    status = 'CHECK_IN'

    uvicorn.run(app, host=networkip, port=networkport)
