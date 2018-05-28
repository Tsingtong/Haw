import face_recognition
import cv2
from src import haw_utils
import threading
import time
from aip import AipFace
import datetime
import pymysql

frame_of_stream = None
mutex = threading.Lock()

g_face_locations = []
g_face_names = []
mutex1 = threading.Lock()


class DbThread(threading.Thread):
    def Update_Table(self, uid):  # 连接mysql 更新学号对应学生的人脸签到记录
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        db = pymysql.connect("123.207.164.55", "pandeku", "pandeku", "django_stu_info", charset='utf8')
        cursor = db.cursor()
        sql = "UPDATE login_register_attend_user SET Is_Attend='%s' WHERE User_Id = '%s'" % ("已签到!", uid)
        #  sql1 = "UPDATE login_register_attend_user SET Attend_Time=CURRENT_TIMESTAMP WHERE User_Id = '%s'" % (uid)
        sql1 = "UPDATE login_register_attend_user SET Attend_Time='%s' WHERE User_Id = '%s'" % (time, uid)
        try:
            cursor.execute(sql)  # 执行更新
            cursor.execute(sql1)  # 执行更新
            db.commit()  # 提交
        except :
            db.rollback()  # 发生错误,回滚
            print("Error: unable to update data")
        db.close()

    def run(self):
        global g_face_names, mutex1
        print('haha')
        db = pymysql.connect("123.207.164.55", "pandeku", "pandeku", "django_stu_info", charset='utf8')
        cursor = db.cursor()
        while True:
            times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if mutex1.acquire():
                loc_uid = g_face_names
                mutex1.release()
            print('get uid:', loc_uid)
            # # Write to db
            # if loc_uid:
            #     for uid in loc_uid:
            #         sql = "UPDATE login_register_attend_user SET Is_Attend='%s' WHERE User_Id = '%s'" % ("已签到!", uid)
            #         #  sql1 = "UPDATE login_register_attend_user SET Attend_Time=CURRENT_TIMESTAMP WHERE User_Id = '%s'" % (uid)
            #         sql1 = "UPDATE login_register_attend_user SET Attend_Time='%s' WHERE User_Id = '%s'" % (times, uid)
            #         try:
            #             cursor.execute(sql)  # 执行更新
            #             cursor.execute(sql1)  # 执行更新
            #             db.commit()  # 提交
            #             print('operation!')
            #         except:
            #             db.rollback()  # 发生错误,回滚
            #             print("Error: unable to update data")
            #         print('Done DB operation!')
            # else:
            #     continue
        db.close()


class RtspStreamThread(threading.Thread):
    def run(self):
        # video_capture = cv2.VideoCapture("rtsp://admin:admin12345@192.168.69.237:554/Streaming/Channels/1")
        video_capture = cv2.VideoCapture(0)
        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()
            global frame_of_stream, mutex
            if mutex.acquire():
                frame_of_stream = frame
                mutex.release()


class ProcessThread(threading.Thread):
    def __init__(self, model_path, distance_threshold):
        super().__init__()
        self.knn_clf = haw_utils.load_trained_model(model_path)
        self.distance_threshold = distance_threshold

    def run(self):
        # Initialize some variables
        db = pymysql.connect("123.207.164.55", "pandeku", "pandeku", "django_stu_info", charset='utf8')
        cursor = db.cursor()
        face_locations = []
        face_encodings = []
        face_names = []
        time.sleep(3)
        global frame_of_stream, mutex
        global g_face_locations, g_face_names, mutex1
        global g_stu_uid, mutex2
        APP_ID = '11155090'
        API_KEY = "rG3P2789bywKHfGN0OnEwAgg"
        SECRET_KEY = "yTjcZQ92dwGBl3zDY1yWzEz9fb4FeYMc"

        client = AipFace(APP_ID, API_KEY, SECRET_KEY)
        options = {
            "ext_fields": "faceliveness",
            "detect_top_num": 10,  # 检测多少个人脸进行比对，默认值1（最对返回10个）
            "user_top_num": 1  # 返回识别结果top人数”当同一个人有多张图片时，只返回比对最高的1个分数
        }
        Group_Id = "test"
        while True:
            if mutex.acquire():
                loc_frame = frame_of_stream
                mutex.release()
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(loc_frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []

                if len(face_locations) == 0:
                    if mutex1.acquire():
                        g_face_locations = []
                        g_face_names = []
                        mutex1.release()
                    continue
                else:
                    # Use the KNN model to find the best matches for the test face
                    closest_distances = self.knn_clf.kneighbors(face_encodings, n_neighbors=1)
                    matches = [closest_distances[0][i][0] <= self.distance_threshold for i in
                               range(len(face_locations))]
                    print(g_face_locations)
                    # Use Baidu API to find the best matches for the test face
                    cv2.imwrite('face.jpg', loc_frame)
                    with open("face.jpg", 'rb') as fp:
                        image = fp.read()
                    Res = client.multiIdentify(Group_Id, image, options)
                    # pandeku de daima
                    if 'error_msg' in Res:
                        print(Res['error_msg'])
                    else:
                        for i in range(Res['result_num']):
                            uid = Res['result'][i]['uid']
                            info = str(Res['result'][i]['user_info']) + "," + str(Res['result'][i]['uid']) + "," + str(
                                Res['result'][i]['group_id']) + "," + str(
                                Res['result'][i]['scores'])
                            with open("right_result_time_ues.txt", 'a') as f:
                                f.write(info+"\n")
                    # pandeku de daima
                    try:
                        Res['result']
                    except KeyError:
                        continue
                    else:
                        for i in range(len(Res['result'])):
                            if Res['result'][i]['scores'][0] > 70:
                                res = Res['result'][i]['uid']
                                face_names.append(res)
                            else:
                                face_names.append('0')
                        print('res:', res)
                        if mutex1.acquire():
                            g_face_locations = face_locations
                            g_face_names = face_names
                            mutex1.release()
                        for uid in face_names:
                            times = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            sql = "UPDATE login_register_attend_user SET Is_Attend='%s' WHERE User_Id = '%s'" % ("已签到!", uid)
                            #  sql1 = "UPDATE login_register_attend_user SET Attend_Time=CURRENT_TIMESTAMP WHERE User_Id = '%s'" % (uid)
                            sql1 = "UPDATE login_register_attend_user SET Attend_Time='%s' WHERE User_Id = '%s'" % (times, uid)
                            try:
                                cursor.execute(sql)  # 执行更新
                                cursor.execute(sql1)  # 执行更新
                                db.commit()  # 提交
                            except:
                                db.rollback()  # 发生错误,回滚
                                print("Error: unable to update data")
                            print('DB Operation Done!')


def display_video():
    # Get a reference to webcam #0 (the default one)
    # video_capture = cv2.VideoCapture("rtsp://admin:admin12345@192.168.69.237:554/Streaming/Channels/1")
    video_capture = cv2.VideoCapture(0)
    global g_face_locations, g_face_names, mutex1
    while True:
        if mutex1.acquire():
            face_locations = g_face_locations
            face_names = g_face_names
            mutex1.release()
        # Grab a single frame of video
        ret, frame = video_capture.read()
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (255, 0, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, 'Student', (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)
            cv2.putText(frame, 'Identified Students:', (30, 30), font, 1, (0, 0, 255), 2)
            for i in range(len(face_names)):
                cv2.putText(frame, str(face_names[i]), (30, 60+i*30), font, 1, (255, 255, 255), 1)

        # Draw Info
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, 'Student Attendance System', (990, 30), font, 0.625, (255, 255, 255), 1)
        cv2.putText(frame, 'Alpha: v1.0.0 (Haw)', (990, 60), font, 0.625, (255, 255, 255), 1)
        cv2.putText(frame, 'Positioning Mode : HOG', (990, 90), font, 0.625, (255, 255, 255), 1)
        cv2.putText(frame, '@author:liuqingtong', (540, 700), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, '@email:1504030521@st.btbu.edu.cn', (480, 715), font, 0.5, (255, 255, 255), 1)
        # Display the resulting image
        # Display the resulting image
        cv2.imshow('Student Attendance System', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    rstp_thread = RtspStreamThread()
    frame_process_thread = ProcessThread(model_path='/Users/liuqingtong/Desktop/PycharmProjects/Haw/Haw/DB/', distance_threshold=0.6)
    rstp_thread.start()
    frame_process_thread.start()
    display_video()
    # db_thread = DbThread()
    # db_thread.start()
