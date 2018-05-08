import face_recognition
import cv2
from src import haw_utils
import threading
import time

frame_of_stream = None
mutex = threading.Lock()

g_face_locations = []
g_face_names = []
mutex1 = threading.Lock()


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
        face_locations = []
        face_encodings = []
        face_names = []
        time.sleep(3)
        global frame_of_stream, mutex
        global g_face_locations, g_face_names, mutex1
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
                    matches = [closest_distances[0][i][0] <= self.distance_threshold for i in range(len(face_locations))]
                    name = 'Unknown'

                    # If a match was found in known_face_encodings, just use the first one.
                    if True in matches:
                        name = self.knn_clf.predict(face_encodings)
                        # face_names.append(name.tolist()[0])
                        for i in range(len(name)):
                            face_names.append(name.tolist()[i])
                    else:
                        face_names.append(name)
                    if mutex1.acquire():
                        g_face_locations = face_locations
                        g_face_names = face_names
                        mutex1.release()


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
        # print('face_loc:', face_locations)
        # print('face_name:', face_names)
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
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

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
