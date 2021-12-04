#!pip install tensorflow==2.4.1 opencv-python mediapipe sklearn matplotlib numpy

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from scipy import stats
import PySimpleGUI as sg
import turtle

sg.theme('Light Green 4')

font = 'Helvetica bold'

layout = [
    [sg.Text('ASL TRANSLATION', size=(100, 2), font=(font, 30), text_color='lightpink', pad=(146,0))],
    [sg.Button("START", size= (100, 2), font=(font, 10))],  
    [sg.Button("Settings", size= (100, 2), font=(font, 10))]
]

# create window
window = sg.Window("Demo", layout, size= (600, 400))



#Detection Function
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Color Conversion
    image.flags.writeable = False                  #Image no longer writeable
    results = model.process(image)                 #Detections using Mediapipe, Frame from CV2
    image.flags.writeable = True                   #Image now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #Color Conversion
    return image, results

#Stylized Landmark Function
def draw_styled_landmarks(image, results):
    #Draw Face Connections
    mp_drawing.draw_landmarks(image, results.face_landmarks,mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(color=(80,110,10),thickness = 1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80,256,121),thickness = 1, circle_radius=1)
                             )
    #Draw Pose Connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,110,10),thickness = 1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121),thickness = 1, circle_radius=1)
                             )
    #Draw L Hand Connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,110,10),thickness = 1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121),thickness = 1, circle_radius=1)
                             )
    #Draw R Hand Connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,110,10),thickness = 1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121),thickness = 1, circle_radius=1)
                             )

#Extracts keypoint values taken from MP landmark function
def extract_keypoints(results):
    pose = np.array([[res.x,res.y,res.z,res.visibility]for res in results.pose_landmarks.landmark]).flatten()if results.pose_landmarks else np.zeros(132)
    lh = np.array([[res.x,res.y,res.z]for res in results.left_hand_landmarks.landmark]).flatten()if results.left_hand_landmarks else np.zeros(63)
    rh = np.array([[res.x,res.y,res.z]for res in results.right_hand_landmarks.landmark]).flatten()if results.right_hand_landmarks else np.zeros(63)
    face = np.array([[res.x,res.y,res.z]for res in results.face_landmarks.landmark]).flatten()if results.face_landmarks else np.zeros(1404)
    return np.concatenate([pose,face,lh,rh,])


        
    return output_frame
#Sets MP to Vars
mp_holistic = mp.solutions.holistic #Holistic Model
mp_drawing = mp.solutions.drawing_utils #Drawing Utilities

actions = np.array(["hello","I Love You","thanks","my", "name","how are" , "you","b", "o", "good"])

DATA_PATH = os.path.join("MP_Data")

#30 Sequences
no_sequences = 30

#30 frames in length
sequence_length = 30

label_map = {label: num for num, label in enumerate(actions)}

        
sequences, labels = [], [] 
for action in actions:
    for sequence in range(no_sequences):
        windows = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            windows.append(res)
        sequences.append(windows)
        labels.append(label_map[action])


while True:
    event = window.read()
    values = window.read()
    # End program if user closes window of presses OK button
    if event == "Settings":
        layout = [
            [sg.Button("Vertices ON", size= (100, 2), font=(font, 10))],
            [sg.Button("Verticies OFF", size= (100, 2), font=(font, 10))],
            [sg.Button("BACK", size= (100, 2), font=(font, 10))]

        ]
        
        # create window
        window = sg.Window("Demo", layout, size= (800, 500))
        event = window.read() 
        values = window.read() 

        if event == 'Verticies ON':
            showVerticies = True
        elif event == 'Verticies OFF':
            showVerticies = False
        elif event == 'BACK':
            window.close()
            sg.theme('Light Green 4')

            background_layout = [(sg.theme_text_color(), sg.theme_background_color()), [sg.Image(r'background.png')]]

            font = 'Helvetica bold'

            layout = [
                [sg.Text('ASL TRANSLATION', size=(100, 2), font=(font, 30), text_color='lightpink', pad=(146,0))],
                [sg.Button("START", size= (100, 2), font=(font, 10))],  
                [sg.Button("HOW TO", size= (100, 2), font=(font, 10))]

            ]

            # create window
            window = sg.Window("Demo", layout, size= (600, 400))



    elif event == "START":
        
        t = turtle.Turtle()

        x = np.array(sequences)

        y = to_categorical(labels).astype(int)

        X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size =0.05)


        log_dir = os.path.join("Logs")
        tb_callback = TensorBoard(log_dir = log_dir)


        model = Sequential()
        model.add(LSTM(64, return_sequences = True, activation = "relu", input_shape = (30, 1662)))
        model.add(LSTM(128, return_sequences = True, activation = "relu"))
        model.add(LSTM(64, return_sequences = False, activation = "relu"))
        model.add(Dense(64, activation = "relu"))
        model.add(Dense(32, activation = "relu"))
        model.add(Dense(actions.shape[0], activation = "softmax"))

        model.load_weights('action.h5')

        res = model.predict(X_test)


        sequence = []
        sentence = []
        predictions = []
        threshold = 0.95

        wn = turtle.Screen()
        wn.setup(300,300)
        wn.bgcolor("White")
        wn.title("Text")
        t.goto(0,0)
        t.hideturtle()

        cap = cv2.VideoCapture(0)
        # Set mediapipe model 
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():

                # Read feed
                ret, frame = cap.read()


                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # 2. Prediction logic
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                sequence = sequence[-30:]

                if len(sequence) == 30:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print(actions[np.argmax(res)])
                    predictions.append(np.argmax(res))


                #3. Viz logic
                    if np.unique(predictions[-10:])[0]==np.argmax(res): 
                        if res[np.argmax(res)] > threshold: 

                            if len(sentence) > 0: 
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                            else:
                                sentence.append(actions[np.argmax(res)])

                    if len(sentence) > 5: 
                        sentence = sentence[-5:]
                #Turtle Text Output
                t.write(' '.join(sentence),font=("Verdana",
                                            15, "normal"))
                if showVerticies ==  True:
                    cv2.imshow('ASL to Text', image)
                elif showVerticies == False:
                    cv2.imshow('ASL to Text', frame)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
    
    
    elif event == sg.WIN_CLOSED:
        break

window.close()