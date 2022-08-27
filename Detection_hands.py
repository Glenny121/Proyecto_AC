# importación de librerías necesarias
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees

def palm_centroid(coordinates_list):
    coordinates = np.array(coordinates_list)
    centroid = np.mean(coordinates, axis=0)
    centroid = int(centroid[0]), int(centroid[1])
    return centroid

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
# captura de video, el 0 indica el puerto para la cámara 
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Pulgar
thumb_points = [0, 2, 3]

# arrays para el indice, medio, anular y meñique
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points = [6, 10, 14, 18]

with mp_hands.Hands(
    model_complexity=1,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    # muestra datos en tiempo real todo el tiempo
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        height, width, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        fingers_counter = "_"

        if results.multi_hand_landmarks:
            coordinates_thumb = []
            coordinates_palm = []
            coordinates_ft = []
            coordinates_fb = []
            #extrae coordenadas de los dedos
            for hand_landmarks in results.multi_hand_landmarks:
                for index in thumb_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_thumb.append([x, y])

                for index in palm_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_palm.append([x, y])

                for index in fingertips_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_ft.append([x, y])

                for index in finger_base_points:
                    x = int(hand_landmarks.landmark[index].x * width)
                    y = int(hand_landmarks.landmark[index].y * height)
                    coordinates_fb.append([x, y])
                ########################
                # Pulgar
                # se forma un triángulo y se extraen sus coordenadas
                p1 = np.array(coordinates_thumb[0])
                p2 = np.array(coordinates_thumb[1])
                p3 = np.array(coordinates_thumb[2])
                # se toman las distancias
                l1 = np.linalg.norm(p2 - p3)
                l2 = np.linalg.norm(p1 - p3)
                l3 = np.linalg.norm(p1 - p2)

                # Calcular el angulo 
                angle = degrees(acos((l1**2 + l3**2 - l2**2) / (2 * l1 * l3)))
                thumb_finger = np.array(False)
                if angle > 150:
                    thumb_finger = np.array(True)
                #############################
                # de indice, medio, anular y meñique
                nx, ny = palm_centroid(coordinates_palm)
                cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
                coordinates_centroid = np.array([nx, ny])
                coordinates_ft = np.array(coordinates_ft)
                coordinates_fb = np.array(coordinates_fb)

                # Distancias
                
                d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft,  axis=1)
                d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb,  axis=1)
                dif = d_centrid_ft - d_centrid_fb
                fingers = np.append(angle - 140, dif)
                print("Dedos:",fingers)
                # si es mayor a 0, se cuenta como estirado
                fingers1 = dif > 0
                fingers1 = np.append(thumb_finger,fingers1)
                #se muestra la cantidad de("Dedos:",fingers1)
                fingers_counter = str(np.count_nonzero(fingers1==True))
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        ##############
        # Visualizacion de cuántos dedos están estirados
        cv2.rectangle(frame, (0, 0), (88, 80), (125, 220, 0), -1)
        cv2.putText(frame, fingers_counter, (15, 65), 1, 5, (255, 255, 255), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0XFF == 27:
            break
cap.release()
cv2.destroyAllWindows()
