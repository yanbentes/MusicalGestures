import csv
import copy
import hand_processing

import cv2 as cv
import mediapipe as mp

from utils import CvFpsCalc
from model import KeyPointClassifier
from synthesizer import Synthesizer

# Initialize synthesizer
synth = Synthesizer()

active_gestures = []
last_gesture = 'Null'
finger_count = 0
octave = 4

# Argument parsing
args = hand_processing.get_args()
cap_device = args.device
cap_width = args.width
cap_height = args.height

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

use_brect = True

# Model Load
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=use_static_image_mode,
    max_num_hands=2,
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence,
)

keypoint_classifier = KeyPointClassifier()

# Read labels
with open('model/keypoint_classifier/keypoint_classifier_label.csv', encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

# Camera setup
cap = cv.VideoCapture(cap_device)
cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

# FPS
cv_fps_calc = CvFpsCalc(buffer_len=10)

while True:
    fps = cv_fps_calc.get()

    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

    ret, image = cap.read()
    if not ret:
        break
    image = cv.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True

    sharp_zone = 0.33  # Left third of the screen
    flat_zone = 0.66  # Right third of the screen

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == 'Left':
                finger_count = hand_processing.finger_counter(hand_landmarks.landmark, mp_hands)
                octave = finger_count if finger_count else 4

                brect = hand_processing.calc_bounding_rect(debug_image, hand_landmarks)

                landmark_list = hand_processing.calc_landmark_list(debug_image, hand_landmarks)

                debug_image = hand_processing.draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = hand_processing.draw_landmarks(debug_image, landmark_list)
                debug_image = hand_processing.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    '',
                    False
                )
            if handedness.classification[0].label == 'Right':
                # Bounding box calculation
                brect = hand_processing.calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = hand_processing.calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = hand_processing.pre_process_landmark(landmark_list)

                # Calculate the angle between thumb tip and wrist
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                reference = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                thumb_angle = hand_processing.calc_thumb_angle(thumb_tip, reference)

                # Determine sharp, flat, or normal based on thumb angle
                note_modifier = hand_processing.calc_note_modifier(thumb_angle)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                gesture_label = keypoint_classifier_labels[hand_sign_id]

                if gesture_label not in ['Stop', 'Null']:
                    gesture_label += note_modifier

                note_file = synth.get_note_file(octave, gesture_label)
                note = f"{octave}-{note_file.split('-')[-1].replace('.wav', '')}"  # Extract note name

                if gesture_label == 'Stop':
                    for gesture in active_gestures:
                        octave, note = gesture.split('-')
                        synth.note_off(int(octave), note)  # Pass octave and note
                    active_gestures.clear()
                    last_gesture = 'Null'
                elif gesture_label == 'Null':
                    last_gesture = 'Null'
                elif gesture_label != last_gesture:
                    synth.note_on(octave, note_file.split('-')[-1].replace('.wav', ''), velocity=64)
                    active_gestures.append(f"{octave}-{note_file.split('-')[-1].replace('.wav', '')}")
                    last_gesture = gesture_label

                # Drawing part
                debug_image = hand_processing.draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = hand_processing.draw_landmarks(debug_image, landmark_list)
                debug_image = hand_processing.draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    gesture_label,
                    False
                )

    debug_image = hand_processing.draw_info2(debug_image, fps, octave)
    cv.namedWindow('Hand Gesture Recognition', flags=cv.WINDOW_GUI_NORMAL)
    cv.resizeWindow('Hand Gesture Recognition', cap_width, cap_height)
    cv.imshow('Hand Gesture Recognition', debug_image)

synth.stop()
cap.release()
cv.destroyAllWindows()
