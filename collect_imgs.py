import os
import cv2
from pathlib import Path

cv2.__version__

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 3
dataset_size = 100

# Try different camera indices until a valid one is found
for camera_index in range(10):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        break

if not cap.isOpened():
    print("Error: No available camera found. Exiting...")
    exit()

try:
    for j in range(number_of_classes):
        class_dir = os.path.join(DATA_DIR, str(j))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        print('Collecting data for class {}'.format(j))

        done = False
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Error: Couldn't read frame. Exiting...")
                break

            cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)
            cv2.imshow('frame', frame)
            if cv2.waitKey(25) == ord('q'):
                done = True
                break

        if done:
            counter = 0
            while counter < dataset_size:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Couldn't read frame. Exiting...")
                    break

                cv2.imshow('frame', frame)
                cv2.waitKey(25)
                img_path = os.path.join(class_dir, '{}.jpg'.format(counter))
                cv2.imwrite(img_path, frame)
                print(f'Saved: {img_path}')
                counter += 1

finally:
    cap.release()
    cv2.destroyAllWindows()
