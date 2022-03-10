# Import the required libraries.
from PyQt5 import QtGui
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras import models
from moviepy.editor import *
import sys
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication,QFileDialog,QMainWindow,QAction, QAction,QLineEdit,QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage,QPixmap
from matplotlib import pyplot as plt
import subprocess
import time

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
# CLASSES_LIST = ["0", "1", "2", "3", "4"]


class ImageProc(QMainWindow):
    CLASSES_LIST = ["Membuka Kertas Contekan", "Membuka Smartphone", "Bertukar Kertas Jawaban", "Melihat Kertas Jawaban Teman"]
    # video_file_path = 'Data Baru TA\Hasil Edit\Data Uji\DataUjiBaruBanget(4).mp4'
    model = 'Model/Fix_AlexNetLSTM_32Batch_20ValRatio_1e2_Fixed.h5'
    LRCN_model = tf.keras.models.load_model(model)
    test_videos_directory = 'test_videos_directory'
    filename = 'DataUjiBaruBanget(5)_Baru1'
    path = None
    capture_duration= None
    SEQUENCE_LENGTH = 20
    # output_video_file_path = f'{test_videos_directory}/{filename}-Output-SeqLen{SEQUENCE_LENGTH}.{format(int(time.time()))}.mp4'
    output_video_file_path = None
    # Specify the height and width to which each video frame will be resized in our dataset.
    IMAGE_HEIGHT , IMAGE_WIDTH = 224, 224
    # Specify the number of frames of a video that will be fed to the model as one sequence.

    def __init__(self):
        super(ImageProc,self).__init__()
        loadUi('UI_TA.ui',self)
        self.image=None
        self.inputButton.clicked.connect(self.inputClicked)
        self.processButton.clicked.connect(self.processClicked)
        self.outputButton.clicked.connect(self.outputClicked)
        self.liveButton.clicked.connect(self.liveClicked)

    @pyqtSlot()
    def inputClicked(self):
        self.path = QFileDialog.getOpenFileName(self, 'Open a file', '',
                                        'All Files (*.*)')

    @pyqtSlot()
    def outputClicked(self):
        self.output_video_file_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

    @pyqtSlot()
    def processClicked(self):
        video_reader = cv2.VideoCapture(self.path[0])
        QMessageBox.question(self, 'Message', 'Video Sedang di proses, tekan OK dan mohon tunggu sampai pesan box terbaru muncul ! ' , QMessageBox.Ok, QMessageBox.Ok) 
        # Get the width and height of the video.
        original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialize the VideoWriter Object to store the output video in the disk.
        video_writer = cv2.VideoWriter(f'{self.output_video_file_path}/my_filename_{dt.datetime.now().timestamp()}.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                    video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

        # Declare a queue to store video frames.
        frames_queue = deque(maxlen = self.SEQUENCE_LENGTH)

        # Initialize a variable to store the predicted action being performed in the video.
        predicted_class_name = ''
        predicted_labels_probabilities = ''
        predicted_label_chance = ''

        # Iterate until the video is accessed successfully.
        while video_reader.isOpened():

                # Read the frame.
                ok, frame = video_reader.read() 
                
                # Check if frame is not read properly then break the loop.
                if not ok:
                    break

                # Resize the Frame to fixed Dimensions.
                # resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                resized_frame = cv2.resize(frame, dsize=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
                # normalized_frame = resized_frame / 255

                # Appending the pre-processed frame into the frames list.
                frames_queue.append(resized_frame)

                # Check if the number of frames in the queue are equal to the fixed sequence length.
                if len(frames_queue) == self.SEQUENCE_LENGTH:

                    # Pass the normalized frames to the model and get the predicted probabilities.
                    predicted_labels_probabilities = self.LRCN_model.predict_on_batch(np.expand_dims(frames_queue, axis = 0))[0]
                    predicted_label_chance = predicted_labels_probabilities * 100
                    # Get the index of class with highest probability.
                    predicted_label = np.argmax(predicted_labels_probabilities)

                    # Get the class name using the retrieved index.
                    predicted_class_name = self.CLASSES_LIST[predicted_label]
                    # (startX, startY, endX, endY) = predicted_labels_probabilities
                    # startX = int(startX * self.IMAGE_WIDTH)
                    # startY = int(startY * self.IMAGE_HEIGHT)
                    # endX = int(endX * self.IMAGE_WIDTH)
                    # endY = int(endY * self.IMAGE_HEIGHT)
                    # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    print(predicted_labels_probabilities)
                    print(predicted_label)
                    print(predicted_class_name)
                        
                # Write predicted class name on top of the frame.

                # cv2.rectangle(frame, (startX, startY), (endX, endY),    COLORS[idx], 2)
                #     y = startY - 15 if startY - 15 > 15 else startY + 15
                #     cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame, f"{predicted_labels_probabilities}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"{predicted_label_chance}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # cv2.putText(frame, predicted_labels_probabilities, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Write The frame into the disk using the VideoWriter Object.
                video_writer.write(frame)

            # Release the VideoCapture and VideoWriter objects.
        video_reader.release()
        video_writer.release()
        QMessageBox.question(self, 'Message', 'Saving Done On: ' + self.output_video_file_path, QMessageBox.Ok, QMessageBox.Ok) 

    @pyqtSlot()
    def liveClicked(self):

        self.capture_duration, done1 = QtWidgets.QInputDialog.getInt(
                self, 'Input Time', 'Enter Time when 100 is 20 second record:')
        # The duration in seconds of the video captured
        msg = QMessageBox()
        msg.setWindowTitle("Time Record")
        msg.setText("Live Sedang proses setelah tombol Ok ditekan dan mohon tunggu sampai pesan box terbaru muncul ! ")
        msg.exec_()
        print(self.capture_duration)
        # QMessageBox.question(self, 'Message', 'Live Sedang proses setelah tombol Ok ditekan dan mohon tunggu sampai pesan box terbaru muncul ! ' , QMessageBox.Ok, QMessageBox.Ok) 
        # capture_duration = 100
        if done1:
            video_reader = cv2.VideoCapture(2)

            # Get the width and height of the video.
            original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    # Initialize the VideoWriter Object to store the output video in the disk.
            video_writer = cv2.VideoWriter(f'{self.output_video_file_path}/my_filename_{dt.datetime.now().timestamp()}.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 
                                        video_reader.get(cv2.CAP_PROP_FPS), (original_video_width, original_video_height))

            # Declare a queue to store video frames.
            frames_queue = deque(maxlen = self.SEQUENCE_LENGTH)

            # Initialize a variable to store the predicted action being performed in the video.
            predicted_class_name = ''
            predicted_labels_probabilities = ''
            predicted_label_chance = ''
            start_time = time.time()
            # Iterate until the video is accessed successfully.
            while video_reader.isOpened():
                while( int(time.time() - start_time) < self.capture_duration ):

                    # Read the frame.
                    ok, frame = video_reader.read() 
                    
                    # Check if frame is not read properly then break the loop.
                    if not ok:
                        break
                    
                    cv2.imshow('Frame', frame)
                    # Resize the Frame to fixed Dimensions.
                    # resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
                    resized_frame = cv2.resize(frame, dsize=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH), interpolation=cv2.INTER_CUBIC)
                    # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
                    # normalized_frame = resized_frame / 255

                    # Appending the pre-processed frame into the frames list.
                    frames_queue.append(resized_frame)

                    # Check if the number of frames in the queue are equal to the fixed sequence length.
                    if len(frames_queue) == self.SEQUENCE_LENGTH:

                        # Pass the normalized frames to the model and get the predicted probabilities.
                        predicted_labels_probabilities = self.LRCN_model.predict_on_batch(np.expand_dims(frames_queue, axis = 0))[0]
                        predicted_label_chance = predicted_labels_probabilities * 100
                        # Get the index of class with highest probability.
                        predicted_label = np.argmax(predicted_labels_probabilities)

                        # Get the class name using the retrieved index.
                        predicted_class_name = self.CLASSES_LIST[predicted_label]
                        # (startX, startY, endX, endY) = predicted_labels_probabilities
                        # startX = int(startX * self.IMAGE_WIDTH)
                        # startY = int(startY * self.IMAGE_HEIGHT)
                        # endX = int(endX * self.IMAGE_WIDTH)
                        # endY = int(endY * self.IMAGE_HEIGHT)
                        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        print(predicted_labels_probabilities)
                        print(predicted_label)
                        print(predicted_class_name)
                            
                    # Write predicted class name on top of the frame.

                    # cv2.rectangle(frame, (startX, startY), (endX, endY),    COLORS[idx], 2)
                    #     y = startY - 15 if startY - 15 > 15 else startY + 15
                    #     cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
                    cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # cv2.putText(frame, f"{predicted_labels_probabilities}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"{predicted_label_chance}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    # cv2.putText(frame, predicted_labels_probabilities, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Write The frame into the disk using the VideoWriter Object.
                    video_writer.write(frame)

                # Release the VideoCapture and VideoWriter objects.
                video_reader.release()
                video_writer.release()
                msg.setText("Saving Done ")
                msg.exec_()
                # QMessageBox.question(self, 'Message', 'Saving Done On: ' + self.output_video_file_path, QMessageBox.Ok, QMessageBox.Ok) 


if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    window=ImageProc()
    window.setWindowTitle('Program TA')
    window.show()
    sys.exit(app.exec_())
