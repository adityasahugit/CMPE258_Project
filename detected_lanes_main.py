import numpy as np
import cv2
import sys
import os
import argparse
from moviepy.editor import VideoFileClip
from IPython.display import HTML
from keras.models import load_model


# Class to average lanes with
class Lanes():
    def __init__(self):
        self.recent_fit = []
        self.avg_fit = []


def road_lines_fn(image):

    new_img = cv2.resize(image, (160, 80), interpolation=cv2.INTER_CUBIC)

    new_img = np.array(new_img)
    new_img = new_img[None, :, :, :]

    # Make prediction with neural network (un-normalize value by multiplying by 255)
    prediction = model.predict(new_img)[0] * 255

    # Add lane prediction to list for averaging
    lanes.recent_fit.append(prediction)
    # Only using last five for average
    if len(lanes.recent_fit) > 5:
        lanes.recent_fit = lanes.recent_fit[1:]

    # Calculate average detection
    lanes.avg_fit = np.mean(np.array([i for i in lanes.recent_fit]), axis=0)

    # Generate fake R & B color dimensions, stack with G
    blanks = np.zeros_like(lanes.avg_fit).astype(np.uint8)
    lane_drawn = np.dstack((blanks, lanes.avg_fit, blanks))
    # Re-size to match the original image
    lane_image = cv2.resize(lane_drawn, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Merge the lane drawing onto the original image
    result = cv2.addWeighted(image, 0.5, lane_image, 0.5, 0 , image,  cv2.CV_32F)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Lane detection program')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')

    args = parser.parse_args()
    video_path = args.input

    if not os.path.exists(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        sys.exit(1)
    clip1 = VideoFileClip(video_path)

    # Load Keras model
    model = load_model('saved_models/best_model.h5')
    # Create lanes object
    lanes = Lanes()

    # Where to save the output video
    vid_output = 'new_output.mp4'
    # Create the clip
    vid_clip = clip1.fl_image(road_lines_fn)
    vid_clip.write_videofile(vid_output, audio=False)
