#!/usr/bin/env python
import cv2

video_path = 'data/test_video/IMG_3571_converted.mp4'
print(f'Testing video: {video_path}')

cap = cv2.VideoCapture(video_path)
print(f'Opened: {cap.isOpened()}')
print(f'FPS: {cap.get(cv2.CAP_PROP_FPS)}')
print(f'Frame count: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}')

ret, frame = cap.read()
print(f'Read first frame: {ret}')
if ret:
    print(f'Frame shape: {frame.shape}')

cap.release()
print('Success!')
