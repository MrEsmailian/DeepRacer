import time

import cv2
import numpy as np
import os
from tqdm import tqdm
import easyocr
import psutil


def split_to_frames(video_path, count=0):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    while success:
        cv2.imwrite(f"Frames\\frame{count}.jpg", image)
        success, image = vidcap.read()
        print('\b\b\b\b\b\b-{}-'.format(count), end='')
        count += 1
    return count


def find_image(source, template):
    res = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED)
    cv2.waitKey(0)
    threshold = 0.3
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val >= threshold:
        return max_val, max_loc
    else:
        return 0.0, (-1, -1)


def find_circles(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.blur(gray, (3, 3))
    detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20,
                                        param1=65, param2=58, minRadius=3, maxRadius=60)
    if detected_circles is not None:
        return detected_circles[0, :]
    else:
        return None


def warp_perspective(input_image, angle_degrees):
    height, width = input_image.shape[:2]
    angle_radians = np.deg2rad(angle_degrees)
    d = width * np.sin(angle_radians) / np.cos(angle_radians)
    src_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ])
    dst_points = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1 - d, height - 1],
        [d, height - 1]
    ])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(input_image, matrix, (width, height))
    return warped_image


def load_images_from_folder(folder_path, size=500):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
                images.append([cv2.resize(img, (size, size)), filename])
    return images


def play_video(fs, ls, fps):
    for i in range(len(fs)):
        f = fs[i]
        f = cv2.resize(f, (700, 700))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = .5
        color = (0, 255, 0) if float(ls[i].split('-')[1]) > 0.5 else (0, 0, 255)
        thickness = 1
        f = cv2.putText(f, ls[i], org, font, fontScale, color, thickness, cv2.LINE_AA)
        cv2.imshow('video', f)
        cv2.waitKey(round(1000 / fps))


def get_system_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')

    return {
        'cpu_usage': cpu_usage,
        'memory_used': memory_info.used,
        'memory_percent': memory_info.percent,
        'disk_used': disk_usage.used,
        'disk_percent': disk_usage.percent
    }


def clear_background(pic, x, y, r):
    for i in range(pic.shape[0]):
        for j in range(pic.shape[1]):
            if pow(i - x, 2) + pow(y - j, 2) > pow(r, 2):
                pic[i, j] = 255
    return pic


def get_label(circles, gray, db, shape):
    max_score = 0
    max_command = 'NO_COMMAND_FOUND'
    max_position = (-1, -1)
    if circles is not None:
        for circle in circles:
            y = round(circle[0])
            x = round(circle[1])
            r = round(circle[2] * 1.1)
            crop = gray[x - r:x + r, y - r:y + r]
            no_background = clear_background(crop, round(crop.shape[0] / 2), round(crop.shape[1] / 2), r)
            fit = cv2.resize(no_background, (round(shape * 0.95), round(shape * 0.95)))
            _, binary = cv2.threshold(fit, round(np.quantile(fit, 0.25) + 5), 255, cv2.THRESH_BINARY)
            temp = binary[round(binary.shape[0] * 0.2):binary.shape[0] - round(binary.shape[0] * 0.2),
                   round(binary.shape[1] * 0.2):binary.shape[1] - round(binary.shape[1] * 0.2)]
            for data in db:
                image = data[0]
                command = data[1]
                score, position = find_image(image, temp)
                if score > 0.5 and score > max_score:
                    max_score = score
                    max_command = command
                    max_position = position
    return max_score, max_command, max_position


def detect_signs(frame_count, jump, db, shape, activate_read):
    frames = []
    labels = []
    reader = easyocr.Reader(['en'])
    for f_num in tqdm(range(frame_count), desc="Loading…", ascii=False, ncols=75):
        if f_num % jump == 0:
            frame = cv2.imread(f'Frames\\frame{f_num}.jpg')
            frames.append(frame)
            vertical_perspective = warp_perspective(frame, 22)
            circles = find_circles(vertical_perspective)
            gray = cv2.cvtColor(vertical_perspective, cv2.COLOR_BGR2GRAY)
            if activate_read:
                result = reader.readtext(gray)
                text = ''
                for detection in result:
                    if detection[1].lower() in accepted_terms:
                        text = f'{detection[1]} - 1 - read'
                if len(text) > 1:
                    labels.append(text)
            if len(labels) < len(frames):
                score, command, position = get_label(circles, gray, db, shape)
                labels.append(f'{command} - {score} - {position}')
    return frames, labels


def save_results(labels, path, runtime, count):
    stream = []
    p = 0
    num = 0
    for label in labels:
        splits = label.split('-')
        if splits[0] != 'NO_COMMAND_FOUND ':
            if len(stream) and splits[0] == stream[len(stream) - 1]:
                p += float(splits[1])
                num += 1
            else:
                if len(stream):
                    stream[len(stream) - 1] += str(round(p / num, 4)) + ' - ' + str(num)
                p = float(splits[1])
                num = 1
                stream.append(splits[0])
    if len(stream):
        stream[len(stream) - 1] += str(round(p / num, 4)) + ' - ' + str(num)
    try:
        with open(path, 'w') as file:
            file.write(f'Execution Time: {runtime} s\n'
                       f'Number of Frames: {count}\n'
                       f'Process: {count/ runtime} Fps\n'
                       f'Number of Commands: {len(stream)}\n')
            file.write('\n'.join(stream))
        print(f"Successfully wrote {len(stream)} strings to {stream}")
    except Exception as e:
        print(f"An error occurred: {e}")


accepted_terms = ['40', '50', '60', '70', '80', '90', 'go', 'stop']


if __name__ == '__main__':
    f_count = 0
    f_count = split_to_frames('.\\Test (1).mp4', f_count)
    f_count = split_to_frames('.\\Test (2).mp4', f_count)
    f_count = split_to_frames('.\\Test (3).mp4', f_count)
    f_count = split_to_frames('.\\Test (4).mp4', f_count)
    f_count = split_to_frames('.\\Test (5).mp4', f_count)
    print()
    # f_count = 100
    speed = 1
    print('=' * 60)
    for data_shape in [50, 100, 200, 500]:
        dataset = load_images_from_folder(f'Data\\all\\', data_shape)
        start = time.time()
        frames, labels = detect_signs(f_count, speed, dataset, data_shape, False)
        save_results(labels, f'.\\Results\\{data_shape}.txt', time.time() - start, f_count)
        # play_video(Frames, labels, 33)
        print('=' * 60)

    # data_shape = 500
    # f_count = 10
    # readable = load_images_from_folder('.\\data\\Readable\\')
    # start = time.time()
    # Frames, labels = detect_signs(f_count, speed, readable, data_shape, True)
    # print('I took {} seconds to detect {} Frames with ocr'.format(time.time() - start, f_count))
    # play_video(Frames, labels, 33)
    # print('=' * 60)
