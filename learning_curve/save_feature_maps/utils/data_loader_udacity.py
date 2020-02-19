import numpy as np
from collections import defaultdict
import os
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torch as th
import utils
from collections import defaultdict
import math
from sklearn.model_selection import train_test_split

class udacity(data.Dataset):

    def __init__(self, dir, load_percent, phase, transform=None, target_transform=None):

        samples, class_vector = make_dataset(root=dir)

        samples_train, samples_test, \
            class_vector_train, class_vector_test = train_test_split(
                samples, class_vector, test_size=0.20, 
                random_state=123, stratify=class_vector
        )

        samples_train, samples_val, \
            class_vector_train, class_vector_val = train_test_split(
                samples_train, class_vector_train, test_size=0.20, 
                random_state=123, stratify=class_vector_train
        )

        if load_percent < 1.0:
            # sizing the samples_train and class_vector_train
            samples_train, _ , class_vector_train, _  = train_test_split(
                    samples_train, class_vector_train, 
                    test_size=1 - load_percent, random_state=123, 
                    stratify = class_vector_train
            )
        
        if phase == 'train':
            self.samples = samples_train
        elif phase == 'val':
            self.samples = samples_val
        elif phase == 'test':
            self.samples = samples_test

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_name, target = self.samples[index]
        # print(target)
        image = self.pil_loader(img_name)

        if self.transform is not None:
            for trans in self.transform:
                if type(trans) == utils.util_funcs.RandomHorizontalFlip:
                    image, target = trans(image, target)
                else:
                    image = trans(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {'img': image, 'target': target.astype('float32')}


    def pil_loader(self,path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

def read_steerings(steering_log, time_scale):
    steerings = defaultdict(list)
    speeds = defaultdict(list)
    with open(steering_log) as f:
        for line in f.readlines()[1:]:
            fields = line.split(",")
            nanosecond, angle, speed = int(fields[0]), float(fields[1]), float(fields[3])
            timestamp = int(nanosecond / time_scale)
            steerings[timestamp].append(angle)
            speeds[timestamp].append(speed)

    return steerings, speeds

def read_image_stamps(image_log, camera, time_scale):
    timestamps = defaultdict(list)
    with open(image_log) as f:
        for line in f.readlines()[1:]:
            if camera not in line:
                continue
            fields = line.split(",")
            nanosecond = int(fields[0])
            timestamp = int(nanosecond / time_scale)
            timestamps[timestamp].append(nanosecond)
    return timestamps

def camera_adjust(angle, speed, camera):

    # Left camera -20 inches, right camera +20 inches (x-direction)
    # Steering should be correction + current steering for center camera

    # Chose a constant speed
    speed = 10.0  # Speed

    # Reaction time - Time to return to center
    # The literature seems to prefer 2.0s (probably really depends on speed)
    if speed < 1.0:
        reaction_time = 0
        angle = angle
    else:
        reaction_time = 2.0 # Seconds

        # Trig to find angle to steer to get to center of lane in 2s
        opposite = 20.0 # inches
        adjacent = speed*reaction_time*12.0 # inches (ft/s)*s*(12 in/ft) = inches (y-direction)
        angle_adj = np.arctan(float(opposite)/adjacent) # radians
        
        # Adjust based on camera being used and steering angle for center camera
        if camera == 'left':
            angle_adj = -angle_adj
        angle = angle_adj + angle

    return angle

def make_dataset(root, fps=10):

    root = os.path.join(root, 'CH2_002/output/')

    # setup
    steering_log = os.path.join(root, 'steering.csv')
    image_log = os.path.join(root, 'camera.csv')

    minmax = lambda xs: (min(xs), max(xs))
    time_scale = int(1e9) / fps

    # read steering and image log
    steerings, speeds = read_steerings(steering_log, time_scale)

    image_stamps = read_image_stamps(image_log, 'center', time_scale)
    image_stampsl = read_image_stamps(image_log, 'left', time_scale)
    image_stampsr = read_image_stamps(image_log, 'right', time_scale)

    # statistics report
    print('timestamp range for all steerings: %d, %d' % minmax(steerings.keys()))
    print('timestamp range for all images: %d, %d' % minmax(image_stamps.keys()))
    print('min and max # of steerings per time unit: %d, %d' % minmax(list(map(len, steerings.values()))))
    print('min and max # of images per time unit: %d, %d' % minmax(list(map(len, image_stamps.values()))))

    # generate images and steerings within one time unit.
    # mean steering will be used for mulitple steering angels within the unit.

    start = max(min(steerings.keys()), min(image_stamps.keys()))
    end = min(max(steerings.keys()), max(image_stamps.keys()))

    print("sampling data from timestamp %d to %d" % (start, end))

    i = start
    samples = []
    steering_commands = []
    count = 0
    
    for i in range(start, end):
        for ids in image_stamps[i]:
            steering = np.mean(steerings[i])

            if np.abs(steering) < 0.2:
                if count <= 15000:
                    count+=1
                else:
                    continue

            item = (os.path.join(root, 'center', '{0}.png'.format(ids)), steering)
            steering_commands.append(item[1])
            samples.append(item)

        for ids in image_stampsl[i]:
            steering = camera_adjust(np.mean(steerings[i]), np.mean(speeds[i]), 'left')

            if np.abs(steering) < 0.2:
                if count <= 15000:
                    count+=1
                else:
                    continue

            item = (os.path.join(root, 'left', '{0}.png'.format(ids)), steering)
            steering_commands.append(item[1])
            samples.append(item)

        for ids in image_stampsr[i]:
            steering = camera_adjust(np.mean(steerings[i]), np.mean(speeds[i]), 'right')

            if np.abs(steering) < 0.2:
                if count <= 15000:
                    count+=1
                else:
                    continue

            item = (os.path.join(root, 'right', '{0}.png'.format(ids)), steering)
            steering_commands.append(item[1])
            samples.append(item)

    steering_commands = th.Tensor(np.array(steering_commands))
    class_vector = th.zeros(steering_commands.size())

    class_vector[th.abs(steering_commands)<2] = 0
    class_vector[th.abs(steering_commands)<1.5] = 1
    class_vector[th.abs(steering_commands)<1.0] = 2
    class_vector[th.abs(steering_commands)<0.5] = 3
    class_vector[th.abs(steering_commands)<0.1] = 4

    return samples, class_vector.numpy()

