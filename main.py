#!/usr/bin/python3
from feature_swarm import FeatureSwarm
from util import *

np.random.seed(1337)

# PARAMETERS
##########################

# KERNELS FOR DIFFERENT OPERATIONS
# not all of them are currently being used
kernel_opening = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
kernel_closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
kernel_dilation = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# MOG2 PARAMETERS
history = 20
varThreshold = 40
lr = 0.01
MOG2 = cv2.createBackgroundSubtractorMOG2(history, varThreshold, detectShadows=False)

# TRACKING
old_hue = None
feature_points = None  # collected feature points that we are tracking
feature_swarm = None  # class used to track all people
last_update = 0  # last time there was a detection using a feature detector (good features to track in our case)
frames_between_detection = 10  # do feature detection every n frames

# identified boxes, a list of boxes in the form of (x,y,w,h, id)
boxes_ided = []

lukas_kanade_params = dict(winSize=(10, 10),
                           maxLevel=2,
                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.80))

good_features_to_track_params = dict(maxCorners=1000, qualityLevel=0.01, minDistance=1, blockSize=5,
                                     useHarrisDetector=True)

# stuff used to check on the error on the count of people of each frame
frame_counts = parse_truth("A1_groundtruthC.txt")
error_per_frame = 0
predicted_avg = 0
truth_avg = 0

# INPUT VIDEO
cap = cv2.VideoCapture('video.mp4')
width = int(cap.get(3))
height = int(cap.get(4))
has_new_frame, frame = cap.read()
print("Running video with witdh: %s, height: %s" % (width, height))

# OUTPUT VIDEO
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

# stuff needed to draw trajectories
color = np.random.randint(0, 255, (1000, 3))
trajectories_mask = np.zeros_like(frame)
subtract_traj_mask = np.ones_like(frame)

index = 0
while has_new_frame:
    original = frame.copy()

    # DETECTION PHASE
    ############################

    # get hue plane, avoid shadows
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, s, v = cv2.split(hsv)
    hue = cv2.blur(hue, (4, 4))  # seems to help

    # get foreground figures
    foreground = MOG2.apply(hue, learningRate=lr)

    # apply morphological transformations to clear away noise and conglomerate blobs
    morph_transformed = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel_opening, iterations=1)
    morph_transformed = cv2.morphologyEx(morph_transformed, cv2.MORPH_CLOSE, kernel_closing, iterations=5)

    # get contours
    contours, _ = cv2.findContours(morph_transformed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = filter_contours(contours, 100)
    boxes = [cv2.boundingRect(contour) for contour in contours]

    # TRACKING PHASE
    ############################

    # mask used to exclude feature points found by the feature detector
    ret, mask = cv2.threshold(morph_transformed, 127, 255, cv2.THRESH_BINARY)

    if old_hue is not None:
        if feature_points is None:
            """
            Get features, create labelings based on 
            boxes and features
            """
            feature_points = cv2.goodFeaturesToTrack(old_hue, mask=foreground, **good_features_to_track_params)

            # add features to empty boxes
            a = generate_center_feature(feature_points, boxes, 1)
            if feature_points is not None and len(feature_points) > 0:
                feature_points = np.concatenate((feature_points, a), axis=0)
            else:
                feature_points = a

            feature_swarm = FeatureSwarm(feature_points, boxes, max_crowdness=40,
                                         growth_thres=10, dist_thres=120, identity_age=50)
        else:
            # find the features points in the next frame
            predicted_points, flags, err = cv2.calcOpticalFlowPyrLK(old_hue, hue, feature_points[:, :, :2], None,
                                                                    **lukas_kanade_params)

            # filter out lost points and update positions of the ones that we are keeping
            feature_swarm.update_current_points(predicted_points, flags)

            # find new features
            if (index - last_update) > frames_between_detection:
                last_update = index
                feature_points = cv2.goodFeaturesToTrack(old_hue, mask=foreground, **good_features_to_track_params)

                # add features to empty boxes
                a = generate_center_feature(feature_points, boxes, 1)
                if feature_points is not None and len(feature_points) > 0:
                    feature_points = np.concatenate((feature_points, a), axis=0)
                else:
                    feature_points = a
            else:
                feature_points = None

        # boxes identified by the feature swarm and their id
        boxes_ided = feature_swarm.update(feature_points, boxes)

        # features we will want to keep track at the next step
        feature_points = feature_swarm.points

    old_hue = hue

    # DRAWING PHASE
    ############################
    # draw frame number
    cv2.putText(original, str(index), (1180, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # counter of number of people
    draw_people_counter(original, len(contours))

    # counters for number of people exiting/entering
    if feature_swarm:
        cv2.putText(original, "en:" + str(len(feature_swarm.entering_left)), (0, 640), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255), 3)
        cv2.putText(original, "ex:" + str(len(feature_swarm.exiting_left)), (0, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255), 3)
        cv2.putText(original, "en:" + str(len(feature_swarm.entering_right)), (1170, 640), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (0, 0, 255), 3)
        cv2.putText(original, "ex:" + str(len(feature_swarm.exiting_right)), (1170, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5,
                    (0, 0, 255), 3)

    # draw the tracks
    for box in boxes_ided:
        x, y, w, h, id = box
        a = int(x + w / 2)
        b = int(y + h / 2)
        id = int(id)
        trajectories_mask = cv2.line(trajectories_mask, (a, b), (a, b), color[id].tolist(), 15)
        original = cv2.circle(original, (a, b), 5, color[id].tolist(), -1)
    trajectories_mask = cv2.addWeighted(trajectories_mask, 0.95, 0, 0, 0)
    original = cv2.add(original, cv2.bitwise_and(trajectories_mask, trajectories_mask, mask=(255 - mask)))

    # draw lines for left and right gates
    cv2.line(original, (120, 0), (120, 720), (255, 255, 125), 6)
    cv2.line(original, (1160, 0), (1160, 720), (255, 255, 125), 6)

    # draw ids
    for box in boxes_ided:
        x, y, w, h, id = box
        id = int(id)
        # cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(original, str(id), (int(x + w / 4), int(y + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 0, 0), 3)

    out.write(original)
    cv2.imshow('frame', original)

    # USER INPUT PHASE
    ############################

    # press "s" to pause/unpause
    # press "q" to qui
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):
        while True:
            key2 = cv2.waitKey(1) or 0xff
            if key2 == ord("s"):
                break
    elif key == ord("q"):
        break

    # get next frame and update counter
    has_new_frame, frame = cap.read()
    index += 1

    # STATISTICS COLLECTION PHASE
    ############################
    # check if the ground truth has any data on this frame
    if index in frame_counts:
        predicted_avg += len(contours)
        truth_avg += frame_counts[index]
        error_per_frame += abs(frame_counts[index] - len(contours))

# print statistics
# statistics relative to counting the number of people
error_per_frame /= len(frame_counts)
predicted_avg /= len(frame_counts)
truth_avg /= len(frame_counts)
print("Avg error per frame: %s" % error_per_frame)
print("Avg predicted per frame: %s" % predicted_avg)
print("Avg truth per frame: %s" % truth_avg)

"""
Save selected pedestrian trajectories to file. First we need
to find a matching between the id in the ground truth (selected pedestrian) and
an id in our history, once we have found that we write down its history
from a certain frame to another (based on the ground truth).
"""
# original = cv2.circle(original, (730, 350), 5, (255, 0, 255), -1)  # 10 frame 1
# original = cv2.circle(original, (1280, 280), 5, (255, 0, 255), -1)  # 36 frame 264
# original = cv2.circle(original, (4, 581), 5, (255, 0, 255), -1)  # 42 253
id_of_selected10 = find_selected_id_in_history(feature_swarm.history, 1, 730, 350)
id_of_selected36 = find_selected_id_in_history(feature_swarm.history, 264, 1280, 280)
id_of_selected42 = find_selected_id_in_history(feature_swarm.history, 253, 4, 581)
process_id_history(10, feature_swarm.history[id_of_selected10], 1, 96)
process_id_history(36, feature_swarm.history[id_of_selected36], 264, 451)
process_id_history(42, feature_swarm.history[id_of_selected42], 253, 476)

# release resources and exit
cap.release()
out.release()
cv2.destroyAllWindows()
