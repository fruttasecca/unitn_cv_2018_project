#!/usr/bin/python3
import numpy as np
import cv2


def parse_truth(truth):
    """
    Parse the ground truth file, mapping each frame to the
    number of people in it.
    :param truth: Path to the truth file.
    :return: A dict mapping keys to the true number of people contained in it.
    """
    frame_to_count = dict()
    with open(truth, "r") as file:
        for line in file:
            frame, id, posx, posy = line.split(',')
            frame = int(frame)
            frame_to_count[frame] = frame_to_count.get(frame, 0) + 1
    return frame_to_count


def box_contains_point(bx, by, bw, bh, px, py):
    """
    Check if a box contains a point.
    :param bx:
    :param by:
    :param bw:
    :param bh:
    :param px:
    :param py:
    :return:
    """
    return bx <= px <= (bx + bw) and by <= py <= (by + bh)


def draw_people_counter(frame, count):
    """
    Draw the counter of people in the middle of the screen.

    :param frame: Frame on which to write.
    :param count: Value to be written, an integer.
    :return:
    """
    height, width, _ = frame.shape

    # set parameters
    corner_point = (int(0.35 * width), int(height * 0.15))  # bottom left corner
    scale = 4
    color = (0, 0, 255)
    width = 10
    # draw
    cv2.putText(frame, 'People: %s' % count, corner_point, cv2.FONT_HERSHEY_PLAIN, scale, color, width)


def generate_center_feature(features, boxes, n, thres=40):
    """
    Add features to empty boxes (no feature points inside them) that have an area greater than
    a certain threshold, features are added by considering
    the center of the box as a feature.
    This is done with the idea that big boxes are probably correct foreground detections, and if they
    are empty its because the feature detector is failing.
    :param features: Existing feature points.
    :param boxes: List of boxes.
    :param n: Number of features to have for each empty box, this is done to get enough features
    to trigger the entity matcher (feature swarm) to consider this box.
    :param thres: Area threshold, boxes below this area are not candidates for new features.
    :return:
    """
    new_features = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        if (w * h) > thres:
            hasfeatures = False
            index = 0
            while index < len(features) and not hasfeatures:
                feat = features[index, 0]
                fx = feat[0]
                fy = feat[1]
                hasfeatures = box_contains_point(x, y, w, h, fx, fy)
                index += 1

            if not hasfeatures:
                for i in range(n):
                    new_features.append((x + w / 2, y + h / 2))
    new_features = np.array(new_features, dtype=np.float32)
    new_features.resize(len(new_features), 1, 2)
    return new_features


def filter_contours(contours, thres):
    """
    Filter out contours based on area and position.
    :param contours: List of contours (x,y,w,h).
    :param thres: Contours with an area lower than this are filtered out.
    :return: List of (filtered) contours.
    """
    filtered = []
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        area = w * h
        if area > thres and 50 < y < 650:
            filtered.append(c)
    return filtered


def find_selected_id_in_history(history, selected_frame, posx, posy):
    """
    Find which one of our ids matches with the selected pedestrian to be tracked,
    which was in posx and posy at the selected_frame.
    :param history:
    :param selected_frame:
    :param posx:
    :param posy:
    :return:
    """
    mindist = 1e10
    minid = -1

    # go through the history of each id and find which was the nearest
    # when the pedestrian to follow appeared in the scene
    for id, h in history.items():
        if h[0][2] <= selected_frame:
            for (x, y, frame) in h:
                if frame == selected_frame:
                    dist = (posx - x) ** 2 + (posy - y) ** 2
                    if dist < mindist:
                        mindist = dist
                        minid = id
    return minid


def process_id_history(id, history, framebeg, frameend):
    """
    Give the history of an identity, save the history
    from a certain frame to another, writing it to a file
    named "estimated_id_<identity number>.txt".
    :param id: Id of the entity we are dealing with, used for the file name.
    :param history: History of such identity.
    :param framebeg: Beginning frame.
    :param frameend: Ending frame.
    """
    i = framebeg
    with open("estimated_id_%i.txt" % id, "w") as file:
        for (x, y, frame) in history:
            if framebeg <= frame <= frameend:
                file.write("%s,%s,%s\n" % (i, x, y))
                i += 1


def resolve_candidates(boxes, point):
    """
    Get the best box for a point, among the candidates, based
    un distance.
    This is something written hastly given that it's only
    run once for the whole script.
    Box to point matching is done with a rtree during the actual loop.
    :param boxes:
    :param point:
    :return:
    """
    if len(boxes) > 1:
        # look for box with the minimum centre
        mindist = 1e10
        minbox = None
        for i, box in enumerate(boxes):
            x, y, w, h, _ = box
            dist = ((x + w / 2) - point[0]) ** 2 + ((x + y / 2) - point[1]) ** 2
            if dist < mindist:
                mindist = dist
                minbox = box
        return minbox
    # return the only box
    return boxes[0]
