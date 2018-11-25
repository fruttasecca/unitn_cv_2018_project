import math
import operator
from collections import Counter

import cv2
import numpy as np
import rtree.index  # used to speed up/make scalable the implementation

from util import box_contains_point, resolve_candidates


class FeatureSwarm():
    def __init__(self, points, boxes, max_points_per_box=20, infer_thres=5, max_lost_dist=10,
                 max_lost_id_age=40, max_point_age=80):

        # number of processed frames
        self.max_lost_id_age = max_lost_id_age
        # number of identities seen, used to create an id for a new identity
        self.identities_counter = 0
        # max number of feature points that should be in a single bounding box
        self.max_points_per_box = max_points_per_box
        # max age of a feature point, points older that this are discarded
        self.max_point_age = max_point_age

        """
        A box will be assigned an identity i if the majority of points in that box
        have decided have identity i and are >= self.infer_thres.
        A new box will be created if there is a cluster of points with the same identity but
        that do not have a box containing them.
        """
        self.infer_thres = infer_thres

        """
        Maximum distance threshold to associate a lost identity (which was lost in a certain position
        x,y) to a box without identity.
        """
        self.max_lost_dist = max_lost_dist

        """
        Maximum age than a lost identity can have, identities that have been lost for too long are
        considered definitively lost and not reusable.
        """
        self.seen_frames = 0

        # a list of boxes and their id, of the form (x,y,w,h,id)
        self.boxes = None

        # numpy array of points (x,y,id,to_be_removed)
        self.points = None

        # Stuff used to keep track of the history of identities, who is exiting or entering, etc
        #####
        # identities considered to have correctly left the scene
        self.exited = set()
        # identities that are moving out or entering
        self.entering_right = set()
        self.exiting_right = set()
        self.entering_left = set()
        self.exiting_left = set()

        # identities that are currently considered lost
        self.lost_identities = dict()

        # history, for each identity a list of (x, y, frame index)
        self.history = dict()

        # wasted ids, which are ids that have been used for such a short amount of time
        # that they are going to be reused
        self.wasted = []

        """
        We are going to "attach" to the matrix of x,y points
        3 columns: 
        -an identity column, to what identity this point belong to, can also be
        consider what identity a point is "voting" for for the box it is into
        -age of the point
        
        the np.float32 is needed to make opencv kanade not angry
        """
        extra = np.ones((points.shape[0], 1, 2), dtype=np.float32)
        points = np.concatenate((points, extra), axis=2)
        points[:, :, 2] = -1  # no identity

        # assign ids to each box
        tmp = []
        for box in boxes:
            x, y, w, h = box
            tmp.append((x, y, w, h, self.identities_counter))
            self.identities_counter += 1
        boxes = tmp

        for point in points:

            # map point to a box
            candidate_boxes = []
            for box in boxes:
                x, y, w, h, _ = box
                if box_contains_point(x, y, w, h, point[0][0], point[0][1]):
                    candidate_boxes.append(box)
            if len(candidate_boxes) > 0:
                assigned_box = resolve_candidates(candidate_boxes, point[0])
                boxid = assigned_box[-1]

                # set box to which the point belongs to
                point[0, 2] = boxid

        # this can be retrieved to have a list of boxes of the form (x,y,w,hue,id)
        self.boxes = boxes

        # this can be retrieved to have an array of the form (x,y,id,boxid,last time since assigned to a box)
        self.points = points

    def update_current_points(self, updated_points, flags):
        """
        Remove points based on flags (1 to be kept, 0 to be removed),
        update the x and y coordinates of remaining points.
        If flags is none the function does not do anything.

        :param updated_points: Numpy array of the same length of the current points of the instance of this class,
            containing updated x and y coordinates for each point.
        :param flags: Numpy array of the same length of the current points of the instance of this class,
            containing either 1 of 0 for each point, 1 means that the point should be kept, 0 that the point
            will be dropped.
        """
        if flags is not None:
            # keep only stuff that was not lost
            updated_points = updated_points[flags == 1]
            old_points = self.points[flags == 1]

            # update positions (x,y) of points
            old_points[:, 0:2] = updated_points[:, :]

            # reshape needed because of how the opencv kanade implementation wants its inputs
            self.points = old_points.reshape(-1, 1, 4)

    def update(self, new_points, starting_boxes):
        """
        Update the current status of the feature swarm, adding new points,
        matching the input boxes and identities, and returning a list
        of boxes (and inferred boxes) with their id.

        :param new_points: np.array of shape (number of points, 1, 2) which is (number of points, 1, x and y).
        :param starting_boxes: List of boxes of the form [x,y,w,h].
        :return: List of boxes of the form (x,y,w,h,id).
        """
        self.seen_frames += 1
        self.points[:, 0, 3] += 1

        """
        Add new feature points to our current points.
        """
        if new_points is not None:
            # format new points in the way we like
            extra = np.ones((new_points.shape[0], 1, 2), dtype=np.float32)
            new_points = np.concatenate((new_points, extra), axis=2)
            new_points[:, :, 2] = -1  # no identity

            # get as points the old ones + the new ones
            self.points = np.concatenate((self.points, new_points), axis=0)

        # points with identity mapped to boxes, boxes now have an identity
        assigned_boxes = self.assign_decided_points_to_boxes(starting_boxes)

        # grow more boxes based on clusters of points with the same identity (when there is no box with such identity)
        self.infer_boxes(assigned_boxes)

        # points with no identiy now have the same identity of the box they are contained into
        self.assign_undecided_points_to_boxes(assigned_boxes)

        # For each box keep only up to a maximum number of points
        self.trim_boxes(assigned_boxes)

        """
        Identities that are contended by multiple boxes are resolved,
        boxes without identities have an identity assigned to them.
        """
        boxes_ided = self.decide_contended_identities(assigned_boxes)

        # keep track of history
        for box in boxes_ided:
            x, y, w, h, id = box
            if id not in self.history:
                self.history[id] = []
            self.history[id].append((x + w / 2, y + h / 2, self.seen_frames))

        # keep updating the position of lost identities based
        # on previous movements, when they were not lost
        self.advance_lost_identities()

        # process history to decide who is entering, exiting, etc.
        self.process_history()

        """
        Remove points that are too old or are to be discarded.
        """
        self.points = self.points[self.points[:, 0, 3] < self.max_point_age]
        return boxes_ided

    def decide_contended_identities(self, boxes):
        """
        Identities that are contended by multiple boxes are resolved,
        boxes without identities have an identity assigned to them.

        :param boxes: List of boxes in the form of (x,y,w,h, list of points, id).
        :return: List of boxes in the form of (x,y,w,h,id).
        """
        # boxes with id to return
        result = []

        """
        For each identity make a list of boxes
        contenting that identity, for each box
        with no identity append it to a list of
        boxes with no identity.
        """
        # boxes that have no id
        boxes_noid = []
        # for each id get a list of contending boxes
        id_to_boxes = dict()
        for box in boxes:
            id = box[5]
            if id == -1:
                boxes_noid.append(box)
            else:
                if id not in id_to_boxes:
                    id_to_boxes[id] = []
                id_to_boxes[id].append(box)

        """
        Decide on what identities are considered to be lost.
        """
        self.lost_identities.clear()
        for identity, history in self.history.items():
            last_time_seen = history[-1][2]  # frame number of last update
            diff = self.seen_frames - last_time_seen

            # not seen this frame & not considered exited & not too old
            if identity not in id_to_boxes and identity not in self.exited and diff < self.max_lost_id_age:
                self.lost_identities[identity] = (history[-1])

        """
        Given an identity contended by multiple boxes
        give identity to the box that has the most points
        voting for it.
        """
        for id, boxes in id_to_boxes.items():
            maxvotes = 0
            maxid = -1
            for boxindex, box in enumerate(boxes):
                x, y, w, h, voters, id = box
                if len(voters) > maxvotes:
                    maxvotes = len(voters)
                    maxid = boxindex

            # assign identity to winning box
            winning = boxes.pop(maxid)
            x, y, w, h, voters, id = winning
            result.append((x, y, w, h, id))

            boxes_noid.extend(boxes)

        """
        For each box with no identity assign it an identity
        by first looking for lost identities that may fit this
        box, or just creating a new id if there are no such identities.
        """
        for box in boxes_noid:
            x, y, w, h, points, id = box
            # if it contains at least a point
            if len(points) >= 1:
                id = self.find_lost_identity((x, y, w, h))
                result.append((x, y, w, h, id))
                # set the identity of all points in this box
                for i in points:
                    self.points[i, 0, 2] = id

        return result

    def find_lost_identity(self, box):
        """
        Search for a lost identity that can be matched to the box, based
        on distance and for how long the identity has been lost, if there
        is no lost identity respecting the threshold minimum distance, either
        try to reuse a wasted identity or create a new one by updating
        the identity counter.
        :param box: Box of the form (x,y,w,h).
        :return: An id for the box.
        """

        # get lost identity nearest to our box
        x, y, w, h = box
        bcx = x + w / 2
        bcy = y + h / 2

        # to do that look for the nearest lost identity within the threshold
        mindist = self.max_lost_dist
        minid = None
        for id, stats in self.lost_identities.items():
            cx, cy, last_seen = stats
            if (self.seen_frames - last_seen) < self.max_lost_id_age:
                dist = math.sqrt((bcx - cx) ** 2 + (bcy - cy) ** 2)
                # if near enough and not too old
                if dist < mindist and (self.seen_frames - self.history[id][-1][-1]) < self.max_lost_id_age:
                    mindist = dist
                    minid = id

        # delete the matched identity from the list of lost identities
        if minid and minid not in self.exited:
            del self.lost_identities[minid]
        else:
            if len(self.wasted) > 0:
                minid = self.wasted.pop()
            else:
                minid = self.identities_counter
                self.identities_counter += 1
        return minid

    def process_history(self):
        """
        Process history, for each
        identity decide which one is exiting or entering
        the scene, and which one is definitively exited, check
        if some ids have been wasted, etc.
        """
        # reset who is entering/exiting
        self.entering_right.clear()
        self.exiting_right.clear()
        self.entering_left.clear()
        self.exiting_left.clear()

        for id, history in list(self.history.items()):  # transform to a list so that we can remove stuff while running
            begx, _, _ = history[0]
            endx, _, lastseen = history[-1]
            goingright = endx > begx

            # also consider that that we have lost track of, but very recently
            if ((self.seen_frames - lastseen) < 5) and (len(history) >= 5):
                # something related to the right gate
                if endx > 1160:
                    if goingright and len(history) > 15:
                        self.exiting_right.add(id)
                        self.exited.add(id)
                    else:
                        self.entering_right.add(id)
                # something related to the left gate
                elif endx < 120:
                    if goingright:
                        self.entering_left.add(id)
                    elif len(history) > 15:
                        self.exiting_left.add(id)
                        self.exited.add(id)

            # if an id has been used for too few frames then it got lost recycle it
            if (self.seen_frames - lastseen) > 1 and len(history) <= 5:
                self.wasted.append(id)
                # make sure it is as if the wasted id did not exist
                self.lost_identities.pop(id, None)
                self.history.pop(id, None)
                self.exited.discard(id)
                self.entering_right.discard(id)
                self.exiting_right.discard(id)
                self.entering_left.discard(id)
                self.exiting_left.discard(id)

    def advance_lost_identities(self):
        """
        Update the x and y coordinate of lost identities based
        on their previous history, if their history is long enough.
        Currently only updating the x coordinate, because pedestrians
        might change their vertical direction after bumping into each other,
        so past info might not be so useful.
        Only histories of lost identities that are longer than 10 are updated.
        """
        for identity in self.lost_identities:
            recent_hist = self.history[identity][-30:]
            if len(recent_hist) > 10:
                oldx, oldy, _ = recent_hist[0]
                lastx, lasty, lastframeseen = recent_hist[-1]
                gradx = (lastx - oldx) / len(recent_hist) if len(recent_hist) > 5 else 0
                """
                Currently not updating on y given the fact that identities usually keep going on their
                horizontal direction, but not always keep the same vertical one.
                """
                # grady = (lasty - oldy) / len(recent_hist)

                self.history[identity].append((lastx + gradx, lasty, lastframeseen))
                self.lost_identities[identity] = (lastx + gradx, lasty, lastframeseen)

    def assign_undecided_points_to_boxes(self, boxes):
        """
        Go through undecided points and try to assign an identity
        to them based on the box they are contained into. If they
        are contained in a box with an identity that identity will
        become the one of the points, otherwise if they are in box
        that only contains undecided points just append the box to the
        list of boxes, the box will have in its list of points
        those undecided points.

        :param boxes: List of boxes of the form (x,y,w,h, points belonging to them and voting for their identity, id)
        """
        # use a tree index to make everything more scalable
        tree_index = rtree.index.Rtree()
        boxdict = dict()
        for box in boxes:
            x, y, w, h, _, _ = box
            tree_index.add(len(boxdict), (x + w / 2, y + h / 2, x + w / 2, y + h / 2))
            boxdict[len(boxdict)] = box

        # look for the nearest box for each point that has no decided identity
        indexes = np.where(self.points[:, 0, 2] == -1)[0]
        for point_index in indexes:
            px, py, = self.points[point_index, 0, :2]
            nearbox = list(tree_index.nearest((px, py, px, py), 1))
            if len(nearbox) > 0:
                nearbox = nearbox[0]
                x, y, w, h, decided, box_id = boxdict[nearbox]

                if box_contains_point(x, y, w, h, px, py):
                    self.points[point_index, 0, 2] = box_id
                    decided.append(point_index)

    def assign_decided_points_to_boxes(self, boxes):
        """
        For each box find points that are inside it and that have an identity,
        use those identities to democratically decide the identity of the box.
        Return boxes with their points voting for their identity, and their identity.

        :param boxes: List of boxes of the form (x,y,w,h)
        :return: List of boxes of the form (x,y,w,h, list of points that have voted for this identiy for this box, id)
        """

        # put boxes in a rtree
        tree_index = rtree.index.Rtree()
        boxdict = dict()
        for box in boxes:
            x, y, w, h, = box
            tree_index.add(len(boxdict), (x + w / 2, y + h / 2, x + w / 2, y + h / 2))
            boxdict[len(boxdict)] = (x, y, w, h, [], [])

        # look for the nearest box for each point that has a decided identity
        # append such point and its vote to the box's lists
        indexes = np.where(self.points[:, 0, 2] != -1)[0]
        for point_index in indexes:
            px, py, vote = self.points[point_index, 0, :3]
            nearbox = list(tree_index.nearest((px, py, px, py), 1))
            if len(nearbox) > 0:
                nearbox = nearbox[0]
                x, y, w, h, points, votes = boxdict[nearbox]

                if box_contains_point(x, y, w, h, px, py):
                    points.append(point_index)
                    votes.append(vote)

        assigned_boxes = []

        # for each box decide identity based on majority voting
        for box in boxdict.values():
            x, y, w, h, points, votes = box

            # group votes by id, decide on id
            summed_votes = Counter(votes)
            sorted_votes = sorted(summed_votes.items(), key=operator.itemgetter(1), reverse=True)

            # id for this box is undecided if there are no points with an identity within this box
            id = -1 if (len(summed_votes) == 0) else sorted_votes[0][0]
            tmp = []

            if id != -1:
                # points assigned to this box should be only the points that have voted for that particular identity
                for p, v in zip(points, votes):
                    if v == id:
                        tmp.append(p)
            assigned_boxes.append([x, y, w, h, tmp, int(id)])
        return assigned_boxes

    def infer_boxes(self, boxes):
        """
        If there is a cluster of points with the same identity
        and there is no box already having that identity, "grow"
        a new box around those points.
        :param boxes: List of boxes of the form (x,y,w,h,points voting for its identity, id)
        """
        # ids that have already been assigned to boxes
        assigned_ids = set(box[-1] for box in boxes if box[1] != -1)

        # get all unique identities in our feature points
        ids = np.unique(self.points[:, 0, 2])
        for id in ids:
            # see if we can grow a box only for identities that have not
            # a box of their own but have enough points with that identity
            # and clustered near each other
            if id not in assigned_ids and id != -1:
                points_indices = np.where(self.points[:, 0, 2] == id)
                points = self.points[points_indices, 0, :2][0]
                decided = list(points_indices[0])
                if len(points) >= self.infer_thres:
                    x, y, w, h = cv2.boundingRect(points)
                    # accept only boxes of certain size
                    if w < 120 and h < 120:
                        boxes.append([x, y, max(50, w), max(50, h), decided, id])

    def trim_boxes(self, boxes):
        """
        Keep only a maximum amount of points for each box.
        :param boxes: List of boxes of form (x,y,w,h,points,id).
        """
        # for each box keep only up to max_crowndess
        for box in boxes:
            points = box[4]
            points = list(reversed(points))
            self.points[points[self.max_points_per_box:], 0, 3] = self.max_point_age
            del points[self.max_points_per_box:]
            box[4] = points
