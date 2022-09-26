
# Original code from https://github.com/rafaelpadilla/review_object_detection_metrics adapted for this project
# Only here for compatibility with code for bounding boxes, not used in the final paper

import sys
from collections import Counter

import numpy as np
import pandas as pd

from enum import Enum
from math import isclose


################  ENUMERATORS  ################
class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    RELATIVE = 1
    ABSOLUTE = 2


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GROUND_TRUTH = 1
    DETECTED = 2


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    """
    XYWH = 1
    XYX2Y2 = 2
    PASCAL_XML = 3
    YOLO = 4


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EVERY_POINT_INTERPOLATION = 1
    ELEVEN_POINT_INTERPOLATION = 2


# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convert_to_relative_values(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # YOLO's format
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return x, y, w, h


# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convert_to_absolute_values(size, box):
    w_box = size[0] * box[2]
    h_box = size[1] * box[3]

    x1 = (float(box[0]) * float(size[0])) - (w_box / 2)
    y1 = (float(box[1]) * float(size[1])) - (h_box / 2)
    x2 = x1 + w_box
    y2 = y1 + h_box
    return round(x1), round(y1), round(x2), round(y2)



################  BOUNDING BOX  ################

class BoundingBox:
    """ Class representing a bounding box. """
    def __init__(self,
                 image_name,
                 class_id=None,
                 coordinates=None,
                 type_coordinates=CoordinatesType.ABSOLUTE,
                 img_size=None,
                 bb_type=BBType.GROUND_TRUTH,
                 confidence=None,
                 format=BBFormat.XYWH):
        """ Constructor.

        Parameters
        ----------
            image_name : str
                String representing the name of the image.
            class_id : str
                String value representing class id.
            coordinates : tuple
                Tuple with 4 elements whose values (float) represent coordinates of the bounding \\
                    box.
                The coordinates can be (x, y, w, h)=>(float,float,float,float) or(x1, y1, x2, y2)\\
                    =>(float,float,float,float).
                See parameter `format`.
            type_coordinates : Enum (optional)
                Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image. Default:'Absolute'.
            img_size : tuple (optional)
                Image size in the format (width, height)=>(int, int) representinh the size of the
                image of the bounding box. If type_coordinates is 'Relative', img_size is required.
            bb_type : Enum (optional)
                Enum identifying if the bounding box is a ground truth or a detection. If it is a
                detection, the confidence must be informed.
            confidence : float (optional)
                Value representing the confidence of the detected object. If detectionType is
                Detection, confidence needs to be informed.
            format : Enum
                Enum (BBFormat.XYWH or BBFormat.XYX2Y2) indicating the format of the coordinates of
                the bounding boxes.
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
                BBFomat.YOLO: <x_center> <y_center> <width> <height>. (relative)
        """

        self._image_name = image_name
        self._type_coordinates = type_coordinates
        self._confidence = confidence
        self._class_id = class_id
        self._format = format
        if bb_type == BBType.DETECTED and confidence is None:
            raise IOError(
                'For bb_type=\'Detected\', it is necessary to inform the confidence value.')
        self._bb_type = bb_type

        if img_size is None:
            self._width_img = None
            self._height_img = None
        else:
            self._width_img = img_size[0]
            self._height_img = img_size[1]

        # If YOLO format (rel_x_center, rel_y_center, rel_width, rel_height), change it to absolute format (x,y,w,h)
        if format == BBFormat.YOLO:
            assert self._width_img is not None and self._height_img is not None
            self._format = BBFormat.XYWH
            self._type_coordinates = CoordinatesType.RELATIVE

        self.set_coordinates(coordinates,
                             img_size=img_size,
                             type_coordinates=self._type_coordinates)

    def set_coordinates(self, coordinates, type_coordinates, img_size=None):
        self._type_coordinates = type_coordinates
        if type_coordinates == CoordinatesType.RELATIVE and img_size is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')

        # If relative coordinates, convert to absolute values
        # For relative coords: (x,y,w,h)=(X_center/img_width , Y_center/img_height)
        if (type_coordinates == CoordinatesType.RELATIVE):
            self._width_img = img_size[0]
            self._height_img = img_size[1]
            if self._format == BBFormat.XYWH:
                (self._x, self._y, self._w,
                 self._h) = convert_to_absolute_values(img_size, coordinates)
                self._x2 = self._w
                self._y2 = self._h
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            elif self._format == BBFormat.XYX2Y2:
                x1, y1, x2, y2 = coordinates
                # Converting to absolute values
                self._x = round(x1 * self._width_img)
                self._x2 = round(x2 * self._width_img)
                self._y = round(y1 * self._height_img)
                self._y2 = round(y2 * self._height_img)
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
            else:
                raise IOError(
                    'For relative coordinates, the format must be XYWH (x,y,width,height)')
        # For absolute coords: (x,y,w,h)=real bb coords
        else:
            self._x = coordinates[0]
            self._y = coordinates[1]
            if self._format == BBFormat.XYWH:
                self._w = coordinates[2]
                self._h = coordinates[3]
                self._x2 = self._x + self._w
                self._y2 = self._y + self._h
            else:  # self._format == BBFormat.XYX2Y2: <left> <top> <right> <bottom>.
                self._x2 = coordinates[2]
                self._y2 = coordinates[3]
                self._w = self._x2 - self._x
                self._h = self._y2 - self._y
        # Convert all values to float
        self._x = float(self._x)
        self._y = float(self._y)
        self._w = float(self._w)
        self._h = float(self._h)
        self._x2 = float(self._x2)
        self._y2 = float(self._y2)

    def get_absolute_bounding_box(self, format=BBFormat.XYWH):
        """ Get bounding box in its absolute format.

        Parameters
        ----------
        format : Enum
            Format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2) to be retreived.

        Returns
        -------
        tuple
            Four coordinates representing the absolute values of the bounding box.
            If specified format is BBFormat.XYWH, the coordinates are (upper-left-X, upper-left-Y,
            width, height).
            If format is BBFormat.XYX2Y2, the coordinates are (upper-left-X, upper-left-Y,
            bottom-right-X, bottom-right-Y).
        """
        if format == BBFormat.XYWH:
            return (self._x, self._y, self._w, self._h)
        elif format == BBFormat.XYX2Y2:
            return (self._x, self._y, self._x2, self._y2)

    def get_relative_bounding_box(self, img_size=None):
        """ Get bounding box in its relative format.

        Parameters
        ----------
        img_size : tuple
            Image size in the format (width, height)=>(int, int)

        Returns
        -------
        tuple
            Four coordinates representing the relative values of the bounding box (x,y,w,h) where:
                x,y : bounding_box_center/width_of_the_image
                w   : bounding_box_width/width_of_the_image
                h   : bounding_box_height/height_of_the_image
        """
        if img_size is None and self._width_img is None and self._height_img is None:
            raise IOError(
                'Parameter \'img_size\' is required. It is necessary to inform the image size.')
        if img_size is not None:
            return convert_to_relative_values((img_size[0], img_size[1]),
                                              (self._x, self._x2, self._y, self._y2))
        else:
            return convert_to_relative_values((self._width_img, self._height_img),
                                              (self._x, self._x2, self._y, self._y2))

    def get_image_name(self):
        """ Get the string that represents the image.

        Returns
        -------
        string
            Name of the image.
        """
        return self._image_name

    def get_confidence(self):
        """ Get the confidence level of the detection. If bounding box type is BBType.GROUND_TRUTH,
        the confidence is None.

        Returns
        -------
        float
            Value between 0 and 1 representing the confidence of the detection.
        """
        return self._confidence

    def get_format(self):
        """ Get the format of the bounding box (BBFormat.XYWH or BBFormat.XYX2Y2).

        Returns
        -------
        Enum
            Format of the bounding box. It can be either:
                BBFormat.XYWH: <left> <top> <width> <height>
                BBFomat.XYX2Y2: <left> <top> <right> <bottom>.
        """
        return self._format

    def set_class_id(self, class_id):
        self._class_id = class_id

    def set_bb_type(self, bb_type):
        self._bb_type = bb_type

    def get_class_id(self):
        """ Get the class of the object the bounding box represents.

        Returns
        -------
        string
            Class of the detected object (e.g. 'cat', 'dog', 'person', etc)
        """
        return self._class_id

    def get_image_size(self):
        """ Get the size of the image where the bounding box is represented.

        Returns
        -------
        tupe
            Image size in pixels in the format (width, height)=>(int, int)
        """
        return (self._width_img, self._height_img)

    def get_area(self):
        assert isclose(self._w * self._h, (self._x2 - self._x) * (self._y2 - self._y))
        assert (self._x2 > self._x)
        assert (self._y2 > self._y)
        return (self._x2 - self._x + 1) * (self._y2 - self._y + 1)

    def get_coordinates_type(self):
        """ Get type of the coordinates (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).

        Returns
        -------
        Enum
            Enum representing if the bounding box coordinates (x,y,w,h) are absolute or relative
                to size of the image (CoordinatesType.RELATIVE or CoordinatesType.ABSOLUTE).
        """
        return self._type_coordinates

    def get_bb_type(self):
        """ Get type of the bounding box that represents if it is a ground-truth or detected box.

        Returns
        -------
        Enum
            Enum representing the type of the bounding box (BBType.GROUND_TRUTH or BBType.DETECTED)
        """
        return self._bb_type

    def __str__(self):
        abs_bb_xywh = self.get_absolute_bounding_box(format=BBFormat.XYWH)
        abs_bb_xyx2y2 = self.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        area = self.get_area()
        return f'image name: {self._image_name}\nclass: {self._class_id}\nbb (XYWH): {abs_bb_xywh}\nbb (X1Y1X2Y2): {abs_bb_xyx2y2}\narea: {area}\nbb_type: {self._bb_type}'

    def __eq__(self, other):
        if not isinstance(other, BoundingBox):
            # unrelated types
            return False
        return str(self) == str(other)

    @staticmethod
    def compare(det1, det2):
        """ Static function to compare if two bounding boxes represent the same area in the image,
            regardless the format of their boxes.

        Parameters
        ----------
        det1 : BoundingBox
            BoundingBox object representing one bounding box.
        dete2 : BoundingBox
            BoundingBox object representing another bounding box.

        Returns
        -------
        bool
            True if both bounding boxes have the same coordinates, otherwise False.
        """
        det1BB = det1.getAbsoluteBoundingBox()
        det1img_size = det1.getImageSize()
        det2BB = det2.getAbsoluteBoundingBox()
        det2img_size = det2.getImageSize()

        if det1.get_class_id() == det2.get_class_id() and \
           det1.get_confidence() == det2.get_confidence() and \
           det1BB[0] == det2BB[0] and \
           det1BB[1] == det2BB[1] and \
           det1BB[2] == det2BB[2] and \
           det1BB[3] == det2BB[3] and \
           det1img_size[0] == det1img_size[0] and \
           det2img_size[1] == det2img_size[1]:
            return True
        return False

    @staticmethod
    def clone(bounding_box):
        """ Static function to clone a given bounding box.

        Parameters
        ----------
        bounding_box : BoundingBox
            Bounding box object to be cloned.

        Returns
        -------
        BoundingBox
            Cloned BoundingBox object.
        """
        absBB = bounding_box.get_absolute_bounding_box(format=BBFormat.XYWH)
        # return (self._x,self._y,self._x2,self._y2)
        new_bounding_box = BoundingBox(bounding_box.get_image_name(),
                                       bounding_box.get_class_id(),
                                       absBB[0],
                                       absBB[1],
                                       absBB[2],
                                       absBB[3],
                                       type_coordinates=bounding_box.getCoordinatesType(),
                                       img_size=bounding_box.getImageSize(),
                                       bb_type=bounding_box.getbb_type(),
                                       confidence=bounding_box.getConfidence(),
                                       format=BBFormat.XYWH)
        return new_bounding_box

    @staticmethod
    def iou(boxA, boxB):
        coords_A = boxA.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        coords_B = boxB.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
        # if boxes do not intersect
        if BoundingBox.have_intersection(coords_A, coords_B) is False:
            return 0
        interArea = BoundingBox.get_intersection_area(coords_A, coords_B)
        union = BoundingBox.get_union_areas(boxA, boxB, interArea=interArea)
        # intersection over union
        iou = interArea / union
        assert iou >= 0
        return iou

    # boxA = (Ax1,Ay1,Ax2,Ay2)
    # boxB = (Bx1,By1,Bx2,By2)
    @staticmethod
    def have_intersection(boxA, boxB):
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if boxA[0] > boxB[2]:
            return False  # boxA is right of boxB
        if boxB[0] > boxA[2]:
            return False  # boxA is left of boxB
        if boxA[3] < boxB[1]:
            return False  # boxA is above boxB
        if boxA[1] > boxB[3]:
            return False  # boxA is below boxB
        return True

    @staticmethod
    def get_intersection_area(boxA, boxB):
        if isinstance(boxA, BoundingBox):
            boxA = boxA.get_absolute_bounding_box(BBFormat.XYX2Y2)
        if isinstance(boxB, BoundingBox):
            boxB = boxB.get_absolute_bounding_box(BBFormat.XYX2Y2)
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # intersection area
        return (xB - xA + 1) * (yB - yA + 1)

    @staticmethod
    def get_union_areas(boxA, boxB, interArea=None):
        area_A = boxA.get_area()
        area_B = boxB.get_area()
        if interArea is None:
            interArea = BoundingBox.get_intersection_area(boxA, boxB)
        return float(area_A + area_B - interArea)

    @staticmethod
    def get_amount_bounding_box_all_classes(bounding_boxes, reverse=False):
        classes = list(set([bb._class_id for bb in bounding_boxes]))
        ret = {}
        for c in classes:
            ret[c] = len(BoundingBox.get_bounding_box_by_class(bounding_boxes, c))
        # Sort dictionary by the amount of bounding boxes
        ret = {k: v for k, v in sorted(ret.items(), key=lambda item: item[1], reverse=reverse)}
        return ret

    @staticmethod
    def get_bounding_box_by_class(bounding_boxes, class_id):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_class_id() == class_id]

    @staticmethod
    def get_bounding_boxes_by_image_name(bounding_boxes, image_name):
        # get only specified bounding box type
        return [bb for bb in bounding_boxes if bb.get_image_name() == image_name]

    @staticmethod
    def get_total_images(bounding_boxes):
        return len(list(set([bb.get_image_name() for bb in bounding_boxes])))

    @staticmethod
    def get_average_area(bounding_boxes):
        areas = [bb.get_area() for bb in bounding_boxes]
        return sum(areas) / len(areas)


################  PASCAL_VOC_EVALUATOR  ################

def calculate_ap_every_point(rec, prec):
    mrec = []
    mrec.append(0)
    [mrec.append(e) for e in rec]
    mrec.append(1)
    mpre = []
    mpre.append(0)
    [mpre.append(e) for e in prec]
    mpre.append(0)
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ii = []
    for i in range(len(mrec) - 1):
        if mrec[1:][i] != mrec[0:-1][i]:
            ii.append(i + 1)
    ap = 0
    for i in ii:
        ap = ap + np.sum((mrec[i] - mrec[i - 1]) * mpre[i])
    return [ap, mpre[0:len(mpre) - 1], mrec[0:len(mpre) - 1], ii]


def calculate_ap_11_point_interp(rec, prec, recall_vals=11):
    mrec = []
    # mrec.append(0)
    [mrec.append(e) for e in rec]
    # mrec.append(1)
    mpre = []
    # mpre.append(0)
    [mpre.append(e) for e in prec]
    # mpre.append(0)
    recallValues = np.linspace(0, 1, recall_vals)
    recallValues = list(recallValues[::-1])
    rhoInterp = []
    recallValid = []
    # For each recallValues (0, 0.1, 0.2, ... , 1)
    for r in recallValues:
        # Obtain all recall values higher or equal than r
        argGreaterRecalls = np.argwhere(mrec[:] >= r)
        pmax = 0
        # If there are recalls above r
        if argGreaterRecalls.size != 0:
            pmax = max(mpre[argGreaterRecalls.min():])
        recallValid.append(r)
        rhoInterp.append(pmax)
    # By definition AP = sum(max(precision whose recall is above r))/11
    ap = sum(rhoInterp) / len(recallValues)
    # Generating values for the plot
    rvals = []
    rvals.append(recallValid[0])
    [rvals.append(e) for e in recallValid]
    rvals.append(0)
    pvals = []
    pvals.append(0)
    [pvals.append(e) for e in rhoInterp]
    pvals.append(0)
    # rhoInterp = rhoInterp[::-1]
    cc = []
    for i in range(len(rvals)):
        p = (rvals[i], pvals[i - 1])
        if p not in cc:
            cc.append(p)
        p = (rvals[i], pvals[i])
        if p not in cc:
            cc.append(p)
    recallValues = [i[0] for i in cc]
    rhoInterp = [i[1] for i in cc]
    return [ap, rhoInterp, recallValues, None]


def get_pascalvoc_metrics(gt_boxes,
                          det_boxes,
                          iou_threshold=0.5,
                          method=MethodAveragePrecision.EVERY_POINT_INTERPOLATION,
                          generate_table=False):
    """Get the metrics used by the VOC Pascal 2012 challenge.
    Args:
        boundingboxes: Object of the class BoundingBoxes representing ground truth and detected
        bounding boxes;
        iou_threshold: IOU threshold indicating which detections will be considered TP or FP
        (dget_pascalvoc_metricsns:
        A dictioanry contains information and metrics of each class.
        The key represents the class and the values are:
        dict['class']: class representing the current dictionary;
        dict['precision']: array with the precision values;
        dict['recall']: array with the recall values;
        dict['AP']: average precision;
        dict['interpolated precision']: interpolated precision values;
        dict['interpolated recall']: interpolated recall values;
        dict['total positives']: total number of ground truth positives;
        dict['total TP']: total number of True Positive detections;
        dict['total FP']: total number of False Positive detections;"""
    ret = {}
    # Get classes of all bounding boxes separating them by classes
    gt_classes_only = []
    classes_bbs = {}
    for bb in gt_boxes:
        c = bb.get_class_id()
        gt_classes_only.append(c)
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['gt'].append(bb)
    gt_classes_only = list(set(gt_classes_only))
    for bb in det_boxes:
        c = bb.get_class_id()
        classes_bbs.setdefault(c, {'gt': [], 'det': []})
        classes_bbs[c]['det'].append(bb)

    # Precision x Recall is obtained individually by each class
    for c, v in classes_bbs.items():
        # Report results only in the classes that are in the GT
        if c not in gt_classes_only:
            continue
        npos = len(v['gt'])
        # sort detections by decreasing confidence
        dects = [a for a in sorted(v['det'], key=lambda bb: bb.get_confidence(), reverse=True)]
        TP = np.zeros(len(dects))
        FP = np.zeros(len(dects))
        # create dictionary with amount of expected detections for each image
        detected_gt_per_image = Counter([bb.get_image_name() for bb in gt_boxes])
        for key, val in detected_gt_per_image.items():
            detected_gt_per_image[key] = np.zeros(val)
        # print(f'Evaluating class: {c}')
        dict_table = {
            'image': [],
            'confidence': [],
            'TP': [],
            'FP': [],
            'acc TP': [],
            'acc FP': [],
            'precision': [],
            'recall': []
        }
        # Loop through detections
        for idx_det, det in enumerate(dects):
            img_det = det.get_image_name()

            if generate_table:
                dict_table['image'].append(img_det)
                dict_table['confidence'].append(f'{100*det.get_confidence():.2f}%')

            # Find ground truth image
            gt = [gt for gt in classes_bbs[c]['gt'] if gt.get_image_name() == img_det]
            # Get the maximum iou among all detectins in the image
            iouMax = sys.float_info.min
            # Given the detection det, find ground-truth with the highest iou
            for j, g in enumerate(gt):
                # print('Ground truth gt => %s' %
                #       str(g.get_absolute_bounding_box(format=BBFormat.XYX2Y2)))
                iou = BoundingBox.iou(det, g)
                if iou > iouMax:
                    iouMax = iou
                    id_match_gt = j
            # Assign detection as TP or FP
            if iouMax >= iou_threshold:
                # gt was not matched with any detection
                if detected_gt_per_image[img_det][id_match_gt] == 0:
                    TP[idx_det] = 1  # detection is set as true positive
                    detected_gt_per_image[img_det][
                        id_match_gt] = 1  # set flag to identify gt as already 'matched'
                    # print("TP")
                    if generate_table:
                        dict_table['TP'].append(1)
                        dict_table['FP'].append(0)
                else:
                    FP[idx_det] = 1  # detection is set as false positive
                    if generate_table:
                        dict_table['FP'].append(1)
                        dict_table['TP'].append(0)
                    # print("FP")
            # - A detected "cat" is overlaped with a GT "cat" with IOU >= iou_threshold.
            else:
                FP[idx_det] = 1  # detection is set as false positive
                if generate_table:
                    dict_table['FP'].append(1)
                    dict_table['TP'].append(0)
                # print("FP")
        # compute precision, recall and average precision
        acc_FP = np.cumsum(FP)
        acc_TP = np.cumsum(TP)
        rec = acc_TP / npos
        prec = np.divide(acc_TP, (acc_FP + acc_TP))
        if generate_table:
            dict_table['acc TP'] = list(acc_TP)
            dict_table['acc FP'] = list(acc_FP)
            dict_table['precision'] = list(prec)
            dict_table['recall'] = list(rec)
            table = pd.DataFrame(dict_table)
        else:
            table = None
        # Depending on the method, call the right implementation
        if method == MethodAveragePrecision.EVERY_POINT_INTERPOLATION:
            [ap, mpre, mrec, ii] = calculate_ap_every_point(rec, prec)
        elif method == MethodAveragePrecision.ELEVEN_POINT_INTERPOLATION:
            [ap, mpre, mrec, _] = calculate_ap_11_point_interp(rec, prec)
        else:
            Exception('method not defined')
        # add class result in the dictionary to be returned
        ret[c] = {
            'precision': prec,
            'recall': rec,
            'AP': ap,
            'interpolated precision': mpre,
            'interpolated recall': mrec,
            'total positives': npos,
            'total TP': np.sum(TP),
            'total FP': np.sum(FP),
            'method': method,
            'iou': iou_threshold,
            'table': table
        }
    # For mAP, only the classes in the gt set should be considered
    mAP = sum([v['AP'] for k, v in ret.items() if k in gt_classes_only]) / len(gt_classes_only)
    return {'per_class': ret, 'mAP': mAP}


def compute_map(predictions: list, gts: list, iou_threshold=0.5):
    """
    Computes mAP metric for the current lists. Supposes 1 prediction and ground truth per image, of the same object.
    :param predictions: list of torch.tensors containing predicted bboxes (with confidence score at index 0)
    :param gts: list of torch.tensors containing ground truth bboxes (with object class at index 0)
    :param iou_threshold: threshold to apply to the Intersection-over-Union of bounding boxes
    :return: a float in range [0,1] containing the mAP for current set
    """

    # 1. creates BoundingBox lists for predictions w/ confidence
    # 2. creates BoundingBox lists for ground truths w/out confidence
    # 3. calls get_pascalvoc_metrics()
    # 4. returns only the mAP value
    pred_bb = []
    gt_bb = []
    for i, bbs in enumerate(zip(predictions, gts)):
        p, g = bbs
        if isinstance(p, dict):
            # used Detector --> more boxes per image
            for bb in range(len(p['boxes'])):
                box = p['boxes'][bb]
                target = p['labels'][bb].item()
                confidence = p['scores'][bb].item()
                pred_bb.append(BoundingBox(
                    image_name=str(i),
                    bb_type=BBType.DETECTED,
                    coordinates=box.int().cpu().tolist(),
                    confidence=confidence,
                    class_id=target,
                    format=BBFormat.XYX2Y2
                ))

        else:
            # used ConditionalRegressor --> just 1 correct box per image
            p = p.int().cpu().tolist()
            g = g.int().cpu().tolist()
            print(p)
            print(g)
            print()

            if p[1] < p[3] and p[2] < p[4]:
                pred_bb.append(BoundingBox(
                    image_name=str(i),
                    bb_type=BBType.DETECTED,
                    coordinates=p[1:],
                    confidence=p[0],
                    class_id=g[0],
                    format=BBFormat.XYX2Y2
                ))
        gt_bb.append(BoundingBox(
            image_name=str(i),
            coordinates=g[1:],
            class_id=g[0],
            format=BBFormat.XYX2Y2
        ))
    return get_pascalvoc_metrics(gt_bb, pred_bb, iou_threshold=iou_threshold)['mAP']


def get_iou(pred, gt):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes (in format X1Y1X2Y2).
    """

    # SUPPOSING bb1 --> prediction, bb2 --> ground truth
    pred = {c: v for c, v in zip(['x1', 'y1', 'x2', 'y2'], pred)}
    gt = {c: v for c, v in zip(['x1', 'y1', 'x2', 'y2'], gt)}

    assert gt['x1'] < gt['x2'] and gt['y1'] < gt['y2']
    if not (pred['x1'] < pred['x2'] and
            pred['y1'] < pred['y2']):
        return 0.0

    # determine the coordinates of the intersection rectangle
    x_left = max(pred['x1'], gt['x1'])
    y_top = max(pred['y1'], gt['y1'])
    x_right = min(pred['x2'], gt['x2'])
    y_bottom = min(pred['y2'], gt['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)

    # compute the area of both AABBs
    bb1_area = (pred['x2'] - pred['x1'] + 1) * (pred['y2'] - pred['y1'] + 1)
    bb2_area = (gt['x2'] - gt['x1'] + 1) * (gt['y2'] - gt['y1'] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def compute_iou(predictions, gts, return_list=False):
    """Computes mean IoU metric from paired lists of GTs and predictions, as torch"""
    acc = []
    for p, g in zip(predictions, gts):
        l_pred = p.int().cpu().tolist()
        l_gt = g.int().cpu().tolist()
        if len(l_gt) == 4 and len(l_pred) == 4:
            acc.append(get_iou(l_pred, l_gt))
        elif len(l_gt) == 5 and len(l_pred) == 5:
            acc.append(get_iou(l_pred[1:], l_gt[1:]))
        else:
            raise ValueError(f'inconsistent or unacceptable sizes for pred ({len(l_pred)}) and ({len(l_gt)})')
    if return_list:
        return acc
    else:
        return sum(acc) / len(acc)


if __name__ == '__main__':
    from pprint import pprint
    bbox1 = BoundingBox(
        image_name='img1',
        coordinates=(0, 0, 100, 100)
    )
    bbox2 = BoundingBox(
        image_name='img1',
        coordinates=(10, 10, 100, 100),
        confidence=0.767,
        bb_type=BBType.DETECTED
    )
    bbox3 = BoundingBox(
        image_name='img1',
        coordinates=(5, 5, 115, 115),
        confidence=0.88,
        bb_type=BBType.DETECTED
    )
    m1 = get_pascalvoc_metrics([bbox1], [bbox2])
    m2 = get_pascalvoc_metrics([bbox1], [bbox2, bbox3])
    pprint(m1)
    pprint(m2)
    print("MAP: ", m1['mAP'], "|", m2['mAP'])

