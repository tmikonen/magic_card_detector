"""
Module for detecting and recognizing Magic: the Gathering cards from an image.
"""

import glob
import cProfile
import pstats
import io
import pickle
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.affinity import scale
from scipy.ndimage import rotate
from PIL import Image as PILImage

import imagehash
import cv2


def order_polygon_points(x_p, y_p):
    """
    Orders polygon points into a counterclockwise order.
    x_p, y_p are the x and y coordinates of the polygon points.
    """
    angle = np.arctan2(y_p - np.average(y_p), x_p - np.average(x_p))
    ind = np.argsort(angle)
    return (x_p[ind], y_p[ind])


def four_point_transform(image, poly):
    """
    A perspective transform for a quadrilateral polygon.
    Slightly modified version of the same function from
    https://github.com/EdjeElectronics/OpenCV-Playing-Card-Detector
    """
    pts = np.zeros((4, 2))
    pts[:, 0] = np.asarray(poly.exterior.coords)[:-1, 0]
    pts[:, 1] = np.asarray(poly.exterior.coords)[:-1, 1]
    # obtain a consistent order of the points and unpack them
    # individually
    rect = np.zeros((4, 2))
    (rect[:, 0], rect[:, 1]) = order_polygon_points(pts[:, 0], pts[:, 1])

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    # width_a = np.sqrt(((b_r[0] - b_l[0]) ** 2) + ((b_r[1] - b_l[1]) ** 2))
    # width_b = np.sqrt(((t_r[0] - t_l[0]) ** 2) + ((t_r[1] - t_l[1]) ** 2))
    width_a = np.sqrt(((rect[1, 0] - rect[0, 0]) ** 2) +
                      ((rect[1, 1] - rect[0, 1]) ** 2))
    width_b = np.sqrt(((rect[3, 0] - rect[2, 0]) ** 2) +
                      ((rect[3, 1] - rect[2, 1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((rect[0, 0] - rect[3, 0]) ** 2) +
                       ((rect[0, 1] - rect[3, 1]) ** 2))
    height_b = np.sqrt(((rect[1, 0] - rect[2, 0]) ** 2) +
                       ((rect[1, 1] - rect[2, 1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order

    rect = np.array([
        [rect[0, 0], rect[0, 1]],
        [rect[1, 0], rect[1, 1]],
        [rect[2, 0], rect[2, 1]],
        [rect[3, 0], rect[3, 1]]], dtype="float32")

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    transform = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform, (max_width, max_height))

    # return the warped image
    return warped


def line_intersection(x, y):
    """
    Calculates the intersection point of two lines, defined by the points
    (x1, y1) and (x2, y2) (first line), and
    (x3, y3) and (x4, y4) (second line).
    If the lines are parallel, (nan, nan) is returned.
    """
    slope_0 = (x[0] - x[1]) * (y[2] - y[3])
    slope_2 = (y[0] - y[1]) * (x[2] - x[3])
    if slope_0 == slope_2:
        # parallel lines
        xis = np.nan
        yis = np.nan
    else:
        xy_01 = x[0] * y[1] - y[0] * x[1]
        xy_23 = x[2] * y[3] - y[2] * x[3]
        denom = slope_0 - slope_2

        xis = (xy_01 * (x[2] - x[3]) - (x[0] - x[1]) * xy_23) / denom
        yis = (xy_01 * (y[2] - y[3]) - (y[0] - y[1]) * xy_23) / denom

    return (xis, yis)


def simplify_polygon(in_poly,
                     length_cutoff=0.15,
                     maxiter=None,
                     segment_to_remove=None):
    """
    Removes segments from a (convex) polygon by continuing neighboring
    segments to a new point of intersection. Purpose is to approximate
    rounded polygons (quadrilaterals) with more sharp-cornered ones.
    """

    x_in = np.asarray(in_poly.exterior.coords)[:-1, 0]
    y_in = np.asarray(in_poly.exterior.coords)[:-1, 1]
    len_poly = len(x_in)
    niter = 0
    if segment_to_remove is not None:
        maxiter = 1
    while len_poly > 4:
        d_in = np.sqrt(np.ediff1d(x_in, to_end=x_in[0]-x_in[-1]) ** 2. +
                       np.ediff1d(y_in, to_end=y_in[0]-y_in[-1]) ** 2.)
        d_tot = np.sum(d_in)
        if segment_to_remove is not None:
            k = segment_to_remove
        else:
            k = np.argmin(d_in)
        if d_in[k] < length_cutoff * d_tot:
            ind = generate_point_indices(k - 1, k + 1, len_poly)
            (xis, yis) = line_intersection(x_in[ind], y_in[ind])
            x_in[k] = xis
            y_in[k] = yis
            x_in = np.delete(x_in, (k+1) % len_poly)
            y_in = np.delete(y_in, (k+1) % len_poly)
            len_poly = len(x_in)
            niter += 1
            if (maxiter is not None) and (niter >= maxiter):
                break
        else:
            break

    out_poly = Polygon([[ix, iy] for (ix, iy) in zip(x_in, y_in)])

    return out_poly


def generate_point_indices(index_1, index_2, max_len):
    """
    Returns the four indices that give the end points of
    polygon segments corresponding to index_1 and index_2,
    modulo the number of points (max_len).
    """
    return np.array([index_1 % max_len,
                     (index_1 + 1) % max_len,
                     index_2 % max_len,
                     (index_2 + 1) % max_len])


def generate_quad_corners(indices, x, y):
    """
    Returns the four intersection points from the
    segments defined by the x coordinates (x),
    y coordinates (y), and the indices.
    """
    (i, j, k, l) = indices

    def gpi(index_1, index_2):
        return generate_point_indices(index_1, index_2, len(x))

    xis = np.empty(4)
    yis = np.empty(4)
    xis.fill(np.nan)
    yis.fill(np.nan)

    if j <= i or k <= j or l <= k:
        pass
    else:
        (xis[0], yis[0]) = line_intersection(x[gpi(i, j)],
                                             y[gpi(i, j)])
        (xis[1], yis[1]) = line_intersection(x[gpi(j, k)],
                                             y[gpi(j, k)])
        (xis[2], yis[2]) = line_intersection(x[gpi(k, l)],
                                             y[gpi(k, l)])
        (xis[3], yis[3]) = line_intersection(x[gpi(l, i)],
                                             y[gpi(l, i)])

    return (xis, yis)


def generate_quad_candidates(in_poly):
    """
    Generates a list of bounding quadrilaterals for a polygon,
    using all possible combinations of four intersection points
    derived from four extended polygon segments.
    The number of combinations increases rapidly with the order
    of the polygon, so simplification should be applied first to
    remove very short segments from the polygon.
    """
    # make sure that the points are ordered
    (x_s, y_s) = order_polygon_points(
        np.asarray(in_poly.exterior.coords)[:-1, 0],
        np.asarray(in_poly.exterior.coords)[:-1, 1])
    x_s_ave = np.average(x_s)
    y_s_ave = np.average(y_s)
    x_shrunk = x_s_ave + 0.9999 * (x_s - x_s_ave)
    y_shrunk = y_s_ave + 0.9999 * (y_s - y_s_ave)
    shrunk_poly = Polygon([[x, y] for (x, y) in
                           zip(x_shrunk, y_shrunk)])
    quads = []
    len_poly = len(x_s)

    for indices in product(range(len_poly), repeat=4):
        (xis, yis) = generate_quad_corners(indices, x_s, y_s)
        if (np.sum(np.isnan(xis)) + np.sum(np.isnan(yis))) > 0:
            # no intersection point for some of the lines
            pass
        else:
            (xis, yis) = order_polygon_points(xis, yis)
            enclose = True
            quad = Polygon([(xis[0], yis[0]),
                            (xis[1], yis[1]),
                            (xis[2], yis[2]),
                            (xis[3], yis[3])])
            if not quad.contains(shrunk_poly):
                enclose = False
            if enclose:
                quads.append(quad)
    return quads

def get_bounding_quad(hull_poly):
    """
    Returns the minimum area quadrilateral that contains (bounds)
    the convex hull (openCV format) given as input.
    """
    simple_poly = simplify_polygon(hull_poly)
    bounding_quads = generate_quad_candidates(simple_poly)
    bquad_areas = np.zeros(len(bounding_quads))
    for iquad, bquad in enumerate(bounding_quads):
        bquad_areas[iquad] = bquad.area
    min_area_quad = bounding_quads[np.argmin(bquad_areas)]

    return min_area_quad


def quad_corner_diff(hull_poly, bquad_poly, region_size=0.9):
    """
    Returns the difference between areas in the corners of a rounded
    corner and the aproximating sharp corner quadrilateral.
    region_size (param) determines the region around the corner where
    the comparison is done.
    """
    bquad_corners = np.zeros((4, 2))
    bquad_corners[:, 0] = np.asarray(bquad_poly.exterior.coords)[:-1, 0]
    bquad_corners[:, 1] = np.asarray(bquad_poly.exterior.coords)[:-1, 1]

    # The point inside the quadrilateral, region_size towards the quad center
    interior_points = np.zeros((4, 2))
    interior_points[:, 0] = np.average(bquad_corners[:, 0]) + \
        region_size * (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))
    interior_points[:, 1] = np.average(bquad_corners[:, 1]) + \
        region_size * (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))

    # The points p0 and p1 (at each corner) define the line whose intersections
    # with the quad together with the corner point define the triangular
    # area where the roundness of the convex hull in relation to the bounding
    # quadrilateral is evaluated.
    # The line (out of p0 and p1) is constructed such that it goes through the
    # "interior_point" and is orthogonal to the line going from the corner to
    # the center of the quad.
    p0_x = interior_points[:, 0] + \
        (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p1_x = interior_points[:, 0] - \
        (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p0_y = interior_points[:, 1] - \
        (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))
    p1_y = interior_points[:, 1] + \
        (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))

    corner_area_polys = []
    for i in range(len(interior_points[:, 0])):
        bline = LineString([(p0_x[i], p0_y[i]), (p1_x[i], p1_y[i])])
        corner_area_polys.append(Polygon(
            [bquad_poly.intersection(bline).coords[0],
             bquad_poly.intersection(bline).coords[1],
             (bquad_corners[i, 0], bquad_corners[i, 1])]))

    hull_corner_area = 0
    quad_corner_area = 0
    for capoly in corner_area_polys:
        quad_corner_area += capoly.area
        hull_corner_area += capoly.intersection(hull_poly).area

    return 1. - hull_corner_area/quad_corner_area


def convex_hull_polygon(contour):
    """
    Returns the convex hull of the given contour as a polygon.
    """
    hull = cv2.convexHull(contour)
    phull = Polygon([[x, y] for (x, y) in
                     zip(hull[:, :, 0], hull[:, :, 1])])
    return phull


def polygon_form_factor(poly):
    """
    The ratio between the polygon area and circumference length,
    scaled by the length of the shortest segment.
    """
    # minimum side length
    d_0 = np.amin(np.sqrt(np.sum(np.diff(np.asarray(poly.exterior.coords),
                                         axis=0) ** 2., axis=1)))
    return poly.area/(poly.length * d_0)


def characterize_card_contour(card_contour,
                              max_segment_area,
                              image_area):
    """
    Calculates a bounding polygon for a contour, in addition
    to several charasteristic parameters.
    """
    phull = convex_hull_polygon(card_contour)
    if (phull.area < 0.1 * max_segment_area or
            phull.area < image_area / 1000.):
        # break after card size range has been explored
        continue_segmentation = False
        is_card_candidate = False
        bounding_poly = None
        crop_factor = 1.
    else:
        continue_segmentation = True
        bounding_poly = get_bounding_quad(phull)
        qc_diff = quad_corner_diff(phull, bounding_poly)
        crop_factor = min(1., (1. - qc_diff * 22. / 100.))
        is_card_candidate = bool(
            0.1 * max_segment_area < bounding_poly.area <
            image_area * 0.99 and
            qc_diff < 0.35 and
            polygon_form_factor(bounding_poly) < 0.33)

    return (continue_segmentation,
            is_card_candidate,
            bounding_poly,
            crop_factor)

#
# CLASSES
#


class CardCandidate:
    """
    Class representing a segment of the image that may be a recognizable card.
    """
    def __init__(self, im_seg, bquad, fraction):
        self.image = im_seg
        self.bounding_quad = bquad
        self.is_recognized = False
        self.recognition_score = 0.
        self.is_fragment = False
        self.image_area_fraction = fraction
        self.name = 'unknown'

    def contains(self, other):
        """
        Returns whether the bounding polygon of the card candidate
        contains the bounding polygon of the other candidate.
        """
        return other.bounding_quad.within(self.bounding_quad)


class ReferenceImage:
    """
    Container for a card image and the associated recoginition data.
    """

    def __init__(self, name, original_image, clahe, phash=None):
        self.name = name
        self.original = original_image
        self.clahe = clahe
        self.adjusted = None
        self.phash = phash

        if self.original is not None:
            self.histogram_adjust()
            self.calculate_phash()

    def calculate_phash(self):
        """
        Calculates the perceptive hash for the image
        """
        self.phash = imagehash.phash(
                        PILImage.fromarray(np.uint8(255 * cv2.cvtColor(
                            self.adjusted, cv2.COLOR_BGR2RGB))),
                        hash_size=32)

    def histogram_adjust(self):
        """
        Adjusts the image by contrast limited histogram adjustmend (clahe)
        """
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        lightness, redness, yellowness = cv2.split(lab)
        corrected_lightness = self.clahe.apply(lightness)
        limg = cv2.merge((corrected_lightness, redness, yellowness))
        self.adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


class TestImage:
    """
    Container for a card image and the associated recoginition data.
    """

    def __init__(self, name, original_image, clahe):
        self.name = name
        self.original = original_image
        self.clahe = clahe
        self.adjusted = None
        self.phash = None
        self.visual = False
        self.histogram_adjust()
        # self.calculate_phash()

        self.candidate_list = []

    def histogram_adjust(self):
        """
        Adjusts the image by contrast limited histogram adjustmend (clahe)
        """
        lab = cv2.cvtColor(self.original, cv2.COLOR_BGR2LAB)
        lightness, redness, yellowness = cv2.split(lab)
        corrected_lightness = self.clahe.apply(lightness)
        limg = cv2.merge((corrected_lightness, redness, yellowness))
        self.adjusted = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    def mark_fragments(self):
        """
        Finds doubly (or multiply) segmented cards and marks all but one
        as a fragment (that is, an unnecessary duplicate)
        """
        for (candidate, other_candidate) in product(self.candidate_list,
                                                    repeat=2):
            if ((candidate.is_recognized or other_candidate.is_recognized) and
                    candidate != other_candidate):
                i_area = candidate.bounding_quad.intersection(
                    other_candidate.bounding_quad).area
                min_area = min(candidate.bounding_quad.area,
                               other_candidate.bounding_quad.area)
                if i_area > 0.5 * min_area:
                    if (candidate.is_recognized and
                            other_candidate.is_recognized):
                        if (candidate.recognition_score <
                                other_candidate.recognition_score):
                            candidate.is_fragment = True
                        else:
                            other_candidate.is_fragment = True
                    else:
                        if candidate.is_recognized:
                            other_candidate.is_fragment = True
                        else:
                            candidate.is_fragment = True

    def plot_image_with_recognized(self, visual=False):
        """
        Plots the recognized cards into the full image.
        """
        # Plotting
        plt.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        for candidate in self.candidate_list:
            if not candidate.is_fragment:
                full_image = self.adjusted
                bquad_corners = np.empty((4, 2))
                bquad_corners[:, 0] = np.asarray(
                    candidate.bounding_quad.exterior.coords)[:-1, 0]
                bquad_corners[:, 1] = np.asarray(
                    candidate.bounding_quad.exterior.coords)[:-1, 1]

                plt.plot(np.append(bquad_corners[:, 0],
                                   bquad_corners[0, 0]),
                         np.append(bquad_corners[:, 1],
                                   bquad_corners[0, 1]), 'g-')
                bounding_poly = Polygon([[x, y] for (x, y) in
                                         zip(bquad_corners[:, 0],
                                             bquad_corners[:, 1])])
                fntsze = int(4 * bounding_poly.length / full_image.shape[1])
                bbox_color = 'white' if candidate.is_recognized else 'red'
                plt.text(np.average(bquad_corners[:, 0]),
                         np.average(bquad_corners[:, 1]),
                         candidate.name,
                         horizontalalignment='center',
                         fontsize=fntsze,
                         bbox=dict(facecolor=bbox_color,
                                   alpha=0.7))

        plt.savefig('results/MTG_card_recognition_results_' +
                    str(self.name.split('.jpg')[0]) +
                    '.jpg', dpi=600, bbox='tight')
        if visual:
            plt.show()

    def print_recognized(self):
        """
        Prints out the recognized cards from the image.
        """
        recognized_list = self.return_recognized()
        print('Recognized cards (' +
              str(len(recognized_list)) +
              ' cards):')
        for card in recognized_list:
            print(card.name)

    def return_recognized(self):
        """
        Returns a list of recognized and non-fragment card candidates.
        """
        recognized_list = []
        for candidate in self.candidate_list:
            if candidate.is_recognized and not candidate.is_fragment:
                recognized_list.append(candidate)
        return recognized_list


class MagicCardDetector:
    """
    MTG card detector class.
    """

    def __init__(self):
        self.reference_images = []
        self.test_images = []

        self.verbose = False
        self.visual = False

        self.hash_separation_thr = 4.
        self.thr_lvl = 70

        self.clahe = cv2.createCLAHE(clipLimit=2.0,
                                     tileGridSize=(8, 8))

    def export_reference_data(self, path):
        """
        Exports the phash and card name of the reference data list.
        """
        hlist = []
        for image in self.reference_images:
            hlist.append(ReferenceImage(image.name,
                                        None,
                                        None,
                                        image.phash))

        with open(path, 'wb') as fhandle:
            pickle.dump(hlist, fhandle)

    def read_prehashed_reference_data(self, path):
        """
        Reads pre-calculated hashes of the reference images.
        """
        print('Reading prehashed data from ' + str(path))
        print('...', end=' ')
        with open(path, 'rb') as filename:
            hashed_list = pickle.load(filename)
        for ref_im in hashed_list:
            self.reference_images.append(
                ReferenceImage(ref_im.name, None, self.clahe, ref_im.phash))
        print('Done.')

    def read_and_adjust_reference_images(self, path):
        """
        Reads and histogram-adjusts the reference image set.
        Pre-calculates the hashes of the images.
        """
        print('Reading images from ' + str(path))
        print('...', end=' ')
        filenames = glob.glob(path + '*.jpg')
        for filename in filenames:
            img = cv2.imread(filename)
            img_name = filename.split(path)[1]
            self.reference_images.append(
                ReferenceImage(img_name, img, self.clahe))
        print('Done.')

    def read_and_adjust_test_images(self, path):
        """
        Reads and histogram-adjusts the test image set.
        """
        print('Reading images from ' + str(path))
        print('...', end=' ')
        filenames = glob.glob(path + '*.jpg')
        for filename in filenames:
            img = cv2.imread(filename)
            img_name = filename.split(path)[1]
            self.test_images.append(
                TestImage(img_name, img, self.clahe))
        print('Done.')

    def contour_image_gray(self, full_image, thresholding='adaptive'):
        """
        Grayscale transform, thresholding, countouring and sorting by area.
        """
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        if thresholding == 'adaptive':
            fltr_size = 1 + 2 * (min(full_image.shape[0],
                                     full_image.shape[1]) // 20)
            thresh = cv2.adaptiveThreshold(gray,
                                           255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY,
                                           fltr_size,
                                           5)
        else:
            _, thresh = cv2.threshold(gray,
                                      70,
                                      255,
                                      cv2.THRESH_BINARY)
        if self.visual:
            plt.imshow(thresh)
            plt.show()

        _, contours, _ = cv2.findContours(
            np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def contour_image_rgb(self, full_image):
        """
        Grayscale transform, thresholding, countouring and sorting by area.
        """
        blue, green, red = cv2.split(full_image)
        blue = self.clahe.apply(blue)
        green = self.clahe.apply(green)
        red = self.clahe.apply(red)
        _, thr_b = cv2.threshold(blue, 110, 255, cv2.THRESH_BINARY)
        _, thr_g = cv2.threshold(green, 110, 255, cv2.THRESH_BINARY)
        _, thr_r = cv2.threshold(red, 110, 255, cv2.THRESH_BINARY)
        _, contours_b, _ = cv2.findContours(
            np.uint8(thr_b), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_g, _ = cv2.findContours(
            np.uint8(thr_g), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, contours_r, _ = cv2.findContours(
            np.uint8(thr_r), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours_b + contours_g + contours_r
        return contours

    def contour_image(self, full_image, mode='gray'):
        """
        Wrapper for selecting the countouring / thresholding algorithm
        """
        if mode == 'gray':
            contours = self.contour_image_gray(full_image,
                                               thresholding='simple')
        elif mode == 'rgb':
            contours = self.contour_image_rgb(full_image)
        elif mode == 'all':
            contours = self.contour_image_gray(full_image,
                                               thresholding='simple')
            contours += self.contour_image_gray(full_image,
                                               thresholding='adaptive')
            contours += self.contour_image_rgb(full_image)
        else:
            raise ValueError('Unknown segmentation mode.')
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours_sorted

    def segment_image(self, test_image, contouring_mode='gray'):
        """
        Segments the given image into card candidates, that is,
        regions of the image that have a high chance
        of containing a recognizable card.
        """
        full_image = test_image.adjusted.copy()
        image_area = full_image.shape[0] * full_image.shape[1]
        max_segment_area = 0.01  # largest card area

        contours = self.contour_image(full_image, mode=contouring_mode) 
        for card_contour in contours:
            (continue_segmentation,
             is_card_candidate,
             bounding_poly,
             crop_factor) = characterize_card_contour(
                 card_contour, max_segment_area, image_area)
            if not continue_segmentation:
                break
            if is_card_candidate:
                if max_segment_area == 0.01:
                    max_segment_area = bounding_poly.area
                warped = four_point_transform(full_image,
                                              scale(bounding_poly,
                                                    xfact=crop_factor,
                                                    yfact=crop_factor,
                                                    origin='centroid'))
                test_image.candidate_list.append(
                    CardCandidate(warped,
                                  bounding_poly,
                                  bounding_poly.area / image_area))
                if self.verbose:
                    print('Segmented ' +
                          str(len(test_image.candidate_list)) +
                          ' candidates.')

    def phash_diff(self, phash_im):
        """
        Calculates the phash difference between the given phash and
        each of the reference images.
        """
        diff = np.zeros(len(self.reference_images))
        for i, ref_im in enumerate(self.reference_images):
            diff[i] = phash_im - ref_im.phash
        return diff

    def phash_compare(self, im_seg):
        """
        Runs perceptive hash comparison between given image and
        the (pre-hashed) reference set.
        """

        card_name = 'unknown'
        is_recognized = False
        recognition_score = 0.
        rotations = np.array([0., 90., 180., 270.])

        d_0_dist = np.zeros(len(rotations))
        d_0 = np.zeros((len(self.reference_images), len(rotations)))
        for j, rot in enumerate(rotations):
            if not -1.e-5 < rot < 1.e-5:
                phash_im = imagehash.phash(PILImage.fromarray(
                    np.uint8(255 * cv2.cvtColor(rotate(im_seg, rot),
                                                cv2.COLOR_BGR2RGB))),
                                           hash_size=32)
            else:
                phash_im = imagehash.phash(PILImage.fromarray(
                    np.uint8(255 * cv2.cvtColor(im_seg,
                                                cv2.COLOR_BGR2RGB))),
                                           hash_size=32)
            d_0[:, j] = self.phash_diff(phash_im)
            d_0_ = d_0[d_0[:, j] > np.amin(d_0[:, j]), j]
            d_0_ave = np.average(d_0_)
            d_0_std = np.std(d_0_)
            d_0_dist[j] = (d_0_ave - np.amin(d_0[:, j]))/d_0_std
            if self.verbose:
                print('Phash statistical distance' + str(d_0_dist[j]))
            if (d_0_dist[j] > self.hash_separation_thr and
                    np.argmax(d_0_dist) == j):
                card_name = self.reference_images[np.argmin(d_0[:, j])]\
                            .name.split('.jpg')[0]
                is_recognized = True
                recognition_score = d_0_dist[j] / self.hash_separation_thr
                break
        return (is_recognized, recognition_score, card_name)

    def recognize_segment(self, image_segment):
        """
        Wrapper for different segmented image recognition algorithms.
        """
        return self.phash_compare(image_segment)

    def run_recognition(self, image_index):
        """
        Tries to recognize cards from the image specified.
        The image has been read in and adjusted previously,
        and is contained in the internal data list of the class.
        """
        test_image = self.test_images[image_index]

        if self.visual:
            print('Original image')
            plt.imshow(cv2.cvtColor(test_image.original,
                                    cv2.COLOR_BGR2RGB))
            plt.show()

        print('Segmentating card candidates out of the image...')

        test_image.candidate_list.clear()
        self.segment_image(test_image, contouring_mode='gray')

        print('Done. Found ' +
              str(len(test_image.candidate_list)) + ' candidates.')
        print('Recognizing candidates.')

        for i_cand, candidate in enumerate(test_image.candidate_list):
            im_seg = candidate.image
            if self.verbose:
                print(str(i_cand + 1) + " / " +
                      str(len(test_image.candidate_list)))

            # Easy fragment / duplicate detection
            for other_candidate in test_image.candidate_list:
                if (other_candidate.is_recognized and
                        not other_candidate.is_fragment):
                    if other_candidate.contains(candidate):
                        candidate.is_fragment = True
            if not candidate.is_fragment:
                (candidate.is_recognized,
                 candidate.recognition_score,
                 candidate.name) = self.recognize_segment(im_seg)

        print('Done. Found ' +
              str(len(test_image.return_recognized())) +
              ' cards.')
        print('Removing duplicates...')
        # Final fragment detection
        test_image.mark_fragments()
        print('Done.')
        print('Plotting and saving the results...')
        test_image.plot_image_with_recognized(self.visual)
        print('Done.')
        test_image.print_recognized()
        print('Recognition done.')


def main():
    """
    Python MTG Card Detector.
    Example run.
    Can be used also purely through the defined classes.
    """
    # Instantiate the detector
    card_detector = MagicCardDetector()

    do_profile = False
    card_detector.visual = True

    # Read the reference and test data sets
    # card_detector.read_and_adjust_reference_images(
    #     '../../MTG/Card_Images/LEA/')
    card_detector.read_prehashed_reference_data('./alpha_reference_phash.dat')
    card_detector.read_and_adjust_test_images('../MTG_alpha_test_images/')

    if do_profile:
        # Start up the profiler.
        profiler = cProfile.Profile()
        profiler.enable()

    # Run the card detection and recognition.
    # 9 = blue aggro
    # 7 = wizards2
    # 5 = swamp
    # 6 = instill energy


    for im_ind in range(7, 8):
        card_detector.run_recognition(im_ind)

    if do_profile:
        # Stop profiling and organize and print profiling results.
        profiler.disable()
        profiler.dump_stats('magic_card_detector.prof')
        profiler_stream = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE
        profiler_stats = pstats.Stats(
            profiler, stream=profiler_stream).sort_stats(sortby)
        profiler_stats.print_stats(20)
        print(profiler_stream.getvalue())


if __name__ == "__main__":
    main()
