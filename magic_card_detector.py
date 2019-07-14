"""
Module for detecting and recognizing Magic: the Gathering cards from an image.
"""

import glob
import cProfile
import pstats
import io

import numpy as np
import matplotlib.pyplot as plt

from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
from shapely.affinity import scale
from scipy.ndimage import rotate
from PIL import Image as PILImage

import imagehash
import cv2


def order_points(pts):
    """
    Orders polygon points for the perspective transform.
    """
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")

    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    rect[0] = pts[np.argmin(pts.sum(axis=1))]
    rect[2] = pts[np.argmax(pts.sum(axis=1))]

    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    # return the ordered coordinates
    return rect


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


def line_intersection(x_p, y_p):
    """
    Calculates the intersection point of two lines, defined by the points
    (x1, y1) and (x2, y2) (first line), and
    (x3, y3) and (x4, y4) (second line).
    If the lines are parallel, (nan, nan) is returned.
    """
    if (x_p[0] - x_p[1]) * (y_p[2] - y_p[3]) == (y_p[0] - y_p[1]) * (x_p[2] - x_p[3]):
        # parallel lines
        xis = np.nan
        yis = np.nan
    else:
        xis = ((x_p[0] * y_p[1] - y_p[0] * x_p[1]) * (x_p[2] - x_p[3]) -
               (x_p[0] - x_p[1]) * (x_p[2] * y_p[3] - y_p[2] * x_p[3])) / \
              ((x_p[0] - x_p[1]) * (y_p[2] - y_p[3]) - (y_p[0] - y_p[1]) * (x_p[2] - x_p[3]))

        yis = ((x_p[0] * y_p[1] - y_p[0] * x_p[1]) * (y_p[2] - y_p[3]) -
               (y_p[0] - y_p[1]) * (x_p[2] * y_p[3] - y_p[2] * x_p[3])) / \
              ((x_p[0] - x_p[1]) * (y_p[2] - y_p[3]) - (y_p[0] - y_p[1]) * (x_p[2] - x_p[3]))
    return (xis, yis)


def simplify_polygon(in_poly, tol=0.05, maxiter=None, segment_to_remove=None):
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
        d_in = np.sqrt(np.ediff1d(x_in, to_end=x_in[0]-x_in[-1]) ** 2.
                       + np.ediff1d(y_in, to_end=y_in[0]-y_in[-1]) ** 2.)
        d_tot = np.sum(d_in)
        if segment_to_remove is not None:
            k = segment_to_remove
        else:
            k = np.argmin(d_in)
        if d_in[k] < tol * d_tot:
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
    quads = []
    len_poly = len(x_s)
    for i in range(0, len_poly):
        for j in range(i + 1, len_poly):
            ind = generate_point_indices(i, j, len_poly)
            (x_ij, y_ij) = line_intersection(x_s[ind], y_s[ind])
            for k in range(j + 1, len_poly):
                ind = generate_point_indices(j, k, len_poly)
                (x_jk, y_jk) = line_intersection(x_s[ind], y_s[ind])
                for l in range(k + 1, len_poly):
                    ind = generate_point_indices(k, l, len_poly)
                    (x_kl, y_kl) = line_intersection(x_s[ind], y_s[ind])
                    ind = generate_point_indices(l, i, len_poly)
                    (x_li, y_li) = line_intersection(x_s[ind], y_s[ind])

                    xis = np.asarray([x_ij, x_jk, x_kl, x_li])
                    yis = np.asarray([y_ij, y_jk, y_kl, y_li])

                    if (np.sum(np.isnan(xis)) + np.sum(np.isnan(yis))) > 0:
                        # no intersection point for some of the lines
                        pass
                    else:
                        xis_ave = np.average(xis)
                        yis_ave = np.average(yis)
                        xis = xis_ave + 1.0001 * (xis - xis_ave)
                        yis = yis_ave + 1.0001 * (yis - yis_ave)
                        (xis, yis) = order_polygon_points(xis, yis)
                        enclose = True
                        quad = Polygon([(xis[0], yis[0]),
                                        (xis[1], yis[1]),
                                        (xis[2], yis[2]),
                                        (xis[3], yis[3])])
                        for x_i, y_i in zip(x_s, y_s):
                            point = Point(x_i, y_i)
                            if (not quad.intersects(point) and not quad.touches(point)):
                                enclose = False
                        if enclose:
                            quads.append(quad)
    return quads


def get_bounding_quad(hull, visual=False):
    """
    Returns the minimum area quadrilateral that contains (bounds)
    the convex hull (openCV format) given as input.
    """
    hull_poly = Polygon([[x, y]
                         for (x, y) in zip(hull[:, 0, 0], hull[:, 0, 1])])
    simple_poly = simplify_polygon(hull_poly)
    if visual:
        plt.plot(hull[:, 0, 0], hull[:, 0, 1], 's-')
        plt.plot(simple_poly)
        plt.title('hull and simplified')
        plt.show()
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
    region_size (param) determines the region around the corner where the comparison is done.
    """
    bquad_corners = np.zeros((4, 2))
    bquad_corners[:, 0] = np.asarray(bquad_poly.exterior.coords)[:-1, 0]
    bquad_corners[:, 1] = np.asarray(bquad_poly.exterior.coords)[:-1, 1]

    interior_x = np.average(
        bquad_corners[:, 0]) + region_size * (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))
    interior_y = np.average(
        bquad_corners[:, 1]) + region_size * (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p0_x = interior_x + (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p1_x = interior_x - (bquad_corners[:, 1] - np.average(bquad_corners[:, 1]))
    p0_y = interior_y - (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))
    p1_y = interior_y + (bquad_corners[:, 0] - np.average(bquad_corners[:, 0]))

    corner_area_polys = []
    for i in range(len(interior_x)):
        bline = LineString([(p0_x[i], p0_y[i]), (p1_x[i], p1_y[i])])
        corner_area_polys.append(Polygon([bquad_poly.intersection(bline).coords[0],
                                          bquad_poly.intersection(
                                              bline).coords[1],
                                          (bquad_corners[i, 0], bquad_corners[i, 1])]))

    hull_corner_area = 0
    quad_corner_area = 0
    for capoly in corner_area_polys:
        quad_corner_area += capoly.area
        hull_corner_area += capoly.intersection(hull_poly).area

    diff = 1. - hull_corner_area/quad_corner_area
    return diff


class CardCandidate:
    """
    Class representing a segment of the image that may be a recognizable card.
    """

    def __init__(self, im_seg, card_cnt, bquad, fraction):
        self.image = im_seg
        self.contour = card_cnt
        self.bounding_quad = bquad
        self.is_recognized = False
        self.is_fragment = False
        self.image_area_fraction = fraction
        self.name = 'unknown'

    def contains(self, other):
        """
        Returns whether the bounding polygon of the card candidate
        contains the bounding polygon of the other candidate.
        """
        return other.bounding_quad.within(self.bounding_quad)


class MTGCardDetector:
    """
    MTG card detector class.
    """

    def __init__(self):
        self.ref_img_clahe = []
        self.ref_filenames = []
        self.test_img_clahe = []
        self.test_filenames = []
        self.warped_list = []
        self.card_list = []
        self.bounding_poly_list = []

        self.candidate_list = []

        self.phash_ref = []

        self.verbose = False
        self.visual = True

        self.hash_separation_thr = 4.
        self.thr_lvl = 70

        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid_size = (8, 8)

    def polygon_form_factor(self, poly):
        """
        The ratio between the polygon area and circumference length,
        scaled by the length of the shortest segment.
        """
        # minimum side length
        d_0 = np.amin(np.sqrt(np.sum(np.diff(np.asarray(poly.exterior.coords),
                                             axis=0) ** 2., axis=1)))
        return poly.area/(poly.length * d_0)

    def read_and_adjust_images(self, path):
        """
        Reads image files and performs adaptive histogram equalization.
        """

        print('Reading images from ' + str(path))
        filenames = glob.glob(path + '*.jpg')
        imgs = []
        img_names = []
        imgs_clahe = []
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit,
                                tileGridSize=self.clahe_tile_grid_size)

        for file_name in filenames:
            imgs.append(cv2.imread(file_name))
            img_names.append(file_name.split(path)[1])

        for img in imgs:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lightness, redness, yellowness = cv2.split(lab)
            corrected_lightness = clahe.apply(lightness)
            limg = cv2.merge((corrected_lightness, redness, yellowness))
            imgs_clahe.append(cv2.cvtColor(limg, cv2.COLOR_LAB2BGR))

        return (imgs_clahe, img_names)

    def read_and_adjust_reference_images(self, path):
        """
        Reads and histogram-adjusts the reference image set.
        Pre-calculates the hashes of the images.
        """

        (self.ref_img_clahe, self.ref_filenames) = self.read_and_adjust_images(path)
        print('Hashing reference set.')
        for i in range(len(self.ref_img_clahe)):
            self.phash_ref.append(imagehash.phash(PILImage.fromarray(np.uint8(
                255 * cv2.cvtColor(self.ref_img_clahe[i], cv2.COLOR_BGR2RGB))), hash_size=32))

    def read_and_adjust_test_images(self, path):
        """
        Reads and histogram-adjusts the reference image set.
        """

        (self.test_img_clahe, self.test_filenames) = self.read_and_adjust_images(path)

    def segment_image(self, full_image):
        """
        Segments the given image into card candidates, that is,
        regions of the image that have a high chance
        of containing a recognizable card.
        """

        self.candidate_list.clear()

        warped_list = []
        card_list = []
        bounding_poly_list = []

        image_area = full_image.shape[0] * full_image.shape[1]
        lca = 0.01  # largest card area

        # grayscale transform, thresholding, countouring and contour sorting by area
        gray = cv2.cvtColor(full_image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.thr_lvl, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(
            np.uint8(thresh), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_s = sorted(contours, key=cv2.contourArea, reverse=True)

        for card in contours_s:
            hull = cv2.convexHull(card)
            phull = Polygon([[x, y]
                             for (x, y) in zip(hull[:, :, 0], hull[:, :, 1])])
            if phull.area < 0.7 * lca:
                # break after card size range has been explored
                break
            bounding_poly = get_bounding_quad(hull)

            qc_diff = quad_corner_diff(phull, bounding_poly)
            if self.verbose:
                print('Quad corner diff = ' + str(qc_diff))

            if qc_diff > 0.05:
                # typically indicates contour around the rounded card corners
                crop_pct = qc_diff * 22.
                if self.verbose:
                    print('Cropping by ' + str(crop_pct) + ' %')
            else:
                crop_pct = 0.

            scale_factor = (1.-crop_pct/100.)
            bounding_poly_cropped = scale(bounding_poly, xfact=scale_factor,
                                          yfact=scale_factor, origin='centroid')

            warped = four_point_transform(full_image, bounding_poly_cropped)

            form_factor = self.polygon_form_factor(bounding_poly)

            if self.verbose:
                print('Image area test (x < 0.99):')
                print('x = ' + str(bounding_poly.area/image_area))
                print('Shape test (0.27 < x 0.32):')
                print('x = ' + str(form_factor))
                print('Card area test (x > 0.7):')
                print('x = ' + str(bounding_poly.area/lca))

            if bounding_poly.area < image_area * 0.99:  # not the whole image test
                if qc_diff < 0.35:  # corner shape test
                    if form_factor > 0.27 and form_factor < 0.32:  # card shape test
                        if bounding_poly.area > 0.7 * lca:  # card equal size test
                            if lca == 0.01:
                                lca = bounding_poly.area
                            warped_list.append(warped)
                            card_list.append(card)
                            bounding_poly_list.append(bounding_poly)
                            print(str(len(card_list)) + ' cards segmented.')

        for im_seg, card_cnt, bquad in zip(warped_list,
                                           card_list,
                                           bounding_poly_list):
            self.candidate_list.append(CardCandidate(im_seg,
                                                     card_cnt,
                                                     bquad,
                                                     bquad.area/image_area))

    def phash_compare(self, im_seg):
        """
        Runs perceptive hash comparison between given image and
        the (pre-hashed) reference set.
        """

        card_name = 'unknown'
        is_recognized = False
        rotations = np.array([0., 90., 180., 270.])

        d_0_dist = np.zeros(len(rotations))
        d_0 = np.zeros((len(self.ref_img_clahe), len(rotations)))
        for j, rot in enumerate(rotations):
            phash_im = imagehash.phash(PILImage.fromarray(
                np.uint8(255 * cv2.cvtColor(rotate(im_seg, rot),
                                            cv2.COLOR_BGR2RGB))), hash_size=32)
            for i in range(len(d_0)):
                d_0[i, j] = phash_im - self.phash_ref[i]
            d_0_ = d_0[:, j][d_0[:, j] > np.amin(d_0[:, j])]
            d_0_ave = np.average(d_0_)
            d_0_std = np.std(d_0_)
            d_0_dist[j] = (d_0_ave - np.amin(d_0[:, j]))/d_0_std
            if self.verbose:
                print('Phash statistical distance' + str(d_0_dist[j]))
            if (d_0_dist[j] > self.hash_separation_thr and
                    np.argmax(d_0_dist) == j):
                card_name = self.ref_filenames[np.argmin(d_0[:, j])]\
                            .split('.jpg')[0]
                is_recognized = True
                break
        return (is_recognized, card_name)

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

        full_image = self.test_img_clahe[image_index].copy()

        if self.visual:
            print('Original image')
            plt.imshow(cv2.cvtColor(self.test_img_clahe[image_index],
                                    cv2.COLOR_BGR2RGB))
            plt.show()

        print('Segmentation of art')
        self.segment_image(full_image)

        if self.visual:
            plt.show()

        print('Recognition')

        iseg = 0
        plt.imshow(cv2.cvtColor(full_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        for candidate in self.candidate_list:
            im_seg = candidate.image
            bquad = candidate.bounding_quad

            iseg += 1
            print(str(iseg) + " / " + str(len(self.candidate_list)))

            for other_candidate in self.candidate_list:
                if (other_candidate.is_recognized and
                        not other_candidate.is_fragment):
                    if other_candidate.contains(candidate):
                        candidate.is_fragment = True
            if not candidate.is_fragment:
                bquad_corners = np.empty((4, 2))
                bquad_corners[:, 0] = np.asarray(bquad.exterior.coords)[:-1, 0]
                bquad_corners[:, 1] = np.asarray(bquad.exterior.coords)[:-1, 1]

                (candidate.is_recognized,
                 candidate.name) = self.recognize_segment(im_seg)

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
                    str(self.test_filenames[image_index].split('.jpg')[0]) +
                    '.jpg', dpi=600, bbox='tight')
        if self.visual:
            plt.show()


def main():
    """
    Python MTG Card Detector.
    Example run.
    Can be used also purely through the defined classes.
    """

    # Instantiate the detector
    card_detector = MTGCardDetector()

    # Read the reference and test data sets
    card_detector.read_and_adjust_reference_images(
        '../../MTG/Card_Images/LEA/')
    card_detector.read_and_adjust_test_images('../MTG_alpha_test_images/')

    # Start up the profiler.
    profiler = cProfile.Profile()
    profiler.enable()

    # Run the card detection and recognition.
    for im_ind in range(0, 7):
        card_detector.run_recognition(im_ind)

    # Stop profiling and organize and print profiling results.
    profiler.disable()
    profiler_stream = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    profiler_stats = pstats.Stats(
        profiler, stream=profiler_stream).sort_stats(sortby)
    profiler_stats.print_stats(20)
    print(profiler_stream.getvalue())


if __name__ == "__main__":
    main()
