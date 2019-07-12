import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
from sklearn.decomposition import PCA
from shapely.geometry import Point, LineString
from shapely.geometry.polygon import Polygon
from shapely.affinity import scale
import itertools
import copy
from  scipy import interpolate
from scipy.ndimage import rotate

import cProfile, pstats, io

from PIL import Image as PILImage
import imagehash

import PIL.Image

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect


def four_point_transform(image, poly):
    pts = np.zeros((4,2))
    pts[:,0] = np.asarray(poly.exterior.coords)[:-1,0]
    pts[:,1] = np.asarray(poly.exterior.coords)[:-1,1]
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")
 
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

def order_polygon_points(x,y):
    a = np.arctan2(y - np.average(y) ,x - np.average(x))
    ind = np.argsort(a)
    return (x[ind], y[ind])

def line_intersection(x1,x2,x3,x4,y1,y2,y3,y4):
    if ((x1-x2)*(y3-y4) == (y1-y2)*(x3-x4)):
        # parallel lines
        xis = np.nan
        yis = np.nan
    else:
        xis = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4) )    
        yis = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4) )
    return (xis,yis)

def simplify_polygon(in_poly, tol = 0.05, maxiter = None, segmentToRemove = None):
    
    #x = xy[:,0].astype(float)
    #y = xy[:,1].astype(float)
    x = np.asarray(in_poly.exterior.coords)[:-1,0]
    y = np.asarray(in_poly.exterior.coords)[:-1,1]
    N = len(x)
    niter = 0
    if segmentToRemove is not None:
        maxiter = 1
    while(N > 4):
        d = np.sqrt(np.ediff1d(x, to_end=x[0]-x[-1])**2. + np.ediff1d(y, to_end=y[0]-y[-1])**2.)
        d_tot = np.sum(d)
        if segmentToRemove is not None:
            k = segmentToRemove
        else:
            k = np.argmin(d)
        if (d[k] < tol * d_tot):
            xis,yis = line_intersection(x[(k-1)%N],x[(k)%N],x[(k+1)%N],x[(k+2)%N],
                                                y[(k-1)%N],y[(k)%N],y[(k+1)%N],y[(k+2)%N])
            x[k] = xis
            y[k] = yis
            x = np.delete(x,(k+1)%N)
            y = np.delete(y,(k+1)%N)
            N = len(x)
            niter += 1
            if (maxiter is not None) and (niter >= maxiter):
                break
        else:
            break
    
    out_poly = Polygon([[ix, iy] for (ix,iy) in zip(x,y)])

    #xy_ = np.zeros((len(x),2))
    #xy_[:,0] = x
    #xy_[:,1] = y
    #return xy_

    return out_poly

def generate_quad_candidates(in_poly):
    
    x_unsorted = np.asarray(in_poly.exterior.coords)[:-1,0]
    y_unsorted = np.asarray(in_poly.exterior.coords)[:-1,1]
    # make sure that the points are ordered
    (x,y) = order_polygon_points(x_unsorted,y_unsorted)
    quads = []
    N = len(x)
    for i in range(0,N):
        for j in range(i+1,N):
            (x_ij, y_ij) = line_intersection(x[i%N], x[(i+1)%N],
                                                   x[j%N], x[(j+1)%N],
                                                   y[i%N], y[(i+1)%N],
                                                   y[j%N], y[(j+1)%N])
            for k in range(j+1,N):
                (x_jk, y_jk) = line_intersection(x[j%N], x[(j+1)%N],
                                                       x[k%N], x[(k+1)%N],
                                                       y[j%N], y[(j+1)%N],
                                                       y[k%N], y[(k+1)%N])
                for m in range(k+1,N):
                    (x_km, y_km) = line_intersection(x[k%N], x[(k+1)%N],
                                                           x[m%N], x[(m+1)%N],
                                                           y[k%N], y[(k+1)%N],
                                                           y[m%N], y[(m+1)%N])
                    (x_mi, y_mi) = line_intersection(x[m%N], x[(m+1)%N],
                                                           x[i%N], x[(i+1)%N],
                                                           y[m%N], y[(m+1)%N],
                                                           y[i%N], y[(i+1)%N])
                    xis = np.asarray([x_ij, x_jk, x_km, x_mi])
                    yis = np.asarray([y_ij, y_jk, y_km, y_mi])
                    
                    if( (np.sum(np.isnan(xis)) + np.sum(np.isnan(yis))) > 0):
                        #no intersection point for some of the lines - no quadrilateral
                        pass
                    else:
                        xis_ave = np.average(xis)
                        yis_ave = np.average(yis)
                        for ii in range(len(xis)):
                            xis[ii] = xis_ave + 1.0001 * (xis[ii] - xis_ave)
                            yis[ii] = yis_ave + 1.0001 * (yis[ii] - yis_ave)
                        (xis,yis) = order_polygon_points(xis,yis)
                        enclose = True
                        quad = Polygon([(xis[0], yis[0]),
                                        (xis[1], yis[1]),
                                        (xis[2], yis[2]),
                                        (xis[3], yis[3])])
                        for ip in range(N):
                            point = Point(x[ip], y[ip])
                            if (not quad.intersects(point) and not quad.touches(point)):
                                enclose = False
                        if(enclose):
                            quads.append(quad)
    return quads
    
def get_bounding_quad(hull, visual = False):
    hull_poly = Polygon([[x, y] for (x,y) in zip(hull[:,0,0],hull[:,0,1])])
    simple_poly = simplify_polygon(hull_poly)    
    #xy = np.transpose(np.asarray([hull[:,0,0], hull[:,0,1]]))
    #xy = simplify_polygon(xy)
    if visual:
        plt.plot(hull[:,0,0],hull[:,0,1],'s-')
        plt.plot(simple_poly)
        plt.title('hull and simplified')
        plt.show()
    bounding_quads = generate_quad_candidates(simple_poly)
    bquad_areas = np.zeros(len(bounding_quads))
    for iq, bquad in enumerate(bounding_quads):
        bquad_areas[iq] = bquad.area
    minAreaQuad = bounding_quads[np.argmin(bquad_areas)]
    #bounding_corners_x = np.asarray(minAreaQuad.exterior.coords)[:-1,0]
    #bounding_corners_y = np.asarray(minAreaQuad.exterior.coords)[:-1,1]
    
    #return np.transpose(np.asarray([bounding_corners_x, bounding_corners_y]))
    return minAreaQuad


def quad_corner_diff(hull_poly, bquad_poly, r = 0.9):
    bquad_corners = np.zeros((4,2))
    bquad_corners[:,0] = np.asarray(bquad_poly.exterior.coords)[:-1,0]
    bquad_corners[:,1] = np.asarray(bquad_poly.exterior.coords)[:-1,1]

    #hull = cv2.convexHull(card_cnt)
    
    interior_x = np.average(bquad_corners[:,0]) + r * (bquad_corners[:,0] - np.average(bquad_corners[:,0]))
    interior_y = np.average(bquad_corners[:,1]) + r * (bquad_corners[:,1] - np.average(bquad_corners[:,1]))
    p0_x = interior_x + (bquad_corners[:,1] - np.average(bquad_corners[:,1]))
    p1_x = interior_x - (bquad_corners[:,1] - np.average(bquad_corners[:,1]))
    p0_y = interior_y - (bquad_corners[:,0] - np.average(bquad_corners[:,0]))
    p1_y = interior_y + (bquad_corners[:,0] - np.average(bquad_corners[:,0]))

    #bquad_poly = Polygon([[x, y] for (x,y) in zip(bquad_corners[:,0],bquad_corners[:,1])])
    #hull_poly = Polygon([[x, y] for (x,y) in zip(hull[:,0,0],hull[:,0,1])])
    
    corner_area_polys = []
    for i in range(len(interior_x)):
        bline = LineString([(p0_x[i], p0_y[i]),(p1_x[i], p1_y[i])])
        corner_area_polys.append(Polygon([bquad_poly.intersection(bline).coords[0],
                                         bquad_poly.intersection(bline).coords[1],
                                         (bquad_corners[i,0],bquad_corners[i,1])]))
 
    hull_corner_area = 0
    quad_corner_area = 0
    for capoly in corner_area_polys:
        quad_corner_area += capoly.area
        hull_corner_area += capoly.intersection(hull_poly).area
    
    diff = 1. - hull_corner_area/quad_corner_area
    return diff

class CardCandidate:

    def __init__(self, im_seg, card_cnt, bquad, fraction):
        self.image = im_seg
        self.contour = card_cnt
        self.bounding_quad = bquad
        self.recognized = False
        self.image_area_fraction = fraction

    def contains(self, other):
        return other.bounding_quad.within(self.bounding_quad)

class MTGCardDetector:

    def __init__(self):
        self.ref_img_clahe = []
        self.ref_filenames = []
        self.test_img_clahe = []
        self.test_filenames = []
        self.warpedlist = []
        self.cardlist = []
        self.bPolylist = []

        self.candidateList = []

        self.phash_ref = []



        self.verbose = False
        self.visual = True

        self.hash_separation_thr = 4.
        self.thr_lvl = 70

    def polygon_form_factor(self,p):
        #minimum side length
        d0 = np.amin(np.sqrt(np.sum(np.diff(np.asarray(p.exterior.coords),axis=0)**2.,axis=1)))
        return p.area/(p.length * d0)

    def read_and_adjust_images(self,path):
        print('Reading images from ' + str(path))
        filenames = glob.glob(path + '*.jpg')
        imgs = []
        img_names = []
        imgs_clahe = []
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

        for fn in filenames:
            imgs.append(cv2.imread(fn))
            img_names.append(fn.split(path)[1])

        for img in imgs:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            imgs_clahe.append(cv2.cvtColor(limg, cv2.COLOR_LAB2BGR))

        return (imgs_clahe, img_names)

    def read_and_adjust_reference_images(self, path):
        (self.ref_img_clahe, self.ref_filenames) = self.read_and_adjust_images(path)
        print('Hashing reference set.')
        for i in range(len(self.ref_img_clahe)):
            self.phash_ref.append(imagehash.phash(PILImage.fromarray(np.uint8(255 * cv2.cvtColor(self.ref_img_clahe[i],cv2.COLOR_BGR2RGB))),hash_size=32))

    def read_and_adjust_test_images(self, path):
        (self.test_img_clahe, self.test_filenames) = self.read_and_adjust_images(path)

    def segment_image(self, im):

        warpedlist = []
        cardlist = []
        bPolylist = []
        
        image_area = im.shape[0]*im.shape[1]
        lca = 0.01 #largest card area
        
        #grayscale transform, thresholding, countouring and contour sorting by area
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self.thr_lvl, 255, cv2.THRESH_BINARY)
        image,contours, hierarchy = cv2.findContours(np.uint8(thresh),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours_s = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for card in contours_s:
            hull = cv2.convexHull(card)
            phull = Polygon([[x, y] for (x,y) in zip(hull[:,:,0],hull[:,:,1])])
            if(phull.area < 0.7 * lca): # break after card size range has been explored
                break
            bPoly = get_bounding_quad(hull)
                        
            qc_diff = quad_corner_diff(phull, bPoly)
            if self.verbose:
                print('Quad corner diff = ' + str(qc_diff))

            if (qc_diff > 0.05): # typically indicates contour around the rounded card corners
                crop_pct = qc_diff * 22.
                if self.verbose:
                    print('Cropping by ' + str(crop_pct) + ' %')
            else:
                crop_pct = 0.

            scale_factor = (1.-crop_pct/100.)
            bPolyCropped = scale(bPoly, xfact=scale_factor, yfact=scale_factor, origin='centroid')
            
            warped = four_point_transform(im, bPolyCropped)
            
            ff = self.polygon_form_factor(bPoly)
            
            if self.verbose:
                print('Image area test (x < 0.99):')
                print('x = ' + str(bPoly.area/image_area))
                print('Shape test (0.27 < x 0.32):')
                print('x = ' + str(ff))
                print('Card area test (x > 0.7):')
                print('x = ' + str(bPoly.area/lca))

            if (bPoly.area < image_area * 0.99): # not the whole image test
                if (qc_diff < 0.35): # corner shape test
                    if (ff > 0.27 and ff < 0.32): #card shape test
                        if (bPoly.area > 0.7 * lca): # card equal size test
                            if (lca == 0.01):
                                lca = bPoly.area
                            warpedlist.append(warped)
                            cardlist.append(card)
                            bPolylist.append(bPoly)
                            print(str(len(cardlist)) + ' cards segmented.')


        for im_seg, card_cnt, bquad in zip(warpedlist, cardlist, bPolylist):
            self.candidateList.append(CardCandidate(im_seg, card_cnt, bquad, bquad.area/image_area))

        """double_segmented = []
        for ip, iPoly in enumerate(self.bPolylist):
            for jp, jPoly in enumerate(self.bPolylist):
                if (iPoly.within(jPoly) and ip != jp):
                    double_segmented.append(ip)
        print('Removing ' + str(len(double_segmented)) + ' duplicates.')
        for ids in reversed(double_segmented):
            del self.warpedlist[ids]
            del self.cardlist[ids]
            del self.bPolylist[ids]
         """   
        #return warpedlist, cardlist, bPolylist

    def phash_compare(self, im_seg):
        card_name = 'unknown'
        d0_dist = np.zeros(4)
        d0 = np.zeros((len(self.ref_img_clahe),4))
        for j in range(4):
            im_seg_rot = rotate(im_seg,j * 90)
            phash_im = imagehash.phash(PILImage.fromarray(np.uint8(255 * cv2.cvtColor(im_seg_rot,cv2.COLOR_BGR2RGB))),hash_size=32)
            for i in range(len(d0)):
                d0[i,j] = phash_im - self.phash_ref[i]        
            d0_ = d0[:,j][d0[:,j] > np.amin(d0[:,j])]
            d0_ave = np.average(d0_)
            d0_std = np.std(d0_)
            d0_dist[j] = (d0_ave - np.amin(d0[:,j]))/d0_std
            if self.verbose:
                print('Phash statistical distance' + str(d0_dist[j]))
            if (d0_dist[j] > self.hash_separation_thr and np.argmax(d0_dist) == j):
                card_name = self.ref_filenames[np.argmin(d0[:,j])].split('.jpg')[0]
                break
        return card_name

    

    def run_recognition(self, im_index):

        im = self.test_img_clahe[im_index].copy()

        if(self.visual):
            print('Original image')
            plt.imshow(cv2.cvtColor(self.test_img_clahe[im_index], cv2.COLOR_BGR2RGB))
            #plt.gcf().set_size_inches(5,5)
            plt.show()

        print('Segmentation of art')
        self.segment_image(im)

        if(self.visual):
            plt.show()
        
        print('Recognition')

        iseg = 0
        plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        #for im_seg, card_cnt, bquad in zip(self.warpedlist, self.cardlist, self.bPolylist):
        for candidate in self.candidateList:
            im_seg = candidate.image
            bquad = candidate.bounding_quad

            #ITERATE OVER RECOGNIZED CANDIDATES TO SEE IF THEY ALREADY CONTAIN THE POLYGON
            iseg += 1
            print(str(iseg) + " / " + str(len(self.candidateList)))
            bquad_corners = np.empty((4,2))
            bquad_corners[:,0] = np.asarray(bquad.exterior.coords)[:-1,0]
            bquad_corners[:,1] = np.asarray(bquad.exterior.coords)[:-1,1]
            
            card_name = self.phash_compare(im_seg)

            plt.plot(np.append(bquad_corners[:,0],bquad_corners[0,0]),
                    np.append(bquad_corners[:,1],bquad_corners[0,1]), 'g-' )
            bPoly = Polygon([[x, y] for (x,y) in zip(bquad_corners[:,0],bquad_corners[:,1])])
            fntsze = int(4 * bPoly.length / im.shape[1])
            plt.text(np.average(bquad_corners[:,0]),np.average(bquad_corners[:,1]),
                    card_name, horizontalalignment='center', fontsize = fntsze,
                    bbox=dict(facecolor='white', alpha=0.7))
        plt.savefig('results/MTG_card_recognition_results_'+str(self.test_filenames[im_index].split('.jpg')[0]) +'.jpg',dpi=600,bbox='tight')
        if(self.visual):
            plt.show()




def main():
    
    
    card_detector = MTGCardDetector()

    card_detector.read_and_adjust_reference_images('../../MTG/Card_Images/LEA/')
    card_detector.read_and_adjust_test_images('../MTG_alpha_test_images/')

    pr = cProfile.Profile()
    pr.enable()

    
    card_detector.run_recognition(1)
    #card_detector.run_recognition(2)

    pr.disable()
    s = io.StringIO()
    sortby = pstats.SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats(10)
    print(s.getvalue())

if __name__ == "__main__":
    main()

