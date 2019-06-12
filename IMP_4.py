# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged


import numpy as np
import os
import matplotlib.pyplot as plt
import shutil

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter, convolve
from scipy.ndimage import label, center_of_mass, map_coordinates
from scipy.misc import imsave

from skimage.color import rgb2gray
from scipy import signal

K = 0.04

import sol4_utils
# from . import sol4_utils


# todo is it allowed to import sol4_utils as..sol4_u?

def harris_corner_detector(im):
    """
    Detects harris corners.
    Make sure the returned coordinates are x major!!!
    :param im: A 2D array representing an image.
    :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
    """

    conv_vec = np.array([[1, 0, -1]])

    x_der = convolve(im, conv_vec)
    y_der = convolve(im, conv_vec.transpose())

    x_der_2 = sol4_utils.blur_spatial(x_der * x_der, 3)

    y_der_2 = sol4_utils.blur_spatial(y_der*y_der, 3)
    x_y_der = sol4_utils.blur_spatial(x_der*y_der, 3)
    y_x_der = sol4_utils.blur_spatial(y_der*x_der, 3)
    response = (x_der_2 * y_der_2 - x_y_der * y_x_der) - K * (x_der_2 + y_der_2) ** 2
    bool_response = non_maximum_suppression(response)
    coor_arr = np.where(bool_response)
    coor_arr = [coor_arr[1], coor_arr[0]]
    coor_arr = np.column_stack(coor_arr)

    return coor_arr

def sample_descriptor(im, pos, desc_rad):
    """
    Samples descriptors at the given corners.
    :param im: A 2D array representing an image.
    :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
    :param desc_rad: "Radius" of descriptors to compute.
    :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
    """


    descriptor_arr = np.zeros([pos.shape[0], 2 * desc_rad + 1, 2 * desc_rad + 1])

    for i in range(pos.shape[0]):
        x_vals = np.arange(pos[i][0] - desc_rad, pos[i][0] + desc_rad + 1)
        y_vals = np.arange(pos[i][1] - desc_rad, pos[i][1] + desc_rad + 1)
        grid = np.meshgrid(x_vals,y_vals)
        desc = map_coordinates(im,[grid[1],grid[0]],order=1,prefilter=False)
        if np.sum(desc) != 0:
            desc = (desc-np.mean(desc))/np.linalg.norm(desc-np.mean(desc))
        descriptor_arr[i] = desc.transpose()
    return descriptor_arr

def find_features(pyr):
    """
    Detects and extracts feature points from a pyramid.
    :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
    :return: A list containing:
                1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                   These coordinates are provided at the pyramid level pyr[0].
                2) A feature descriptor array with shape (N,K,K)
    """

    feature_lst = spread_out_corners(pyr[0], 3, 3, 3)
    descriptors = sample_descriptor(pyr[2],feature_lst/4,3)
    return [feature_lst,descriptors]


def match_features(desc1, desc2, min_score):
    """
    Return indices of matching descriptors.
    :param desc1: A feature descriptor array with shape (N1,K,K).
    :param desc2: A feature descriptor array with shape (N2,K,K).
    :param min_score: Minimal match score.
    :return: A list containing:
                1) An array with shape (M,) and dtype int of matching indices in desc1.
                2) An array with shape (M,) and dtype int of matching indices in desc2.
    """

    match = []
    flat_des1 = desc1.reshape(desc1.shape[0],desc1.shape[1] * desc1.shape[2])
    flat_des2 = desc2.reshape(desc2.shape[0], desc2.shape[1] * desc2.shape[2])
    combine = np.dot(flat_des1,flat_des2.transpose())

    for i in range (combine.shape[0]):
        max_index_row = combine[i].argsort()[-2:]
        for j in max_index_row:
            max_index_col = combine[:,j].argsort()[-2:]
            if i in max_index_col and combine[i][j] >= min_score:
                match.append([i,j])

    match = np.array(match)
    return [match.transpose()[0],match.transpose()[1]]  #todo should be list?

def apply_homography(pos1, H12):
    """
    Apply homography to inhomogenous points.
    :param pos1: An array with shape (N,2) of [x,y] point coordinates.
    :param H12: A 3x3 homography matrix.
    :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
    """

    ext_post1 = np.insert(pos1,[2],1,axis=1)
    ext_post1 = np.dot(H12,ext_post1.transpose()).transpose()
    h_pos1 = ext_post1[:,[0,1]]
    h_pos1 = (h_pos1.transpose() / ext_post1[:,2].transpose()).transpose()

    return h_pos1



def ransac_homography(points1, points2, num_iter, inlier_tol,translation_only=False):
    """
    Computes homography between two sets of points using RANSAC.
    :param pos1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
    :param pos2: An array with shape (N,2) containing N rows of q [x,y] coordinates of matched points in image 2.
    :param num_iter: Number of RANSAC iterations to perform.
    :param inlier_tol: inlier tolerance threshold.
    :param translation_only: see estimate rigid transform
    :return: A list containing:
                1) A 3x3 normalized homography matrix.
                2) An Array with shape (S,) where S is the number of inliers,
                    containing the indices in pos1/pos2 of the maximal set of inlier matches found.
    """

    coor = np.array([0,0])
    max_v = 0

    for i in range(num_iter):
        if translation_only:
            rand_coor = np.random.choice(points1.shape[0],1)
        else:
            rand_coor = np.random.choice(points1.shape[0], 2)

        mat_homog = estimate_rigid_transform(points1[rand_coor],points2[rand_coor],translation_only)
        points2_n = apply_homography(points1,mat_homog) - points2
        points2_n = np.sum(points2_n ** 2, axis=1)

        num_inlier = np.sum(points2_n<inlier_tol)
        if num_inlier > max_v:
            coor = rand_coor
            max_v = num_inlier


    best_match = coor
    mat_homog = estimate_rigid_transform(points1[best_match],points2[best_match])
    points2_n = points2 - apply_homography(points1, mat_homog)
    points2_n = np.sum(points2_n ** 2, axis=1)
    points2_n = np.where(points2_n < inlier_tol)[0]

    return [mat_homog,points2_n]  #todo should be np?


def display_matches(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """
    image = np.hstack((im1, im2))
    plt.imshow(image, 'gray')
    for i in range(points1.shape[0]):
        if i in inliers:
            plt.plot((points1[i][0], points2[i][0] + im1.shape[1]),
                     (points1[i][1], points2[i][1]), c='y', lw=.5,
                     ms=1, marker='.')
        else:
            plt.plot((points1[i][0], points2[i][0] + im1.shape[1]),
                     (points1[i][1], points2[i][1]), c='b', lw=.2,
                     ms=1, marker='.')
        plt.plot(points1[i][0], points1[i][1], color='red', marker='.',
                 markersize=2, lw=0)
        plt.plot(points2[i][0] + im1.shape[1], points2[i][1], color='red',
                 marker='.', markersize=2, lw=0)
    plt.show()



def display_matches2(im1, im2, points1, points2, inliers):
    """
    Dispalay matching points.
    :param im1: A grayscale image.
    :param im2: A grayscale image.
    :parma pos1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
    :param pos2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
    :param inliers: An array with shape (S,) of inlier matches.
    """

    # print(points1)
    # print(inliers)
    match = points1[inliers]
    # print('lala',match1)
    points2_n = points2.transpose() # changed here
    points2_n[0] = points2_n[0] +  im2.shape[1]
    match = np.insert(match,np.arange(1,match.shape[0]+1),points2_n.transpose()[inliers],axis=0)
    # print('match',match)

    index_non_match = np.delete(np.arange(points1.shape[0]),inliers)
    non_match = points1[index_non_match]
    non_match = np.insert(non_match, np.arange(non_match.shape[0]), points2[index_non_match], axis=0)

    # print(match.shape[0])
    # print(match)
    # print('gagagaga',match.transpose())
    # print(match.transpose()[0][2])
    for i in range(0,match.shape[0],2):
        plt.plot((match.transpose())[0][i:i+2], (match.transpose())[1][i:i+2], c='y', lw=.5, ms=1., marker='.', mew=3., mec='red',mfc='red' )
        # plt.plot(match.transpose()[0][i:i+2], match.transpose()[1][i:i+2], mfc='r', c='y', lw=.4, ms=10, marker='o')
    for i in range(0,non_match.shape[0],2):
        plt.plot(non_match.transpose()[0][i:i+2], non_match.transpose()[1][i:i+2],c='b', lw=.5, ms=1., marker='.')

    plt.imshow(np.hstack([im1,im2]),cmap='gray')
    plt.show()





def accumulate_homographies(H_succesive, m):
    """
    Convert a list of succesive homographies to a
    list of homographies to a common reference frame.
    :param H_successive: A list of M-1 3x3 homography
      matrices where H_successive[i] is a homography which transforms points
      from coordinate system i to coordinate system i+1.
    :param m: Index of the coordinate system towards which we would like to
      accumulate the given homographies.
    :return: A list of M 3x3 homography matrices,
      where H2m[i] transforms points from coordinate system i to coordinate system m
    """

    # todo think about indexes problems with that m == M//2


    H_to_ret = H_succesive.copy()
    H_to_ret.insert(m,np.eye(3))
    # todo check that it is a list and not np array

    for i in range(m-1):
        H_to_ret[i] = np.linalg.multi_dot(H_succesive[i:m][::-1])
        H_to_ret[i] = H_to_ret[i]/H_to_ret[i][2][2]

    for i in range(m+1,len(H_to_ret)):
        H_to_ret[i] = np.linalg.inv(H_to_ret[i])

    for i in range(m+2,len(H_to_ret)):
        H_to_ret[i] = np.linalg.multi_dot(H_to_ret[i-1:i+1])
        H_to_ret[i] = H_to_ret[i]/H_to_ret[i][2][2]

    return H_to_ret

def compute_bounding_box(homography, w, h):
    """
    computes bounding box of warped image under homography, without actually warping the image
    :param homography: homography
    :param w: width of the image
    :param h: height of the image
    :return: 2x2 array, where the first row is [x,y] of the top left corner,
     and the second row is the [x,y] of the bottom right corner
    """

    # todo in what step to do int??
    # todo should i do -1? if so why?


    all_corners = np.array([[0, 0], [w, 0], [0, h], [w, h]])
    all_corners = apply_homography(all_corners,homography)
    top_left = [np.min(all_corners[:,0]),np.min(all_corners[:,1])]
    bottom_right = [np.max(all_corners[:,0]),np.max(all_corners[:,1])]
    return np.array([top_left,bottom_right]).astype(np.int)

def warp_channel(image, homography):
    """
    Warps a 2D image with a given homography.
    :param image: a 2D image.
    :param homography: homograhpy.
    :return: A 2d warped image.
    """


    boundings = compute_bounding_box(homography,image.shape[1],image.shape[0])
    x_vals = np.arange(boundings[0][0], boundings[1][0])
    y_vals = np.arange(boundings[0][1], boundings[1][1])
    x_vals,y_vals = np.meshgrid(x_vals,y_vals)

    x_vals_f, y_vals_f = x_vals.flatten(),y_vals.flatten()
    coor_lst = np.array([x_vals_f.tolist(),y_vals_f.tolist()])
    inv_hom = np.linalg.inv(homography)
    new_coor = apply_homography(coor_lst.transpose(),inv_hom).reshape(x_vals.shape[0],x_vals.shape[1],2)
    warped_im = map_coordinates(image,[new_coor[:, :, 1], new_coor[:, :, 0]],order=1,prefilter=False)
    return warped_im



def warp_image(image, homography):
  """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
  return np.dstack([warp_channel(image[...,channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
  """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
  translation_over_thresh = [0]
  last = homographies[0][0,-1]
  for i in range(1, len(homographies)):
    if homographies[i][0,-1] - last > minimum_right_translation:
      translation_over_thresh.append(i)
      last = homographies[i][0,-1]
  return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
  """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
  centroid1 = points1.mean(axis=0)
  centroid2 = points2.mean(axis=0)

  if translation_only:
    rotation = np.eye(2)
    translation = centroid2 - centroid1

  else:
    centered_points1 = points1 - centroid1
    centered_points2 = points2 - centroid2

    sigma = centered_points2.T @ centered_points1
    U, _, Vt = np.linalg.svd(sigma)

    rotation = U @ Vt
    translation = -rotation @ centroid1 + centroid2

  H = np.eye(3)
  H[:2,:2] = rotation
  H[:2, 2] = translation
  return H


def non_maximum_suppression(image):
  """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
  # Find local maximas.
  neighborhood = generate_binary_structure(2,2)
  local_max = maximum_filter(image, footprint=neighborhood)==image
  local_max[image<(image.max()*0.1)] = False

  # Erode areas to single points.
  lbs, num = label(local_max)
  centers = center_of_mass(local_max, lbs, np.arange(num)+1)
  centers = np.stack(centers).round().astype(np.int)
  ret = np.zeros_like(image, dtype=np.bool)
  ret[centers[:,0], centers[:,1]] = True

  return ret


def spread_out_corners(im, m, n, radius):
  """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
  corners = [np.empty((0,2), dtype=np.int)]
  x_bound = np.linspace(0, im.shape[1], n+1, dtype=np.int)
  y_bound = np.linspace(0, im.shape[0], m+1, dtype=np.int)
  for i in range(n):
    for j in range(m):
      # Use Harris detector on every sub image.
      sub_im = im[y_bound[j]:y_bound[j+1], x_bound[i]:x_bound[i+1]]
      sub_corners = harris_corner_detector(sub_im)
      sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis,:]
      corners.append(sub_corners)
  corners = np.vstack(corners)
  legit = ((corners[:,0]>radius) & (corners[:,0]<im.shape[1]-radius) &
           (corners[:,1]>radius) & (corners[:,1]<im.shape[0]-radius))
  ret = corners[legit,:]
  return ret


class PanoramicVideoGenerator:
  """
  Generates panorama from a set of images.
  """

  def __init__(self, data_dir, file_prefix, num_images):
    """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
    print(file_prefix)
    self.file_prefix = file_prefix
    self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
    self.files = list(filter(os.path.exists, self.files))
    self.panoramas = None
    self.homographies = None
    print('found %d images' % len(self.files))

  def align_images(self, translation_only=False):
    """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
    # Extract feature point locations and descriptors.
    # print('here')
    points_and_descriptors = []
    for file in self.files:
      # print('kaka')
      image = sol4_utils.read_image(file, 1)
      # print('bla',image)
      self.h, self.w = image.shape
      pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
      points_and_descriptors.append(find_features(pyramid))

    # Compute homographies between successive pairs of images.
    Hs = []

    for i in range(len(points_and_descriptors) - 1):
      points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
      desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

      # Find matching feature points.
      ind1, ind2 = match_features(desc1, desc2, .7)
      points1, points2 = points1[ind1, :], points2[ind2, :]

      # Compute homography using RANSAC.
      # print(points1)
      # print(points2)
      H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

      # Uncomment for debugging: display inliers and outliers among matching points.
      # In the submitted code this function should be commented out!
      # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

      Hs.append(H12)

    # Compute composite homographies from the central coordinate system.
    # print(Hs)
    accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
    self.homographies = np.stack(accumulated_homographies)
    self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
    self.homographies = self.homographies[self.frames_for_panoramas]


  def generate_panoramic_images(self, number_of_panoramas):
    """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
    assert self.homographies is not None

    # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
    self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
    for i in range(self.frames_for_panoramas.size):
      self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

    # change our reference coordinate system to the panoramas
    # all panoramas share the same coordinate system
    global_offset = np.min(self.bounding_boxes, axis=(0, 1))
    self.bounding_boxes -= global_offset

    slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
    warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
    # every slice is a different panorama, it indicates the slices of the input images from which the panorama
    # will be concatenated
    for i in range(slice_centers.size):
      slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
      # homography warps the slice center to the coordinate system of the middle image
      warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
      # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
      warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

    panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

    # boundary between input images in the panorama
    x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
    x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                  x_strip_boundary,
                                  np.ones((number_of_panoramas, 1)) * panorama_size[0]])
    x_strip_boundary = x_strip_boundary.round().astype(np.int)

    self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
    for i, frame_index in enumerate(self.frames_for_panoramas):
      # warp every input image once, and populate all panoramas
      image = sol4_utils.read_image(self.files[frame_index], 2)
      warped_image = warp_image(image, self.homographies[i])
      x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
      y_bottom = y_offset + warped_image.shape[0]

      for panorama_index in range(number_of_panoramas):
        # take strip of warped image and paste to current panorama
        boundaries = x_strip_boundary[panorama_index, i:i + 2]
        image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
        x_end = boundaries[0] + image_strip.shape[1]
        self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

    # crop out areas not recorded from enough angles
    # assert will fail if there is overlap in field of view between the left most image and the right most image
    crop_left = int(self.bounding_boxes[0][1, 0])
    crop_right = int(self.bounding_boxes[-1][0, 0])
    assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
    print(crop_left, crop_right)
    self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]


  def save_panoramas_to_video(self):
    assert self.panoramas is not None
    out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
    try:
      shutil.rmtree(out_folder)
    except:
      print('could not remove folder')
      pass
    os.makedirs(out_folder)
    # save individual panorama images to 'tmp_folder_for_panoramic_frames'
    for i, panorama in enumerate(self.panoramas):
      imsave('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
    if os.path.exists('%s.mp4' % self.file_prefix):
      os.remove('%s.mp4' % self.file_prefix)
    # write output video to current folder
    os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
              (out_folder, self.file_prefix))


  def show_panorama(self, panorama_index, figsize=(20, 20)):
    assert self.panoramas is not None
    plt.figure(figsize=figsize)
    plt.imshow(self.panoramas[panorama_index].clip(0, 1))
    plt.show()


# im1 = rgb2gray(imread('ig1.jpg'))
#
# im2 = rgb2gray(imread('ig2.jpg'))
#
# im3 = rgb2gray(imread('ig3.jpg'))
#
# im4 = rgb2gray(imread('ig4.jpg'))
#
# im5 = rgb2gray(imread('ig5.jpg'))
#
#
# pyr1,bla1 = sol4_utils.build_gaussian_pyramid(im1,3,3)
# pyr2,bla2 = sol4_utils.build_gaussian_pyramid(im2,3,3)
# pyr3,bla3 = sol4_utils.build_gaussian_pyramid(im3,3,3)
# pyr4,bla4 = sol4_utils.build_gaussian_pyramid(im4,3,3)
# pyr5,bla5 = sol4_utils.build_gaussian_pyramid(im5,3,3)
#
#
# featu1, desc1 = find_features(pyr1)
# featu2, desc2 = find_features(pyr2)
# featu3, desc3 = find_features(pyr3)
# featu4, desc4 = find_features(pyr4)
# featu5, desc5 = find_features(pyr5)
#
# match10,match20 = match_features(desc1,desc2,0.5)
# match21,match30 = match_features(desc2,desc3,0.5)
# match31,match40 = match_features(desc3,desc4,0.5)
# match41,match50 = match_features(desc4,desc5,0.5)
#
#
# homog_1_2,match_p_1_2 = ransac_homography(featu1[match10],featu2[match20],100,6)
# homog_2_3,match_p_2_3 = ransac_homography(featu2[match21],featu3[match30],100,6)
# homog_3_4,match_p_3_4 = ransac_homography(featu3[match31],featu4[match40],100,6)
# homog_4_5,match_p_4_5 = ransac_homography(featu4[match41],featu5[match50],100,6)
#
# new_homog = accumulate_homographies([homog_1_2,homog_2_3,homog_3_4,homog_4_5],2)
# new_homog2 = sol4_s2.accumulate_homographies([homog_1_2,homog_2_3,homog_3_4,homog_4_5],2)
#
# print('tomer',new_homog)
# print('shra',new_homog2)
# print(compute_bounding_box(homog_2_3,512,512))
# print(sol4_s2.compute_bounding_box(homog_2_3,512,512))

# print(sol4_s2.warp_channel(im1,homog_1_2))
# print(warp_channel(im1,homog_1_2))
