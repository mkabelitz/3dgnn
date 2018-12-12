import glob

import warnings
import struct

import numpy as np
import PIL.Image
import cv2
import json
import array
#import Imath
#import OpenEXR
from scipy.ndimage import imread

from numpy import linalg

# import logging
import sys
import os
import yaml

# logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
#                     level=logging.DEBUG,
#                     stream=sys.stdout)

"""
----- Own utils
"""


def get_random_image_triplet(path, areas, names):
    area_index = np.random.randint(6)
    name = np.random.choice(names[area_index])
    rgb = cv2.cvtColor(cv2.imread(path + areas[area_index] + '/data/preprocessed/rgb/' + name + 'rgb.png'),
                       cv2.COLOR_BGR2RGB)
    depth = cv2.cvtColor(cv2.imread(path + areas[area_index] + '/data/preprocessed/depth/' + name + 'depth.png'),
                         cv2.COLOR_BGR2GRAY)
    label = cv2.cvtColor(cv2.imread(path + areas[area_index] + '/data/preprocessed/label/' + name + 'label.png'),
                         cv2.COLOR_BGR2GRAY)
    return rgb, depth, label


def get_image_batch(batch_size, path, areas, names, preallocated_x=None, preallocated_y=None):
    # this doesn't seem to make any difference :)
    rgbd = preallocated_x if preallocated_x is not None else np.zeros((batch_size, 360, 360, 4))
    label = preallocated_y if preallocated_y is not None else np.zeros((batch_size, 360, 360))

    for i in range(batch_size):
        rgb, d, l = get_random_image_triplet(path, areas, names)
        rgbd[i, :, :, :3] = rgb
        rgbd[i, :, :, 3] = d
        label[i] = l.reshape((-1, 1, 360, 360))

    return rgbd, label.astype(np.int64)


def load_names(path):
    offset = len(path)
    return [(file.rsplit('_', 1)[0] + '_')[offset:] for file in glob.glob(path + '*')]


def load_names_by_substring(path, substring):
    offset = len(path)
    return [(file.rsplit('_', 1)[0] + '_')[offset:] for file in glob.glob(path + '*' + substring + '*')]


def load_png_to_numpy(path):
    return np.array((PIL.Image.open(path)).getdata())


def pointcloud_from_names(names, path, subsample_step=1):
    '''
    Takes a list of filenames from the Stanford 2D-3D dataset and computes a pointcloud from the corresponding
    RGB and depth files.

    Parameters
    ----------
    names: list of strings
        Includes all the filenames to be merged into pointcloud, each name up to 'blablabla_domain_'.

    subsample_step: int
        Number of pixels skipped in x and y direction, so we end up with 1 out of subsample_step**2 pixels.

    Returns
    -------
    point_cloud_xyz : ndarray
        Array of global xyz coordinates.

    point_cloud_rgb : ndarray
        Array of RGB values corresponding to the points in point_cloud_xyz in same order.
    '''

    assert 1080 % subsample_step == 0
    assert subsample_step > 0

    # invert via intrinsic camera paramters
    def apply_k(x, y, z, k):
        cc_x = k[0][2]
        cc_y = k[1][2]
        fc_x = k[0][0]
        fc_y = k[1][1]
        new_x = ((x - (cc_x / subsample_step)) * z) / (fc_x / subsample_step)
        new_y = ((y - (cc_y / subsample_step)) * z) / (fc_y / subsample_step)
        new_z = z
        return new_x, new_y, new_z

    # for subsampling 1080x1080 images at resolution res x res
    res = int(1080 / subsample_step)

    # collect xyz and RGB data here
    point_cloud_xyz = np.empty((1, 4), dtype=np.float32)
    point_cloud_rgb = np.empty((1, 3), dtype=np.uint8)

    for count, name in enumerate(names):

        # image specific camera settings
        camera_json = load_labels(path + 'data/pose/' + name + 'pose.json')
        camera_k_matrix = camera_json['camera_k_matrix']
        camera_rt_matrix = np.array(camera_json['camera_rt_matrix'])

        # load and subsample RGB and depth image
        rgb_np_frame = load_png_to_numpy(path + 'data/rgb/' + name + 'rgb.png')
        rgb_sub = (rgb_np_frame.reshape((1080, 1080, 3))[::subsample_step, ::subsample_step]).reshape(-1, 3)
        depth_np_frame = load_png_to_numpy(path + 'data/depth/' + name + 'depth.png')
        depth_sub = (depth_np_frame.reshape((1080, 1080))[::subsample_step, ::subsample_step]).reshape(-1)

        # preallocate point cloud for current image
        cur_point_cloud_xyz = np.ones((res ** 2, 4), dtype=np.float32)

        for x in range(res ** 2):
            depth = depth_sub[x]
            # project all pixels with missing depth value to global [0, 0, 0]
            if depth == 2 ** 16 - 1:
                cur_point_cloud_xyz[x] = [0, 0, 0, 1]
                continue
            # assign image coordinates and depth
            new_x, new_y, new_z = x % res, int(x / res), (depth / 512.)
            # invert via K matrix: image coordinates -> camera coordinates
            new_x, new_y, new_z = apply_k(new_x, new_y, new_z, camera_k_matrix)
            cur_point_cloud_xyz[x] = [new_x, new_y, new_z, 1]

        # invert via RT matrix: camera coordinates -> global coordinates
        rt_inv = linalg.inv(np.concatenate((camera_rt_matrix, [[0, 0, 0, 1]])))
        cur_point_cloud_xyz = np.matmul(cur_point_cloud_xyz, np.transpose(rt_inv))

        # add new points with xyz and RGB values
        point_cloud_xyz = np.concatenate((point_cloud_xyz, cur_point_cloud_xyz), axis=0)
        point_cloud_rgb = np.concatenate((point_cloud_rgb, rgb_sub), axis=0)

    return point_cloud_xyz[:, :-1], point_cloud_rgb


"""
----- Utils from Alex for reading and writing binary pcd files
"""


# !/usr/bin/env python3

def write_binary_pcd(filename, xyz, rgb=None):
    '''
    Save the provided data as a 3D PCL pointcloud.

    Parameters
    ----------
    filename: string
        filename to be saved to.

    xyz: Nx3 ndarray
        xyz data to be used, should be np.float32.

    rgb: None or Nx3/Nx4 ndarray
        If provided this is stored as rgb/rgba data, should be np.uint8.
        Likely the alpha channel has no semantic value.

    '''

    # Determine how many color channels we will be saving
    color_channels = rgb is not None

    # Setup the header
    header = '# .PCD v0.7 - Point Cloud Data file format\n'
    header += 'VERSION 0.7\n'
    header += 'FIELDS x y z' + ' rgba' * (color_channels) + '\n'
    header += 'SIZE 4 4 4' + ' 4' * (color_channels) + '\n'
    header += 'TYPE F F F' + ' U' * (color_channels) + '\n'
    header += 'COUNT 1 1 1' + ' 1' * (color_channels) + '\n'
    header += 'WIDTH {0}\n'
    header += 'HEIGHT 1\n'
    header += 'VIEWPOINT 0 0 0 1 0 0 0\n'
    header += 'POINTS {0}\n'
    header += 'DATA binary\n'
    header = header.format(xyz.shape[0])

    # Copy together all the data after checking that xyz is in the correct format.
    if xyz.dtype != np.float32:
        warnings.warn('xyz was supplied as {}, casting to np.float32'.format(xyz.dtype))
        xyz = xyz.astype(np.float32)
    data = [xyz]

    # where possible fix the color values add them in the needed format
    if rgb is not None:
        if rgb.dtype != np.uint8:
            warnings.warn('rgb was supplied as {}, casting to np.uint8'.format(rgb.dtype))
            rgb = rgb.astype(np.uint8)
        if xyz.shape[0] != rgb.shape[0]:
            raise Exception('xyz ({}) and rgb ({}) point counts don\'t match.'.format(xyz.shape[0], rgb.shape[0]))
        if rgb.shape[1] == 3:
            rgb = np.concatenate([rgb, np.zeros((xyz.shape[0], 1), dtype=np.float32)], axis=1)
        elif rgb.shape[1] != 4:
            raise Exception('Only 3 or 4 dimensional rgb inputs are allowed. Got:{}'.format(rgb.shape[1]))

        data += [np.asarray([struct.unpack('f', d) for d in rgb.astype(np.byte)], dtype=np.float32)]

    # Make a big block out of them
    data = np.concatenate(data, axis=1).flatten()

    # Write both the header and then the binary data.
    with open(filename, 'wb') as f:
        for x in header:
            f.write(x.encode('utf-8'))
        data.tofile(f)


def read_pcd(filename):
    '''
    Read a pcd file from the provided filename, ascii or binary.
    It pretty much only support xyz point clouds, possibly with rgb(a) colors.
    Other fields are untested and might cause problems thus they will not work.
    Will be extended as needed.

    Parameters
    ----------
    filename: string
        filename to be opened.

    Returns
    -------
    xyz : ndarray
        Array of xyz coordinates.

    rgb : ndarray, optional
        rgb(a) colors for the points if found in the point cloud.
    '''

    # Parse the initial header
    header = ''
    with open(filename, 'rb') as input:
        # Get byte by byte
        aByte = input.read(1)
        while aByte and ord(aByte) != 0:
            aByte = input.read(1)
            header += aByte.decode("utf-8")

            # Check if the header is done and what type the data is.
            # Load accordingly
            if header.endswith('binary\n'):
                data = np.fromfile(input, dtype=np.float32)
                break
            elif header.endswith('ascii\n'):
                data = np.loadtxt(input, dtype=np.float32)
                break

    # Parse further information from the header. We need the fields and hte point counts
    hlines = header.split('\n')
    point_count = -1
    fields = []

    for hl in hlines:
        if hl.lower().startswith('fields'):
            fields = hl.split(' ')[1:]
        if hl.lower().startswith('points'):
            point_count = int(hl.split(' ')[1])

    # Check if we got what we need.
    if point_count == -1:
        raise Exception('Could not identify the point cloud. Maybe a broken header?')

    if len(fields) == 0:
        raise Exception('Could not identify the fields to be loaded. Maybe a broken header?')

    # Get the indices for the x,y,z and possibly rgb(a) fields
    fields = {f: i for i, f in enumerate(fields)}
    xyz_idx = [fields['x'], fields['y'], fields['z']]
    del fields['x']
    del fields['y']
    del fields['z']

    # Check for rgb and possibly rgba, only use one.
    rgb_idx = -1
    rgb_found = 0
    if 'rgb' in fields:
        rgb_idx = fields['rgb']
        rgb_found += 1
        del fields['rgb']
    if 'rgba' in fields:
        if rgb_idx != -1:
            warnings.warn('Found both an rgb and rgba field, using the latter.')
        rgb_idx = fields['rgba']
        rgb_found += 1
        del fields['rgba']

    # Anything else is not supported right now and would mess up the results, we cannot trust this.
    if len(fields) != 0:
        raise Exception('Additional fields found which are not supported: {}'.format(', '.join(fields.keys())))

    # Extract the relevant data
    channels = 3 + rgb_found

    # Chop off the random zeros at the end :-/ Wth PCL man!?
    data = data[:point_count * channels].reshape((point_count, channels)).astype(np.float32)

    xyz = data[:, xyz_idx]

    if rgb_found:
        rgba = np.asarray([bytearray(struct.pack("f", d)) for d in data[:, rgb_idx]])
        return xyz, rgba
    else:
        return xyz


"""
----- Utils from original S2D3D repo (https://github.com/alexsax/2D-3D-Semantics)
"""

""" Semantics """


def get_index(color):
    ''' Parse a color as a base-256 number and returns the index
    Args:
        color: A 3-tuple in RGB-order where each element \in [0, 255]
    Returns:
        index: an int containing the indec specified in 'color'
    '''
    return color[0] * 256 * 256 + color[1] * 256 + color[2]


def get_color(i):
    ''' Parse a 24-bit integer as a RGB color. I.e. Convert to base 256
    Args:
        index: An int. The first 24 bits will be interpreted as a color.
            Negative values will not work properly.
    Returns:
        color: A color s.t. get_index( get_color( i ) ) = i
    '''
    b = (i) % 256  # least significant byte
    g = (i >> 8) % 256
    r = (i >> 16) % 256  # most significant byte
    return r, g, b


""" Label functions """


def load_labels(label_file):
    """ Convenience function for loading JSON labels """
    with open(label_file) as f:
        return json.load(f)


def parse_label(label):
    """ Parses a label into a dict """
    res = {}
    clazz, instance_num, room_type, room_num, area_num = label.split("_")
    res['instance_class'] = clazz
    res['instance_num'] = int(instance_num)
    res['room_type'] = room_type
    res['room_num'] = int(room_num)
    res['area_num'] = int(area_num)
    return res


"""
----- Utils for reading exr fils (TODO: I lost the link to source...)
"""

""" EXR Functions """


def normalize_array_for_matplotlib(arr_to_rescale):
    ''' Rescales an array to be between [0, 1]
    Args:
        arr_to_rescale:
    Returns:
        An array in [0,1] with f(0) = 0.5
    '''
    return (arr_to_rescale / np.abs(arr_to_rescale).max()) / 2 + 0.5


# def read_exr(image_fpath):
#     """ Reads an openEXR file into an RGB matrix with floats """
#     f = OpenEXR.InputFile(image_fpath)
#     dw = f.header()['dataWindow']
#     w, h = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
#     im = np.empty((h, w, 3))
#
#     # Read in the EXR
#     FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
#     channels = f.channels(["R", "G", "B"], FLOAT)
#     for i, channel in enumerate(channels):
#         im[:, :, i] = np.reshape(array.array('f', channel), (h, w))
#     return im























def load_cfg(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cfg : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cfg = yaml.load(stream)
    cfg = make_paths_absolute(os.path.dirname(yaml_filepath), cfg)
    return cfg


def make_paths_absolute(dir_, cfg):
    """
    Make all values for keys ending with `_path` absolute to dir_.

    Parameters
    ----------
    dir_ : str
    cfg : dict

    Returns
    -------
    cfg : dict
    """
    for key in cfg.keys():
        if key.endswith("_path"):
            cfg[key] = os.path.join(dir_, cfg[key])
            cfg[key] = os.path.abspath(cfg[key])
            # if not os.path.isfile(cfg[key]):
            #     logging.error("%s does not exist.", cfg[key])
        if type(cfg[key]) is dict:
            cfg[key] = make_paths_absolute(dir_, cfg[key])
    return cfg


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--file",
                        dest="filename",
                        help="experiment definition file",
                        metavar="FILE",
                        required=True)
    return parser