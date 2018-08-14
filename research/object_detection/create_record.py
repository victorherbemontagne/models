# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""
Example usage:
from parent folder of data_raw
  python create_record_hair.py --data_dir=data_raw --output_dir=data
"""

import hashlib
import io
import logging
import os
import random
import re

from tqdm import tqdm
import contextlib2
from lxml import etree
import numpy as np
import PIL.Image
from  PIL import ImageDraw
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('data_dir', '', 'Root directory to raw of LFW dataset.')
flags.DEFINE_string('mask_name', '', 'name of the mask folder')
flags.DEFINE_string('output_dir', '', 'Path to directory to output TFRecords.')
flags.DEFINE_string('label_map_path', 'data/label.pbtxt.txt',
                    'Path to label map proto')

flags.DEFINE_boolean('faces_only', False, 'If True, generates bounding boxes '
                     'for pet faces.  Otherwise generates bounding boxes (as '
                     'well as segmentations for full pet bodies).  Note that '
                     'in the latter case, the resulting files are much larger.')
flags.DEFINE_string('mask_type', 'png', 'How to represent instance '
                    'segmentation masks. Options are "png" or "numerical".')
flags.DEFINE_integer('num_shards', 1, 'Number of TFRecord shards')

FLAGS = flags.FLAGS


def get_class_name_from_filename(file_name):
  """Gets the class name from a file.

  Args:
    file_name: The file name to get the class name from.
               ie. "american_pit_bull_terrier_105.jpg"

  Returns:
    A string of the class name.
  """

  return ("hair")

def generate_dict_for_image(img_name,path_image):
  print(path_image)
  img = cv2.imread(os.path.join(path_image,img_name))
  h,w,d = img.shape

  template = {
    "filename":img_name,
    "size": {
      "width":w,
      "height":h,
      "depth":d
    },
    "object":[
        {
          "name":"hair",
          "mask_value":1,
          "difficult":0,
          "truncated":0,
          "pose":"Unspecified",
          "bndbox":{
            "xmin":0,
            "ymin":0,
            "xmax":0,
            "ymax":0
          }
        }
    ]
  }
  return(template)

def give_list_pictures(path):
  return(os.listdir(path))

def dict_to_tf_example(data,
                       mask_path,
                       label_map_dict,
                       image_subdirectory,
                       ignore_difficult_instances=False,
                       faces_only=False,
                       mask_type='png'):
  """Convert XML derived dict to tf.Example proto.

  Notice that this function normalizes the bounding box coordinates provided
  by the raw data.

  Args:
    data: dict holding PASCAL XML fields for a single image (obtained by
      running dataset_util.recursive_parse_xml_to_dict)
    mask_path: String path to PNG encoded mask.
    label_map_dict: A map from string label names to integers ids.
    image_subdirectory: String specifying subdirectory within the
      Pascal dataset directory holding the actual image data.
    ignore_difficult_instances: Whether to skip difficult instances in the
      dataset  (default: False).
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.

  Returns:
    example: The converted tf.Example.

  Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
  """
  img_path = os.path.join(image_subdirectory, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  image_ = np.asarray(image)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  with tf.gfile.GFile(mask_path, 'rb') as fid:
    encoded_mask_png = fid.read()
  encoded_png_io = io.BytesIO(encoded_mask_png)
  mask = PIL.Image.open(encoded_png_io)
  if mask.format != 'PNG':
    raise ValueError('Mask format not PNG')

  mask_np = np.asarray(mask)
  # nonbackground_indices_x = np.any(mask_np != 0, axis=0)
  # nonbackground_indices_y = np.any(mask_np != 0, axis=1)
  # nonzero_x_indices = np.where(nonbackground_indices_x)
  # nonzero_y_indices = np.where(nonbackground_indices_y)
  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
  truncated = []
  poses = []
  difficult_obj = []
  masks = []
  if 'object' in data:
    for obj in data['object']:
      difficult = bool(int(obj['difficult']))
      if ignore_difficult_instances and difficult:
        continue
      difficult_obj.append(int(difficult))

      if faces_only:
        xmin = float(obj['bndbox']['xmin'])
        xmax = float(obj['bndbox']['xmax'])
        ymin = float(obj['bndbox']['ymin'])
        ymax = float(obj['bndbox']['ymax'])
      else:
        mask_element_x = np.any(mask_np == obj["mask_value"], axis=0)
        mask_element_y = np.any(mask_np == obj["mask_value"], axis=1)
        mask_x_indices = np.where(mask_element_x)
        mask_y_indices = np.where(mask_element_y)
        xmin = float(np.min(mask_x_indices))
        xmax = float(np.max(mask_x_indices))
        ymin = float(np.min(mask_y_indices))
        ymax = float(np.max(mask_y_indices))
        # cv2.rectangle(image_,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(255,255,255))
        # cv2.putText(image_,str(obj["mask_value"]),(int(xmin),int(ymax)),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        # cv2.imshow("win",image_)
        # cv2.waitKey()

      xmins.append(xmin / width)
      ymins.append(ymin / height)
      xmaxs.append(xmax / width)
      ymaxs.append(ymax / height)
      class_name = obj["name"]
      classes_text.append(class_name.encode('utf8'))
      classes.append(label_map_dict[class_name])
      truncated.append(int(obj['truncated']))
      poses.append(obj['pose'].encode('utf8'))
      if not faces_only:
        mask_remapped = (mask_np == obj["mask_value"]).astype(np.uint8)
        masks.append(mask_remapped)

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
      'image/object/view': dataset_util.bytes_list_feature(poses),
  }
  if not faces_only:
    if mask_type == 'numerical':
      mask_stack = np.stack(masks).astype(np.float32)
      masks_flattened = np.reshape(mask_stack, [-1])
      feature_dict['image/object/mask'] = (
          dataset_util.float_list_feature(masks_flattened.tolist()))
    elif mask_type == 'png':
      encoded_mask_png_list = []
      for mask in masks:
        img = PIL.Image.fromarray(mask)
        output = io.BytesIO()
        img.save(output, format='PNG')
        encoded_mask_png_list.append(output.getvalue())

      feature_dict['image/object/mask'] = (
          dataset_util.bytes_list_feature(encoded_mask_png_list))

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def create_tf_record(output_filename,
                     num_shards,
                     label_map_dict,
                     image_dir,
                     mask_name,
                     examples,
                     faces_only=False,
                     mask_type='png'):
  """Creates a TFRecord file from examples.

  Args:
    output_filename: Path to where output file is saved.
    num_shards: Number of shards for output file.
    label_map_dict: The label map dictionary.
    annotations_dir: Directory where annotation files are stored.
    image_dir: Directory where image files are stored.
    examples: Examples to parse and save to tf record.
    faces_only: If True, generates bounding boxes for pet faces.  Otherwise
      generates bounding boxes (as well as segmentations for full pet bodies).
    mask_type: 'numerical' or 'png'. 'png' is recommended because it leads to
      smaller file sizes.
  """
  with contextlib2.ExitStack() as tf_record_close_stack:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
        tf_record_close_stack, output_filename, num_shards)
    for idx, example in tqdm(enumerate(examples)):
      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(examples))
      data = generate_dict_for_image(example,image_dir)
      mask_path = "data_raw/{}/{}ppm.png".format(mask_name,example[:-3])
      # xml_path = os.path.join(annotations_dir, 'xmls', example + '.xml')
      # mask_path = os.path.join(annotations_dir, 'trimaps', example + '.png')

      # if not os.path.exists(xml_path):
      #   logging.warning('Could not find %s, ignoring example.', xml_path)
      #   continue
      # with tf.gfile.GFile(xml_path, 'r') as fid:
      #   xml_str = fid.read()
      # xml = etree.fromstring(xml_str)
      # data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

      try:
        tf_example = dict_to_tf_example(
            data,
            mask_path,
            label_map_dict,
            image_dir,
            faces_only=faces_only,
            mask_type=mask_type)
        if tf_example:
          shard_idx = idx % num_shards
          output_tfrecords[shard_idx].write(tf_example.SerializeToString())
      except ValueError as e:
        # raise
        logging.warning('Invalid example: %s, ignoring.', example) # come from example where their is no hair


# TODO(derekjchow): Add test for pet/PASCAL main files.
def main(_):
  data_dir = FLAGS.data_dir
  mask_name = FLAGS.mask_name
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  logging.info('Reading from LFW dataset.')
  image_dir = os.path.join(data_dir, 'images')
  # annotations_dir = os.path.join(data_dir, 'annotations') ADDED
  # examples_path = os.path.join(annotations_dir, 'trainval.txt') ADDED
  mask_dir = os.path.join(data_dir, mask_name)
  examples_list = [img_path[:-8]+".jpg" for img_path in os.listdir(mask_dir)]

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
  random.seed(42)
  random.shuffle(examples_list)
  num_examples = len(examples_list)
  num_train = int(0.7 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  if not FLAGS.faces_only:
    train_output_path = os.path.join(FLAGS.output_dir,
                                     'pictures_with_masks_train.record')
    val_output_path = os.path.join(FLAGS.output_dir,
                                   'pictures_with_masks_val.record')
  create_tf_record(
      train_output_path,
      FLAGS.num_shards,
      label_map_dict,
      image_dir,
      mask_name,
      train_examples, #liste avec les noms des images choisies pour le train
      faces_only=FLAGS.faces_only,
      mask_type=FLAGS.mask_type)
  create_tf_record(
      val_output_path,
      FLAGS.num_shards,
      label_map_dict,
      image_dir, #path to images
      mask_name,
      val_examples,#liste avec les noms des images choisies pour la validation
      faces_only=FLAGS.faces_only,
      mask_type=FLAGS.mask_type)


if __name__ == '__main__':
  tf.app.run()
