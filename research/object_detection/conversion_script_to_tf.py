import tensorflow as tf
import os
from object_detection.utils import dataset_util
from tqdm import tqdm
import cv2



flags = tf.app.flags
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('picture_paths', '', 'Path to the input data to serialize ')
FLAGS = flags.FLAGS


def create_tf_example(path_image,mask_face,mask_hair):

  with open(path_image, "rb") as file:
    image_encoded = file.read()

  # TODO(user): Populate the following variables from your example.
  height = 250 # Image height
  width = 250 # Image width
  filename = b"" # Filename of the image. Empty if image is not from file
  encoded_image_data = image_encoded # Encoded image bytes
  image_format = b".jpg" # b'jpeg' or b'png'

  xmins = [1,1] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [1,1] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [1,1] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [1,1] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = [b"face"] # List of string class name of bounding box (1 per box)
  classes = [1,2] # List of integer class id of bounding box (1 per box)
  mask = [mask_face,mask_hair]

  tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      "image/object/mask": dataset_util.bytes_list_feature(mask)
  }))
  return tf_example


def serializes_mask_for_link(name_mask):
  path_mask_face = os.path.join("mask_face",name_mask)
  path_mask_hair = os.path.join("mask_hair",name_mask)

  with open(path_mask_face, "rb") as file:
    content_string_face = file.read()
  
  with open(path_mask_hair, "rb") as file:
    content_string_hair = file.read()

  return(content_string_face,content_string_hair)

def choose_img(picture_path,name_mask):
  picture_file_name = name_mask[:-8]+".jpg"
  full_path = os.path.join(picture_path,picture_file_name)
  return(full_path)


def main(_):
  print("output path",FLAGS.output_path)
  

  picture_paths = FLAGS.picture_paths

  # TODO(user): Write code to read in your dataset to examples variable
  list_picture = os.listdir(picture_paths)
  list_mask = os.listdir("mask_face")

  list_mask_train = list_mask[:int(len(list_mask)*0.7)]
  
  print("Processing the train set of size ->{}".format(int(len(list_mask)*0.7)))
  writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,"train.record"))
  for name_mask in tqdm(list_mask_train):

    mask_face,mask_hair = serializes_mask_for_link(name_mask)
    path_image = choose_img(picture_paths,name_mask) #name mask il va ressembler a un truc Prenom_Nom_4chiffres.ppm.png


    tf_example = create_tf_example(path_image,mask_face,mask_hair)
    writer.write(tf_example.SerializeToString())

  writer.close()

  list_mask_eval = list_mask[int(len(list_mask)*0.7)+1:]
  writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_path,"eval.record"))
  
  print("Processing the eval set of size ->{}".format(int(len(list_mask)*0.3)))
  for name_mask in tqdm(list_mask_eval):

    mask_face,mask_hair = serializes_mask_for_link(name_mask)
    path_image = choose_img(picture_paths,name_mask) #name mask il va ressembler a un truc Prenom_Nom_4chiffres.ppm.png


    tf_example = create_tf_example(path_image,mask_face,mask_hair)
    writer.write(tf_example.SerializeToString())

  writer.close()
  

if __name__ == '__main__':
  tf.app.run()