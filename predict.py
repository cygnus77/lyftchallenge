import sys, skvideo.io, json, base64
import numpy as np
from PIL import Image
from io import BytesIO, StringIO
import tensorflow as tf
import time

if len(sys.argv) != 3:
    raise Exception("required model and image file path")
file = sys.argv[-1]
modelfile = sys.argv[-2]

# Define encoder function
def encode(array):
  pil_img = Image.fromarray(array)
  buff = BytesIO()
  pil_img.save(buff, format="PNG")
  return base64.b64encode(buff.getvalue()).decode("utf-8")

def preprocess_input(x):
    return (x / 127.5) - 1.

video = skvideo.io.vread(file)

answer_key = {}

with tf.gfile.GFile(modelfile, "rb") as f:
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
  ph_image = tf.placeholder(tf.float32, [None, 600, 800, 3])
  img = tf.slice(ph_image, [0,0,0,0], [-1, 576, 800, 3] )
  img = tf.div(img, 127.5)
  img = tf.subtract(img, 1.)

  ph_processed_image = tf.placeholder(tf.float32, [None, 576, 800, 3])

  output_tensor = tf.import_graph_def(
    graph_def,
    # usually, during training you use queues, but at inference time use placeholders
    # this turns into "input
    input_map={"input_1": ph_processed_image},
    return_elements=["conv2d_transpose_4/Sigmoid:0"]
  )
    
  output_tensor = tf.rint(output_tensor)
  output_tensor = tf.cast(output_tensor, tf.uint8)
  output_tensor = tf.squeeze(output_tensor, axis=[0])
  pad = tf.constant(0, dtype=tf.uint8, shape=[600-576, 800, 2])
  output_tensor = tf.map_fn(lambda x: tf.concat([x, pad], axis=0), output_tensor)
  output_tensors = tf.unstack(output_tensor, axis=3)

  with tf.Session(graph=graph) as sess:

    # Frame numbering starts at 1
    frame = 1
    group_by = 30
    for startidx in range(0, video.shape[0], group_by):
      batch_size = min(video.shape[0], startidx + group_by) - startidx
      batch = video[startidx:startidx + batch_size]
      image = sess.run(img, feed_dict={ph_image: batch})
      result = sess.run(output_tensors, feed_dict={ph_processed_image: image})
      for i in range(batch_size):
        answer_key[frame] = [encode(result[1][i]), encode(result[0][i])]
        frame+=1
# Print output in proper json format
print (json.dumps(answer_key))

