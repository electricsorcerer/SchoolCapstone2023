import tensorflow as tf

# Path to the input binary .pb file
input_pb_file ='D:\Coding\School\img_rec_proj\TheModel\\saved_model.pb'

# Path to the output text .pbtxt file
output_pbtxt_file = 'D:\Coding\School\img_rec_proj\TheModel\\saved_model_text.pbtxt'

# Load the binary .pb file
with tf.io.gfile.GFile(input_pb_file, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

# Convert to text format and save
tf.io.write_graph(graph_def, '.', output_pbtxt_file, as_text=True)

print(f"Conversion complete. Saved as {output_pbtxt_file}")