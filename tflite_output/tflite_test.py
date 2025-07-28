import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="tflite_output/tavns_model_20250728_152530.tflite")
interpreter.allocate_tensors()

for d in interpreter.get_tensor_details():
    print(d['name'], d['dtype'], d['shape'])

print("\nOps in model:")
for op in interpreter._get_ops_details():
    print(op['op_name'])