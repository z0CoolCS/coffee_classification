from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

#model = tf.keras.models.load_model('saved_model_v2/my_model')
with tf.device('/cpu:0'):
	model = tf.keras.models.load_model('saved_model_v2/my_model') #tf.keras.models.load_model('my_model.h5')


def predict(inp):
	logits = model.predict(np.expand_dims(inp, 0))
	#class_pred = tf.math.argmax(tf.nn.softmax(logits), 1).numpy()[0]
	probs = tf.nn.softmax(logits)
	return probs.numpy()[0]

def build_image(bytes_inp):
	file_bytes = np.fromstring(bytes_inp, np.uint8)
	img = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED).astype(np.float32)
	#img = cv2.imdecode(np.frombuffer(bytes_inp, np.uint8), -1).astype(np.float32)
	dim = (256, 256)
	resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
	return resized

@app.route("/", methods = ["GET", "POST"])
def index():
	if request.method == "POST":
		file = request.files.get("file")
		
		if file is None or file.filename == "":
			return jsonify({ "error" : "no file"})

		try:
			file = file.read()
			print(type(file))
			img = build_image(file)
			#print()
			probs = list(predict(img))
			print('VALUE', probs)
			print('CLASS', type(probs))

			return jsonify({ "ok" : str(probs)})
		except Exception as e:
			return jsonify({ "error" : e})

	return jsonify({ "ok" : "ok"})


if __name__ == "__main__":
	app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
