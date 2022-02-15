import imp
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import flask 
import io

app = flask.Flask(__name__)
model = None

def load_model():
    global model 
    model = tf.keras.models.load_model("prod-cats-dogs")

class DefaultConfig:
    img_size: int = 224
    labels: list = ['cat', 'dog']

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read the image in PIL format
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            # preprocess the image and prepare it for classification
            image = prepare_image(image, 
            target=(DefaultConfig.img_size,DefaultConfig.img_size)
            )
            # do prediction on the image
            pred = model.predict(image)
            data["prediction"] = DefaultConfig.labels[pred]
            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()