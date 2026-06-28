import os
import io
import base64
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload

IMG_SIZE   = 256
MODEL_PATH = os.path.join('saved_model', 'generator.keras')

# ── Custom layer (required to load the Keras model) ───────────────────────────
class InstanceNorm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        c = input_shape[-1]
        self.scale  = self.add_weight(shape=(c,), initializer='ones',  trainable=True, name='scale')
        self.offset = self.add_weight(shape=(c,), initializer='zeros', trainable=True, name='offset')

    def call(self, x):
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(var + self.epsilon)
        return self.scale * x_norm + self.offset


# ── Load model once at startup ────────────────────────────────────────────────
print('Loading GAN generator...')
generator = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'InstanceNorm': InstanceNorm}
)
print('Model loaded.')


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess(image_bytes):
    """PIL image bytes → model-ready tensor."""
    img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
    arr = arr * 2.0 - 1.0                            # [-1, 1]
    return tf.expand_dims(arr[..., np.newaxis], axis=0), img


def postprocess(tensor):
    """Model output tensor → PIL image."""
    arr = tensor[0].numpy().squeeze()          # (256, 256)
    arr = np.clip(arr * 0.5 + 0.5, 0, 1)      # [-1,1] → [0,1]
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def to_base64(pil_img):
    """PIL image → base64 PNG string for embedding in HTML."""
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    allowed = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        return jsonify({'error': f'Unsupported format: {ext}'}), 400

    image_bytes = file.read()

    try:
        input_tensor, input_img = preprocess(image_bytes)
        output_tensor = generator(input_tensor, training=False)
        output_img    = postprocess(output_tensor)

        before_b64 = to_base64(input_img)
        after_b64  = to_base64(output_img)

        return jsonify({
            'before': before_b64,
            'after':  after_b64,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
