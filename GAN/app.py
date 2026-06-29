import os
import io
import base64
import numpy as np
import tensorflow as tf
import requests as http_requests
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'saved_model', 'generator.keras')
IMG_SIZE   = 256

STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY')


# ── Custom layer needed to load the Keras model ───────────────────────────────
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


# ── Load generator once at startup ───────────────────────────────────────────
print('Loading GAN generator model...')
generator = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={'InstanceNorm': InstanceNorm}
)
print('Model ready.')


# ── Image helpers ─────────────────────────────────────────────────────────────
def preprocess(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('L').resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    tensor = tf.expand_dims(arr[..., np.newaxis], axis=0)
    return tensor, img


def postprocess(tensor):
    arr = tensor[0].numpy().squeeze()
    arr = np.clip(arr * 0.5 + 0.5, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L')


def enhance_with_stability(pil_img):
    """Generate realistic face from GAN output using Stability AI v2beta structure control."""
    rgb_img = pil_img.convert('RGB').resize((1024, 1024), Image.LANCZOS)

    buf = io.BytesIO()
    rgb_img.save(buf, format='PNG')
    buf.seek(0)

    response = http_requests.post(
        "https://api.stability.ai/v2beta/stable-image/control/structure",
        headers={
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "image/*",
        },
        files={
            "image": ("image.png", buf, "image/png"),
        },
        data={
            "prompt": (
                "realistic human face profile view, photorealistic portrait, "
                "natural skin texture, clear facial features, side profile, "
                "professional portrait photography, high detail, 8k"
            ),
            "negative_prompt": (
                "blurry, cartoon, anime, distorted, low quality, x-ray, noise, ugly, deformed"
            ),
            "control_strength": "0.7",
            "output_format":    "png",
        },
        timeout=90,
    )

    if response.status_code != 200:
        raise Exception(f"Stability AI error {response.status_code}: {response.text}")

    return Image.open(io.BytesIO(response.content)).convert('RGB')


def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file received.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    allowed = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    if os.path.splitext(file.filename)[1].lower() not in allowed:
        return jsonify({'error': 'Unsupported file type.'}), 400

    try:
        image_bytes          = file.read()
        input_tensor, in_img = preprocess(image_bytes)
        output_tensor        = generator(input_tensor, training=False)
        gan_img              = postprocess(output_tensor)
        enhanced_img         = enhance_with_stability(gan_img)

        return jsonify({
            'before':   pil_to_b64(in_img),
            'after':    pil_to_b64(gan_img),
            'enhanced': pil_to_b64(enhanced_img),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
