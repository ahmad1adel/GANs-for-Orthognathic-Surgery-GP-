import os
import io
import base64
import numpy as np
import tensorflow as tf
import requests as http_requests
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'saved_model', 'generator.keras')
IMG_SIZE   = 256

load_dotenv(os.path.join(BASE_DIR, '.env'))
STABILITY_API_KEY = os.environ.get('STABILITY_API_KEY')
print(f'Stability API key loaded: {"YES" if STABILITY_API_KEY else "MISSING"}')


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


def enhance_from_original(original_pil):
    """
    Realistic 'after surgery' result via Stability img2img.

    Runs on the ORIGINAL clean photo (not the blurry GAN output) at low strength,
    so the real person, hair and grayscale look are preserved and only the jaw /
    chin profile is refined.
    """
    # Square, decent resolution; keep original detail
    rgb_img = original_pil.convert('RGB').resize((1024, 1024), Image.LANCZOS)

    buf = io.BytesIO()
    rgb_img.save(buf, format='PNG')
    buf.seek(0)

    response = http_requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "Authorization": f"Bearer {STABILITY_API_KEY}",
            "Accept": "image/*",
        },
        files={
            "image": ("image.png", buf, "image/png"),
        },
        data={
            "prompt": (
                "same young man, side profile portrait, black and white photograph, "
                "stronger and more forward lower jaw, better chin projection, "
                "smooth defined jawline, balanced facial profile, "
                "realistic skin, sharp focus, studio lighting, "
                "keep the same hairstyle, nose, forehead, ear and eye"
            ),
            "negative_prompt": (
                "woman, female, different person, cartoon, anime, painting, sketch, "
                "skull, x-ray, deformed, distorted, blurry, extra face, color, colorized"
            ),
            "mode":          "image-to-image",
            "strength":      "0.35",
            "model":         "sd3.5-large-turbo",
            "output_format": "png",
        },
        timeout=90,
    )

    if response.status_code != 200:
        raise Exception(f"Stability AI error {response.status_code}: {response.text[:300]}")

    # Grayscale to match the medical black & white style
    return Image.open(io.BytesIO(response.content)).convert('L').convert('RGB')


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

        # Keep a clean, full-detail copy of the original for the realistic edit
        original_img         = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        input_tensor, in_img = preprocess(image_bytes)
        output_tensor        = generator(input_tensor, training=False)
        gan_img              = postprocess(output_tensor)

        # Realistic result is edited from the ORIGINAL photo, not the blurry GAN output
        enhanced_img         = enhance_from_original(original_img)

        return jsonify({
            'before':   pil_to_b64(in_img),
            'after':    pil_to_b64(gan_img),
            'enhanced': pil_to_b64(enhanced_img),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
