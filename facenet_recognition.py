"""
FaceNet Face Recognition System
A complete implementation of FaceNet for face verification and recognition using triplet loss
"""

import os
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from numpy import genfromtxt
import pandas as pd
import tensorflow as tf
import keras
from keras import backend as K
import PIL

K.set_image_data_format('channels_last')

keras.config.enable_unsafe_deserialization()


@keras.saving.register_keras_serializable()
class ScaleSum(keras.layers.Layer):
    """Replacement for the serialized Lambda layers in the Keras 2.x model.
    Computes: inputs[0] + inputs[1] * scale
    """
    def __init__(self, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs[0] + inputs[1] * self.scale

    def get_config(self):
        config = super().get_config()
        config['scale'] = self.scale
        return config

# For Jupyter notebooks (comment out if running as script)
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2


# ============================================
# TRIPLET LOSS FUNCTION
# ============================================

def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss function for FaceNet.
    
    Triplet loss ensures that:
    - Distance(Anchor, Positive) < Distance(Anchor, Negative)
    - With a margin alpha
    
    Arguments:
    y_true -- true labels (not used in this implementation)
    y_pred -- python list containing three objects:
            anchor -- encodings for anchor images, shape (None, 128)
            positive -- encodings for positive images, shape (None, 128)
            negative -- encodings for negative images, shape (None, 128)
    alpha -- margin parameter (default 0.2)
    
    Returns:
    loss -- triplet loss value
    """
    
    # Extract anchor, positive, and negative encodings
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Convert to tensors
    anchor = tf.convert_to_tensor(anchor, dtype=tf.float32)
    positive = tf.convert_to_tensor(positive, dtype=tf.float32)
    negative = tf.convert_to_tensor(negative, dtype=tf.float32)
    
    # Step 1: Distance between anchor and positive
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    
    # Step 2: Distance between anchor and negative
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    # Step 3: Triplet loss formula: max(||A-P||^2 - ||A-N||^2 + alpha, 0)
    basic_loss = pos_dist - neg_dist + alpha
    
    # Step 4: Sum over all examples
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss


# ============================================
# LOAD FACENET MODEL
# ============================================

def load_facenet_model(model_json_path='keras-facenet-h5/model.json',
                       model_weights_path='keras-facenet-h5/model.h5'):
    """
    Load FaceNet model from JSON and weights files.

    Handles Keras 2.x -> Keras 3.x migration by patching the saved JSON:
      - Registers the old 'Functional' top-level class.
      - Replaces serialised Lambda layers (whose bytecode is not portable)
        with the equivalent ScaleSum custom layer defined above.

    Arguments:
        model_json_path -- path to model architecture JSON file
        model_weights_path -- path to model weights H5 file

    Returns:
        model -- loaded Keras model
    """
    from keras.src.models.functional import Functional

    with open(model_json_path, 'r') as f:
        config = json.load(f)

    for layer in config['config']['layers']:
        if layer['class_name'] == 'Lambda':
            scale = layer['config']['arguments']['scale']
            layer['class_name'] = 'ScaleSum'
            layer['config'] = {
                'name': layer['name'],
                'trainable': True,
                'dtype': 'float32',
                'scale': scale,
            }

    patched_json = json.dumps(config)

    custom_objects = {'Functional': Functional, 'ScaleSum': ScaleSum}
    with keras.saving.custom_object_scope(custom_objects):
        model = keras.models.model_from_json(patched_json)

    model.load_weights(model_weights_path)
    return model


# ============================================
# IMAGE ENCODING FUNCTION
# ============================================

def img_to_encoding(image_path, model):
    """
    Convert image to 128-dimensional face encoding.
    Raises ValueError if the image doesn't appear to contain a face.
    
    Arguments:
        image_path -- path to input image
        model -- FaceNet model
    
    Returns:
        embedding -- normalized 128-dimensional face encoding
    """
    img = keras.utils.load_img(image_path, target_size=(160, 160))
    img_array = np.around(np.array(img) / 255.0, decimals=12)

    if np.std(img_array) < 0.15:
        raise ValueError(f"Image '{image_path}' does not appear to contain a face.")

    x_train = np.expand_dims(img_array, axis=0)
    embedding = model.predict_on_batch(x_train)
    embedding = embedding / np.linalg.norm(embedding, ord=2)
    return embedding


# ============================================
# DATABASE CREATION
# ============================================

def create_database(FRmodel, image_paths=None):
    """
    Create database of face encodings for known people
    
    Arguments:
        FRmodel -- FaceNet model
        image_paths -- dictionary mapping names to image paths
    
    Returns:
        database -- dictionary mapping names to face encodings
    """
    if image_paths is None:
        # Default database
        image_paths = {
            "danielle": "images/danielle.png",
            "younes": "images/younes.jpg",
            "tian": "images/tian.jpg",
            "andrew": "images/andrew.jpg",
            "kian": "images/kian.jpg",
            "dan": "images/dan.jpg",
            "sebastiano": "images/sebastiano.jpg",
            "bertrand": "images/bertrand.jpg",
            "kevin": "images/kevin.jpg",
            "felix": "images/felix.jpg",
            "benoit": "images/benoit.jpg",
            "arnaud": "images/arnaud.jpg"
        }
    
    database = {}
    for name, path in image_paths.items():
        database[name] = img_to_encoding(path, FRmodel)
        print(f"✅ Added {name} to database")
    
    return database


# ============================================
# FACE VERIFICATION FUNCTION
# ============================================

def verify(image_path, identity, database, model, threshold=0.5):
    """
    Verify if the person in the image matches the claimed identity
    
    Arguments:
        image_path -- path to image to verify
        identity -- string, name of the person to verify against
        database -- dictionary of known face encodings
        model -- FaceNet model
        threshold -- distance threshold for verification (default 0.5)
    
    Returns:
        dist -- distance between image and database encoding
        door_open -- True if identity verified, False otherwise
    """
    
    try:
        encoding = img_to_encoding(image_path, model)
    except ValueError as e:
        print(f"⚠️ {e}")
        return None, False
    
    dist = np.linalg.norm(encoding - database[identity])
    
    # Step 3: Verify based on threshold
    if dist < threshold:
        print(f"It's {identity}, welcome in!")
        door_open = True
    else:
        print(f"It's not {identity}, please go away")
        door_open = False
    
    return dist, door_open


# ============================================
# FACE RECOGNITION FUNCTION
# ============================================

def who_is_it(image_path, database, model, threshold=0.5):
    """
    Identify the person in the image by comparing with database
    
    Arguments:
        image_path -- path to image to identify
        database -- dictionary of known face encodings
        model -- FaceNet model
        threshold -- distance threshold for recognition
    
    Returns:
        min_dist -- minimum distance found
        identity -- name of identified person
    """
    
    try:
        encoding = img_to_encoding(image_path, model)
    except ValueError as e:
        print(f"⚠️ {e}")
        return None, None
    
    min_dist = 100  # Initialize to large value
    identity = None
    
    for name, db_enc in database.items():
        # Compute L2 distance
        dist = np.linalg.norm(encoding - db_enc)
        
        # Update if this is the closest match
        if dist < min_dist:
            min_dist = dist
            identity = name
    
    # Step 3: Check if match is within threshold
    if min_dist > threshold:
        print("Not in the database.")
        identity = None
    else:
        print(f"It's {identity}, the distance is {min_dist:.4f}")
    
    return min_dist, identity


# ============================================
# VISUALIZATION FUNCTIONS
# ============================================

def display_image(image_path, title=None):
    """
    Display an image using matplotlib
    
    Arguments:
        image_path -- path to image
        title -- optional title for the image
    """
    import matplotlib.pyplot as plt
    
    img = keras.utils.load_img(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def display_comparison(image1_path, image2_path, label1=None, label2=None):
    """
    Display two images side by side for comparison
    
    Arguments:
        image1_path -- path to first image
        image2_path -- path to second image
        label1 -- label for first image
        label2 -- label for second image
    """
    import matplotlib.pyplot as plt
    
    img1 = keras.utils.load_img(image1_path)
    img2 = keras.utils.load_img(image2_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(img1)
    axes[0].set_title(label1 if label1 else "Image 1")
    axes[0].axis('off')
    
    axes[1].imshow(img2)
    axes[1].set_title(label2 if label2 else "Image 2")
    axes[1].axis('off')
    
    plt.show()


def display_verification_result(image_path, identity, dist, door_open):
    """
    Display verification result with visual feedback
    
    Arguments:
        image_path -- path to verified image
        identity -- identity that was verified
        dist -- distance between encodings
        door_open -- whether verification succeeded
    """
    import matplotlib.pyplot as plt
    
    img = keras.utils.load_img(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    
    if door_open:
        color = 'green'
        status = "✅ ACCESS GRANTED"
    else:
        color = 'red'
        status = "❌ ACCESS DENIED"
    
    plt.title(f"Verification: {identity}\n{status}\nDistance: {dist:.4f}", 
              color=color, fontsize=14)
    plt.axis('off')
    plt.show()


# ============================================
# TEST FUNCTIONS
# ============================================

def test_triplet_loss():
    """Test the triplet loss implementation"""
    print("\n🧪 Testing Triplet Loss Function...")
    
    tf.random.set_seed(1)
    y_true = (None, None, None)
    y_pred = (
        keras.random.normal([3, 128], mean=6, stddev=0.1, seed=1),
        keras.random.normal([3, 128], mean=1, stddev=1, seed=1),
        keras.random.normal([3, 128], mean=3, stddev=4, seed=1)
    )
    
    loss = triplet_loss(y_true, y_pred)
    print(f"   Loss value: {loss}")
    
    # Additional tests
    y_pred_perfect = ([[1., 1.]], [[1., 1.]], [[1., 1.,]])
    loss = triplet_loss(y_true, y_pred_perfect, 5)
    assert loss == 5, "Wrong value"
    
    y_pred_perfect = ([[1., 1.]], [[1., 1.]], [[0., 0.,]])
    loss = triplet_loss(y_true, y_pred_perfect, 3)
    assert loss == 1., "Wrong value"
    
    y_pred_perfect = ([[1., 1.]], [[0., 0.]], [[1., 1.,]])
    loss = triplet_loss(y_true, y_pred_perfect, 0)
    assert loss == 2., "Wrong value"
    
    print("   ✅ All triplet loss tests passed!")


def test_verify_function(FRmodel, database, test_cases):
    """
    Test the verify function with multiple test cases
    
    Arguments:
        FRmodel -- FaceNet model
        database -- database of face encodings
        test_cases -- list of (image_path, identity, expected_result)
    """
    print("\n🧪 Testing Verify Function...")
    
    for image_path, identity, expected in test_cases:
        dist, door_open = verify(image_path, identity, database, FRmodel)
        
        if door_open == expected:
            print(f"   ✅ {image_path} - {identity}: CORRECT (dist={dist:.4f})")
        else:
            print(f"   ❌ {image_path} - {identity}: WRONG (dist={dist:.4f})")
        
        display_verification_result(image_path, identity, dist, door_open)


def test_recognition_function(FRmodel, database, test_cases):
    """
    Test the who_is_it function with multiple test cases
    
    Arguments:
        FRmodel -- FaceNet model
        database -- database of face encodings
        test_cases -- list of (image_path, expected_identity)
    """
    print("\n🧪 Testing Recognition Function...")
    
    for image_path, expected in test_cases:
        min_dist, identity = who_is_it(image_path, database, FRmodel)
        
        if identity == expected:
            print(f"   ✅ {image_path} - Recognized: {identity} (dist={min_dist:.4f})")
        else:
            print(f"   ❌ {image_path} - Expected: {expected}, Got: {identity} (dist={min_dist:.4f})")
        
        display_image(image_path, title=f"Recognized: {identity}")


def _get_face_images(directory='images'):
    """Return sorted list of face image filenames (skips diagrams/charts)."""
    skip = {'distance_kiank.png', 'distance_matrix.png', 'f_x.png',
            'inception_block1a.png', 'pixel_comparison.png', 'triplet_comparison.png'}
    return sorted(
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f not in skip
    )


# ============================================
# MAIN EXECUTION
# ============================================

def main():
    """Main execution function"""
    
    print("=" * 60)
    print("👤 FaceNet Face Recognition System")
    print("=" * 60)
    
    # ========== TRIPLET LOSS TEST ==========
    print("\n📐 Testing Triplet Loss...")
    test_triplet_loss()
    
    # ========== LOAD MODEL ==========
    print("\n📂 Loading FaceNet model...")
    print("   This may take a moment...")
    
    try:
        FRmodel = load_facenet_model()
        print("   ✅ Model loaded successfully!")
        
        # Display model info
        print(f"\n📊 Model Information:")
        print(f"   Input shape: {FRmodel.inputs}")
        print(f"   Output shape: {FRmodel.outputs}")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")
        print("   Please ensure model files exist at 'keras-facenet-h5/model.json' and 'keras-facenet-h5/model.h5'")
        return
    
    # ========== CREATE DATABASE ==========
    print("\n📚 Creating face database...")
    database = create_database(FRmodel)
    print(f"   ✅ Database created with {len(database)} people")
    
    # ========== DISPLAY DATABASE SAMPLE ==========
    print("\n🖼️ Database sample images:")
    display_comparison("images/danielle.png", "images/kian.jpg", 
                       "Danielle", "Kian")
    
    # ========== TEST VERIFICATION ==========
    print("\n" + "=" * 60)
    print("🔐 Face Verification Tests")
    print("=" * 60)
    
    # Test verification with Younes
    print("\n📸 Testing: camera_0.jpg (should be Younes)")
    dist, door_open = verify("images/camera_0.jpg", "younes", database, FRmodel)
    display_verification_result("images/camera_0.jpg", "younes", dist, door_open)
    
    # Test verification with Kian
    print("\n📸 Testing: camera_2.jpg (should be Kian)")
    dist, door_open = verify("images/camera_2.jpg", "kian", database, FRmodel)
    display_verification_result("images/camera_2.jpg", "kian", dist, door_open)
    
    # ========== TEST RECOGNITION ==========
    print("\n" + "=" * 60)
    print("🔍 Face Recognition Tests")
    print("=" * 60)
    
    # Test recognition with camera_0.jpg
    print("\n📸 Testing recognition on camera_0.jpg...")
    min_dist, identity = who_is_it("images/camera_0.jpg", database, FRmodel)
    display_image("images/camera_0.jpg", title=f"Recognized: {identity}")
    
    # Test recognition with younes.jpg
    print("\n📸 Testing recognition on younes.jpg...")
    min_dist, identity = who_is_it("images/younes.jpg", database, FRmodel)
    display_image("images/younes.jpg", title=f"Recognized: {identity}")
    
    # ========== INTERACTIVE MODE ==========
    print("\n" + "=" * 60)
    print("🎮 Interactive Mode")
    print("=" * 60)
    
    while True:
        print("\nOptions:")
        print("1. Verify a person (check if image matches a specific identity)")
        print("2. Recognize a person (identify who is in the image)")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            identity = input("👤 Enter identity to verify: ").strip()
            if identity not in database:
                print(f"❌ Identity '{identity}' not found in database!")
                continue

            print("  a) Single image")
            print("  b) All images")
            sub = input("  Choose (a/b): ").strip().lower()

            if sub == 'b':
                image_files = _get_face_images()
                for img_name in image_files:
                    img_path = f"images/{img_name}"
                    print(f"\n📸 {img_name}:")
                    dist, door_open = verify(img_path, identity, database, FRmodel)
            else:
                image_name = input("\n📁 Enter image filename: ").strip()
                image_path = f"images/{image_name}"
                dist, door_open = verify(image_path, identity, database, FRmodel)
                display_verification_result(image_path, identity, dist, door_open)
        
        elif choice == '2':
            print("  a) Single image")
            print("  b) All images")
            sub = input("  Choose (a/b): ").strip().lower()

            if sub == 'b':
                image_files = _get_face_images()
                for img_name in image_files:
                    img_path = f"images/{img_name}"
                    print(f"\n📸 {img_name}:")
                    min_dist, identity = who_is_it(img_path, database, FRmodel)
            else:
                image_name = input("\n📁 Enter image filename: ").strip()
                image_path = f"images/{image_name}"
                min_dist, identity = who_is_it(image_path, database, FRmodel)
                display_image(image_path, title=f"Recognized: {identity}")
        
        elif choice == '3':
            print("\n👋 Goodbye! Thanks for using Face Recognition System!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1, 2, or 3.")
    
    # ========== SAVE DATABASE ==========
    save_db = input("\n💾 Save database for future use? (y/n): ").strip().lower()
    if save_db == 'y':
        import pickle
        with open('face_database.pkl', 'wb') as f:
            pickle.dump(database, f)
        print("✅ Database saved as 'face_database.pkl'")
    
    print("\n🎉 Program completed successfully!")


if __name__ == "__main__":
    main()