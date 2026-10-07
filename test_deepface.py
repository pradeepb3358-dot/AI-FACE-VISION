from deepface import DeepFace

print("Starting DeepFace...")

try:
    models = DeepFace.build_model("Facenet")
    print("FaceNet model loaded successfully")
except Exception as e:
    print("ERROR:")
    print(e)