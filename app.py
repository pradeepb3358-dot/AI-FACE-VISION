import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import os
import tempfile
from PIL import Image
from io import BytesIO

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VisionX AI Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #16213e 0%, transparent 30%),
        radial-gradient(circle at 90% 20%, #24103f 0%, transparent 30%),
        #070b14;
    color: white;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1400px;
}

.hero {
    padding: 30px;
    border-radius: 25px;
    background: linear-gradient(
        135deg,
        rgba(20,30,60,.95),
        rgba(60,20,80,.90)
    );
    border: 1px solid rgba(255,255,255,.12);
    box-shadow: 0 15px 50px rgba(0,0,0,.35);
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    color: #b9c4d6;
    font-size: 17px;
}

.card {
    padding: 20px;
    border-radius: 20px;
    background: rgba(255,255,255,.055);
    border: 1px solid rgba(255,255,255,.10);
    margin-bottom: 18px;
}

.step {
    color: #8ab4ff;
    font-weight: bold;
    letter-spacing: 1px;
}

.metric-card {
    padding: 20px;
    border-radius: 18px;
    background: rgba(255,255,255,.06);
    border: 1px solid rgba(255,255,255,.10);
    text-align: center;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
}

.metric-label {
    color: #aab5c5;
}

.result {
    padding: 18px;
    border-radius: 16px;
    background: rgba(0, 200, 150, .10);
    border: 1px solid rgba(0, 220, 170, .25);
}

.warning {
    padding: 18px;
    border-radius: 16px;
    background: rgba(255, 170, 0, .10);
    border: 1px solid rgba(255, 170, 0, .25);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.markdown("""
<div class="hero">
    <div class="step">AI + COMPUTER VISION LAB</div>
    <h1>🧠 VisionX Face Intelligence</h1>
    <p>
    Template Matching • Viola-Jones • FaceNet • DeepFace • OpenCV • TensorFlow
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## ⚙️ Control Center")

    mode = st.radio(
        "Input Mode",
        [
            "📁 Upload Image",
            "📷 Camera"
        ]
    )

    st.markdown("---")

    st.markdown("### 🧩 Processing Modules")

    use_preprocess = st.checkbox(
        "OpenCV Preprocessing",
        value=True
    )

    use_template = st.checkbox(
        "Template Matching",
        value=True
    )

    use_viola = st.checkbox(
        "Viola-Jones",
        value=True
    )

    use_facenet = st.checkbox(
        "FaceNet Embedding",
        value=True
    )

    use_deepface = st.checkbox(
        "DeepFace Analysis",
        value=False
    )

    st.markdown("---")

    st.markdown("### 🎯 Template Settings")

    threshold = st.slider(
        "Template Threshold",
        0.30,
        0.99,
        0.70,
        0.01
    )

    st.markdown("---")

    st.caption(
        "VisionX AI Lab\n\n"
        "Built with Python + Streamlit + OpenCV + TensorFlow"
    )

# =========================================================
# IMAGE INPUT
# =========================================================

uploaded = None

if mode == "📁 Upload Image":

    uploaded = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

else:

    uploaded = st.camera_input(
        "Take a picture using your camera"
    )

# =========================================================
# IMAGE DECODER
# =========================================================

def read_image(uploaded_file):

    data = uploaded_file.read()

    image_array = np.frombuffer(
        data,
        dtype=np.uint8
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    return image


# =========================================================
# IMAGE DISPLAY
# =========================================================

if uploaded is None:

    st.info(
        "👆 Upload an image or capture an image using the camera "
        "to start the AI pipeline."
    )

    st.markdown("""
    <div class="card">

    ### 🚀 Processing Pipeline

    **01** Input Image  
    ↓  
    **02** OpenCV Preprocessing  
    ↓  
    **03** Template Matching  
    ↓  
    **04** Viola-Jones Face Detection  
    ↓  
    **05** FaceNet Embedding  
    ↓  
    **06** DeepFace Analysis  
    ↓  
    **07** Similarity Calculation  
    ↓  
    **08** Final AI Report

    </div>
    """, unsafe_allow_html=True)

    st.stop()


image = read_image(uploaded)

if image is None:

    st.error("Unable to read image.")

    st.stop()


original = image.copy()

# =========================================================
# IMAGE INFORMATION
# =========================================================

height, width, channels = image.shape

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

brightness = float(np.mean(gray))

blur_value = float(np.std(gray))

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-value">{width}×{height}</div>
        <div class="metric-label">Resolution</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-value">{channels}</div>
        <div class="metric-label">Channels</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-value">{brightness:.1f}</div>
        <div class="metric-label">Brightness</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
        <div class="metric-value">{blur_value:.1f}</div>
        <div class="metric-label">Intensity STD</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# =========================================================
# STEP 01
# =========================================================

st.markdown(
    '<div class="card"><div class="step">STEP 01</div>'
    '<h2>📷 Input Image</h2></div>',
    unsafe_allow_html=True
)

c1, c2 = st.columns(2)

with c1:

    st.image(
        cv2.cvtColor(original, cv2.COLOR_BGR2RGB),
        caption="Original Input",
        use_container_width=True
    )

with c2:

    st.markdown("""
    ### Input Details

    - Image successfully loaded
    - OpenCV image matrix created
    - RGB/BGR conversion completed
    - Resolution calculated
    - Pixel statistics calculated
    """)

# =========================================================
# STEP 02
# =========================================================

processed = image.copy()

if use_preprocess:

    st.markdown(
        '<div class="card"><div class="step">STEP 02</div>'
        '<h2>🔬 OpenCV Preprocessing</h2></div>',
        unsafe_allow_html=True
    )

    start = time.perf_counter()

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(
        gray
    )

    process_time = time.perf_counter() - start

    a, b, c = st.columns(3)

    with a:
        st.image(
            gray,
            caption="Grayscale",
            use_container_width=True
        )

    with b:
        st.image(
            enhanced,
            caption="Contrast Enhanced",
            use_container_width=True
        )

    with c:
        st.image(
            edges,
            caption="Canny Edges",
            use_container_width=True
        )

    st.success(
        f"Preprocessing completed in {process_time:.4f} seconds"
    )

# =========================================================
# STEP 03 TEMPLATE MATCHING
# =========================================================

template_score = 0.0
template_location = None

if use_template:

    st.markdown(
        '<div class="card"><div class="step">STEP 03</div>'
        '<h2>🎯 Template Matching</h2></div>',
        unsafe_allow_html=True
    )

    st.write(
        "Template Matching searches for a reference pattern "
        "inside the uploaded image."
    )

    template_file = st.file_uploader(
        "Upload template/reference image",
        type=["jpg", "jpeg", "png"],
        key="template"
    )

    if template_file:

        template = read_image(
            template_file
        )

        if template is not None:

            start = time.perf_counter()

            source_gray = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2GRAY
            )

            template_gray = cv2.cvtColor(
                template,
                cv2.COLOR_BGR2GRAY
            )

            th, tw = template_gray.shape

            sh, sw = source_gray.shape

            if th <= sh and tw <= sw:

                result = cv2.matchTemplate(
                    source_gray,
                    template_gray,
                    cv2.TM_CCOEFF_NORMED
                )

                _, max_val, _, max_loc = cv2.minMaxLoc(
                    result
                )

                template_score = float(
                    max_val
                )

                template_location = max_loc

                output = image.copy()

                top_left = max_loc

                bottom_right = (
                    top_left[0] + tw,
                    top_left[1] + th
                )

                cv2.rectangle(
                    output,
                    top_left,
                    bottom_right,
                    (0, 255, 0),
                    3
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

                x, y = top_left

                m1, m2, m3 = st.columns(3)

                with m1:
                    st.metric(
                        "Similarity",
                        f"{template_score:.2%}"
                    )

                with m2:
                    st.metric(
                        "Threshold",
                        f"{threshold:.2%}"
                    )

                with m3:
                    st.metric(
                        "Processing Time",
                        f"{elapsed:.4f}s"
                    )

                st.image(
                    cv2.cvtColor(
                        output,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Template Detection",
                    use_container_width=True
                )

                if template_score >= threshold:

                    st.success(
                        "✅ Template detected successfully."
                    )

                else:

                    st.warning(
                        "⚠️ Template confidence is below threshold."
                    )

            else:

                st.error(
                    "Template must be smaller than the input image."
                )

    else:

        st.info(
            "Upload a reference/template image to activate matching."
        )

# =========================================================
# STEP 04 VIOLA-JONES
# =========================================================

faces = []

if use_viola:

    st.markdown(
        '<div class="card"><div class="step">STEP 04</div>'
        '<h2>👤 Viola-Jones Face Detection</h2></div>',
        unsafe_allow_html=True
    )

    cascade_path = cv2.data.haarcascades + \
        "haarcascade_frontalface_default.xml"

    face_detector = cv2.CascadeClassifier(
        cascade_path
    )

    gray_for_face = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    start = time.perf_counter()

    faces = face_detector.detectMultiScale(
        gray_for_face,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40)
    )

    elapsed = time.perf_counter() - start

    face_output = image.copy()

    for i, (x, y, w, h) in enumerate(faces):

        cv2.rectangle(
            face_output,
            (x, y),
            (x + w, y + h),
            (255, 0, 255),
            3
        )

        cv2.putText(
            face_output,
            f"Face {i + 1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 255),
            2
        )

    f1, f2, f3 = st.columns(3)

    with f1:
        st.metric(
            "Faces Detected",
            len(faces)
        )

    with f2:
        st.metric(
            "Detection Time",
            f"{elapsed:.4f}s"
        )

    with f3:
        status = (
            "DETECTED"
            if len(faces) > 0
            else "NOT FOUND"
        )

        st.metric(
            "Status",
            status
        )

    st.image(
        cv2.cvtColor(
            face_output,
            cv2.COLOR_BGR2RGB
        ),
        caption="Viola-Jones Detection",
        use_container_width=True
    )

# =========================================================
# STEP 05 FACENET
# =========================================================

facenet_status = "Not executed"
embedding_vector = None

if use_facenet:

    st.markdown(
        '<div class="card"><div class="step">STEP 05</div>'
        '<h2>🧬 FaceNet Embedding</h2></div>',
        unsafe_allow_html=True
    )

    st.write(
        "FaceNet converts a detected face into a numerical "
        "embedding vector. Similar faces produce similar vectors."
    )

    if len(faces) > 0:

        try:

            from deepface import DeepFace

            x, y, w, h = faces[0]

            face_crop = image[
                y:y+h,
                x:x+w
            ]

            if face_crop.size > 0:

                face_rgb = cv2.cvtColor(
                    face_crop,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    face_rgb,
                    caption="Face selected for embedding",
                    width=300
                )

                with st.spinner(
                    "Generating FaceNet embedding..."
                ):

                    start = time.perf_counter()

                    temp_path = os.path.join(
                        tempfile.gettempdir(),
                        "visionx_face.jpg"
                    )

                    cv2.imwrite(
                        temp_path,
                        face_crop
                    )

                    embedding_result = DeepFace.represent(
                        img_path=temp_path,
                        model_name="Facenet",
                        detector_backend="opencv",
                        enforce_detection=False
                    )

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    if embedding_result:

                        embedding_vector = np.array(
                            embedding_result[0]["embedding"]
                        )

                        facenet_status = "Completed"

                        e1, e2, e3 = st.columns(3)

                        with e1:
                            st.metric(
                                "Embedding Size",
                                len(embedding_vector)
                            )

                        with e2:
                            st.metric(
                                "Model",
                                "FaceNet"
                            )

                        with e3:
                            st.metric(
                                "Time",
                                f"{elapsed:.2f}s"
                            )

                        st.success(
                            "✅ Face embedding generated."
                        )

                        st.write(
                            "First 10 embedding values:"
                        )

                        st.code(
                            np.round(
                                embedding_vector[:10],
                                4
                            ).tolist()
                        )

        except Exception as e:

            st.warning(
                "FaceNet could not be loaded. "
                "Check DeepFace/TensorFlow installation."
            )

            st.code(
                str(e)
            )

    else:

        st.info(
            "Detect a face first using Viola-Jones."
        )

# =========================================================
# STEP 06 DEEPFACE
# =========================================================

deepface_result = None

if use_deepface:

    st.markdown(
        '<div class="card"><div class="step">STEP 06</div>'
        '<h2>🤖 DeepFace Analysis</h2></div>',
        unsafe_allow_html=True
    )

    if uploaded is not None:

        try:

            from deepface import DeepFace

            with st.spinner(
                "Running DeepFace analysis..."
            ):

                temp_input = os.path.join(
                    tempfile.gettempdir(),
                    "visionx_input.jpg"
                )

                cv2.imwrite(
                    temp_input,
                    image
                )

                start = time.perf_counter()

                deepface_result = DeepFace.analyze(
                    img_path=temp_input,
                    actions=[
                        "age",
                        "gender",
                        "race",
                        "emotion"
                    ],
                    detector_backend="opencv",
                    enforce_detection=False
                )

                elapsed = (
                    time.perf_counter()
                    - start
                )

            if isinstance(
                deepface_result,
                list
            ):

                analysis = deepface_result[0]

            else:

                analysis = deepface_result

            d1, d2, d3 = st.columns(3)

            with d1:
                st.metric(
                    "Predicted Age",
                    analysis.get(
                        "age",
                        "N/A"
                    )
                )

            with d2:
                st.metric(
                    "Gender",
                    analysis.get(
                        "dominant_gender",
                        "N/A"
                    )
                )

            with d3:
                st.metric(
                    "Time",
                    f"{elapsed:.2f}s"
                )

            st.write(
                "Dominant Emotion:",
                analysis.get(
                    "dominant_emotion",
                    "N/A"
                )
            )

            st.write(
                "Dominant Region:",
                analysis.get(
                    "dominant_race",
                    "N/A"
                )
            )

        except Exception as e:

            st.warning(
                "DeepFace analysis failed."
            )

            st.code(
                str(e)
            )

# =========================================================
# STEP 07 FACE COMPARISON
# =========================================================

st.markdown(
    '<div class="card"><div class="step">STEP 07</div>'
    '<h2>🔗 Face Similarity</h2></div>',
    unsafe_allow_html=True
)

reference_face = st.file_uploader(
    "Upload a second/reference face for comparison",
    type=["jpg", "jpeg", "png"],
    key="comparison"
)

if reference_face and embedding_vector is not None:

    try:

        from deepface import DeepFace

        reference_image = read_image(
            reference_face
        )

        reference_path = os.path.join(
            tempfile.gettempdir(),
            "visionx_reference.jpg"
        )

        cv2.imwrite(
            reference_path,
            reference_image
        )

        with st.spinner(
            "Comparing faces..."
        ):

            start = time.perf_counter()

            ref_result = DeepFace.represent(
                img_path=reference_path,
                model_name="Facenet",
                detector_backend="opencv",
                enforce_detection=False
            )

            reference_embedding = np.array(
                ref_result[0]["embedding"]
            )

            # Cosine similarity
            dot_product = np.dot(
                embedding_vector,
                reference_embedding
            )

            norm_a = np.linalg.norm(
                embedding_vector
            )

            norm_b = np.linalg.norm(
                reference_embedding
            )

            similarity = (
                dot_product /
                (norm_a * norm_b + 1e-10)
            )

            similarity = float(
                np.clip(
                    similarity,
                    -1,
                    1
                )
            )

            elapsed = (
                time.perf_counter()
                - start
            )

        percentage = max(
            0,
            similarity
        ) * 100

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Cosine Similarity",
                f"{similarity:.4f}"
            )

        with c2:

            st.metric(
                "Similarity Score",
                f"{percentage:.2f}%"
            )

        with c3:

            st.metric(
                "Comparison Time",
                f"{elapsed:.2f}s"
            )

        if similarity >= 0.70:

            st.success(
                "🟢 High similarity detected."
            )

        elif similarity >= 0.50:

            st.warning(
                "🟡 Moderate similarity detected."
            )

        else:

            st.error(
                "🔴 Low similarity detected."
            )

    except Exception as e:

        st.error(
            "Face comparison failed."
        )

        st.code(
            str(e)
        )

elif reference_face:

    st.info(
        "Generate a FaceNet embedding first."
    )

# =========================================================
# STEP 08 FINAL REPORT
# =========================================================

st.markdown(
    '<div class="card"><div class="step">STEP 08</div>'
    '<h2>📊 Final AI Report</h2></div>',
    unsafe_allow_html=True
)

report = {

    "Input Width": width,

    "Input Height": height,

    "Brightness": round(
        brightness,
        2
    ),

    "Template Score": round(
        template_score,
        4
    ),

    "Faces Detected": len(faces),

    "FaceNet": facenet_status,

    "DeepFace": (
        "Completed"
        if deepface_result is not None
        else "Not executed"
    )
}

report_df = pd.DataFrame(
    list(report.items()),
    columns=[
        "Parameter",
        "Result"
    ]
)

st.dataframe(
    report_df,
    use_container_width=True,
    hide_index=True
)

# =========================================================
# FINAL STATUS
# =========================================================

if len(faces) > 0:

    final_status = "FACE DETECTED"

else:

    final_status = "NO FACE DETECTED"

st.markdown(
    f"""
    <div class="result">
        <h2>🎯 Final Status: {final_status}</h2>
        <p>
        VisionX completed the selected computer vision pipeline.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================================================
# DOWNLOAD REPORT
# =========================================================

csv_data = report_df.to_csv(
    index=False
)

st.download_button(
    label="⬇️ Download AI Report",
    data=csv_data,
    file_name="visionx_ai_report.csv",
    mime="text/csv"
)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#78859a;">
        VisionX AI Lab • OpenCV • TensorFlow • FaceNet • DeepFace • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)