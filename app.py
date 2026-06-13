import streamlit as st
import os
import base64
import hashlib

SUPABASE_AVAILABLE = True
try:
    import supabase_db as db
except Exception:
    SUPABASE_AVAILABLE = False

CV2_AVAILABLE = False
HAS_FACE_RECOGNIZER = False
np = None
cv2 = None
try:
    import numpy as np
    import cv2
    CV2_AVAILABLE = True
    HAS_FACE_RECOGNIZER = hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create')
except Exception:
    pass

WEBRTC_AVAILABLE = False
WEBRTC_USE_TRANSFORM = True
webrtc_streamer = None
WebRtcMode = None
RTCConfiguration = None
VideoTransformerBase = None
VideoProcessorBase = None

# Thread-safe shared dict for live video processor
_LIVE_DATA = {"rec": None, "idmap": {}, "match_result": None}

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    try:
        from streamlit_webrtc import VideoTransformerBase
        WEBRTC_USE_TRANSFORM = True
    except ImportError:
        from streamlit_webrtc import VideoProcessorBase
        WEBRTC_USE_TRANSFORM = False
    WEBRTC_AVAILABLE = True
except Exception:
    pass

DB_PATH = "students.db"
IMG_FOLDER = "registered_faces"
ADMIN_PASSWORD = "admin123"

os.makedirs(IMG_FOLDER, exist_ok=True)

st.set_page_config(
    page_title="PTU Face Recognition System",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @media (max-width: 768px) {
        .block-container { padding: 0.75rem 0.4rem; }
        h1 { font-size: 1.3rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 0.95rem !important; }
        .stTabs [data-baseweb="tab"] { font-size: 0.8rem !important; padding: 6px 12px !important; }
        .profile-card-success, .profile-card-unknown { flex-direction: column !important; text-align: center !important; }
        .avatar-large { width: 70px !important; height: 70px !important; }
    }
    @media (max-width: 480px) {
        .profile-card-success, .profile-card-unknown { padding: 12px !important; }
        .avatar-large { width: 60px !important; height: 60px !important; }
    }
    .profile-card-success {
        background: white; border-left: 6px solid #22c55e; padding: 18px;
        border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        display: flex; gap: 16px; align-items: center; margin-top: 10px;
    }
    .profile-card-unknown {
        background: #fef2f2; border-left: 6px solid #ef4444; padding: 18px;
        border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        display: flex; gap: 16px; align-items: center; margin-top: 10px;
    }
    .avatar-large {
        width: 90px; height: 90px; border-radius: 50%;
        object-fit: cover; border: 3px solid #e2e8f0;
    }
    .status-ok { color: #16a34a; font-weight: bold; }
    .status-err { color: #dc2626; font-weight: bold; }
    .result-container { min-height: 300px; display: flex; flex-direction: column; justify-content: center; }
</style>
""", unsafe_allow_html=True)

MAJORS_LIST = [
    "Civil Engineering",
    "Electrical Power Engineering",
    "Computer Engineering and Information Technology",
    "Electronic Engineering",
    "Mechanical Engineering"
]
SEMESTERS_LIST = [f"Semester {i}" for i in range(1, 11)]

for key, val in {"logged_in": False, "supabase_url": "", "supabase_key": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = val

def get_local_db_hash():
    if not os.path.exists(DB_PATH):
        return "empty"
    try:
        import sqlite3
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT admission_id, image_path FROM students")
        records = c.fetchall()
        conn.close()
        return hashlib.md5(str(records).encode()).hexdigest()
    except Exception:
        return "empty"

def get_local_students():
    import sqlite3
    if not os.path.exists(DB_PATH):
        return []
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT admission_id, name, roll_no, department, semester, image_path FROM students")
        rows = c.fetchall()
        conn.close()
        return rows
    except Exception:
        return []

def save_local_student(adm_id, name, r_no, dept, seme, file_bytes=None, file_path=None):
    import sqlite3
    if file_bytes is not None and file_path:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(file_bytes)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students
                 (admission_id TEXT PRIMARY KEY, name TEXT, roll_no TEXT, department TEXT, semester TEXT, image_path TEXT)""")
    c.execute("INSERT OR REPLACE INTO students VALUES (?, ?, ?, ?, ?, ?)",
              (adm_id, name, r_no, dept, seme, file_path or ""))
    conn.commit()
    conn.close()

def delete_local_student(adm_id):
    import sqlite3
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT image_path FROM students WHERE admission_id=?", (adm_id,))
    row = c.fetchone()
    if row and row[0] and os.path.exists(row[0]):
        os.remove(row[0])
    c.execute("DELETE FROM students WHERE admission_id=?", (adm_id,))
    conn.commit()
    conn.close()

FACE_SIZE = (150, 150)

def align_and_resize_face(gray_img, x, y, w, h):
    face = gray_img[y:y+h, x:x+w]
    face = cv2.resize(face, FACE_SIZE, interpolation=cv2.INTER_CUBIC)
    face = cv2.equalizeHist(face)
    return face

def draw_face_boxes(img, faces, labels=None):
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = labels[i] if labels and i < len(labels) else "Face"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return img

def opencv_to_base64(img):
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode()

def train_from_records(records, use_url=False):
    if not HAS_FACE_RECOGNIZER:
        return None, {}
    import requests
    face_samples = []
    ids = []
    id_map = {}
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    for index, (adm_id, src) in enumerate(records):
        try:
            if use_url and src and src.startswith("http"):
                resp = requests.get(src, timeout=10)
                fb = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(fb, cv2.IMREAD_GRAYSCALE)
            elif not use_url and src and os.path.exists(src):
                img = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            else:
                continue
            faces = cascade.detectMultiScale(img, 1.1, 4)
            for (x, y, w, h) in faces:
                face_samples.append(align_and_resize_face(img, x, y, w, h))
                ids.append(index)
                id_map[index] = adm_id
        except Exception:
            continue
    if not face_samples:
        return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(face_samples, np.array(ids))
    return recognizer, id_map

def get_recognizer(use_supabase):
    cache_key = "sb_hash" if use_supabase else "local_hash"
    rec_key = "sb_recognizer" if use_supabase else "local_recognizer"
    idm_key = "sb_idmap" if use_supabase else "local_idmap"
    cur_hash = db.get_db_hash() if use_supabase else get_local_db_hash()
    if st.session_state.get(cache_key) != cur_hash:
        if use_supabase:
            students = db.get_all_students()
            records = [(s["admission_id"], s.get("image_url", "")) for s in students]
        else:
            records = [(r[0], r[5]) for r in get_local_students()]
        rec, idmap = train_from_records(records, use_url=use_supabase)
        st.session_state[rec_key] = rec
        st.session_state[idm_key] = idmap
        st.session_state[cache_key] = cur_hash
    return st.session_state.get(rec_key), st.session_state.get(idm_key, {})

def lookup_student(target_id, use_supabase):
    if use_supabase:
        s = db.get_student_by_id(target_id)
        if s:
            return (s.get("admission_id"), s.get("name"), s.get("roll_no"),
                    s.get("department"), s.get("semester"), s.get("image_url", ""))
        return None
    for r in get_local_students():
        if r[0] == target_id:
            return r
    return None

def render_student_card(student, placeholder=None):
    img_src = student[5] if student[5] and student[5].startswith("http") else (
        student[5] if student[5] and os.path.exists(student[5]) else
        "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
    )
    card = f"""
    <div class='profile-card-success'>
        <img src='{img_src}' class='avatar-large'>
        <div>
            <h4 style='color:#16a34a; margin:0 0 4px 0;'>Verified Student</h4>
            <h5 style='margin:0 0 4px 0; color:#0f172a;'><b>Name:</b> {student[1]}</h5>
            <p style='margin:0; font-size:13px; color:#334155;'><b>ID:</b> {student[0]} | <b>Roll:</b> {student[2]}</p>
            <p style='margin:0; font-size:13px; color:#0284c7;'><b>Major:</b> {student[3]}</p>
            <p style='margin:0; font-size:12px; color:#64748b;'><b>Semester:</b> {student[4]}</p>
        </div>
    </div>
    """
    if placeholder:
        placeholder.markdown(card, unsafe_allow_html=True)
    return card

def render_unknown_card(placeholder=None):
    card = """
    <div class='profile-card-unknown'>
        <img src='https://cdn-icons-png.flaticon.com/512/57/57708.png' class='avatar-large'>
        <div>
            <h4 style='color:#dc2626; margin:0 0 4px 0;'>Unknown Person</h4>
            <p style='margin:0; font-size:13px; color:#7f1d1d;'>This person is not registered in the database.</p>
        </div>
    </div>
    """
    if placeholder:
        placeholder.markdown(card, unsafe_allow_html=True)
    return card

st.sidebar.title("PTU Systems")
use_supabase = bool(st.session_state.get("supabase_url") and SUPABASE_AVAILABLE)

if CV2_AVAILABLE:
    st.sidebar.success("Camera Module: Ready")
else:
    st.sidebar.warning("Camera Module: Not Available")

if use_supabase:
    st.sidebar.success("Database: Supabase Cloud")
else:
    st.sidebar.info("Database: Local SQLite")

if WEBRTC_AVAILABLE:
    pass
else:
    st.sidebar.info("Live Video: Install streamlit-webrtc")

menu = ["Face Scanner", "Admin Panel"]
choice = st.sidebar.selectbox("Navigation", menu)

if choice == "Face Scanner":
    st.header("Face Recognition Scanner")
    st.caption("Verify a person's identity by scanning their face against the database.")

    if not CV2_AVAILABLE:
        st.error("OpenCV is not available. Install it: `pip install opencv-python-headless numpy`")
        st.stop()

    tab_photo, tab_live = st.tabs([" Take Photo ", " Live Video "])

    with tab_photo:
        st.subheader("Capture a Photo")
        left, right = st.columns([1.1, 1])
        with left:
            img_file = st.camera_input("Click 'Take Photo' to capture", key="take_photo_cam")
            if img_file is None:
                st.markdown("**Or upload a photo:**")
                img_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"], key="photo_upload")
        with right:
            with st.container():
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                ph = st.empty()
                st.markdown('</div>', unsafe_allow_html=True)

                if img_file is not None:
                    fb = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
                    oimg = cv2.imdecode(fb, 1)
                    gray = cv2.cvtColor(oimg, cv2.COLOR_BGR2GRAY)
                    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = cascade.detectMultiScale(gray, 1.1, 4)

                    if len(faces) > 0:
                        boxed = draw_face_boxes(oimg.copy(), faces)
                        st.image(f"data:image/jpeg;base64,{opencv_to_base64(boxed)}",
                                 caption=f"{len(faces)} face(s) detected", use_container_width=True)

                        if not HAS_FACE_RECOGNIZER:
                            st.warning("Face detected. Install opencv-contrib-python-headless for recognition.")
                        else:
                            rec, idmap = get_recognizer(use_supabase)
                            matched = None
                            if rec is not None:
                                (x, y, w, h) = faces[0]
                                roi = align_and_resize_face(gray, x, y, w, h)
                                lid, conf = rec.predict(roi)
                                if conf < 50 and lid in idmap:
                                    matched = lookup_student(idmap[lid], use_supabase)
                            if matched:
                                render_student_card(matched, ph)
                            else:
                                render_unknown_card(ph)
                    else:
                        ph.error("No face detected.")
                else:
                    ph.info("Position your face and click 'Take Photo'.")

    with tab_live:
        st.subheader("Live Video Feed")
        st.caption("Real-time face detection and recognition. Allow camera access when prompted.")

        if not WEBRTC_AVAILABLE:
            st.warning("Live Video mode requires `streamlit-webrtc`. Install it:")
            st.code("pip install streamlit-webrtc")
            st.info("In the meantime, use the 'Take Photo' tab above.")
            st.stop()

        if not HAS_FACE_RECOGNIZER:
            st.warning("Recognition unavailable (opencv-contrib not installed). Detection only.")

        rec, idmap = get_recognizer(use_supabase)
        _LIVE_DATA["rec"] = rec
        _LIVE_DATA["idmap"] = idmap
        _LIVE_DATA["match_result"] = None

        if WEBRTC_AVAILABLE:
            if WEBRTC_USE_TRANSFORM:
                class FaceProcessor(VideoTransformerBase):
                    def __init__(self):
                        self.cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        )
                        self.last_match_id = None

                    def transform(self, frame):
                        data = _LIVE_DATA
                        img = frame.to_ndarray(format="bgr24")
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.cascade.detectMultiScale(gray, 1.1, 4)

                        labels = []
                        current_match = None
                        for (x, y, w, h) in faces:
                            label = "Face"
                            r = data["rec"]
                            im = data["idmap"]
                            if r is not None and im:
                                roi = align_and_resize_face(gray, x, y, w, h)
                                lid, conf = r.predict(roi)
                                if conf < 50 and lid in im:
                                    target = im[lid]
                                    label = target
                                    if current_match is None:
                                        current_match = target
                            labels.append(label)

                        draw_face_boxes(img, faces, labels)

                        if current_match and current_match != self.last_match_id:
                            self.last_match_id = current_match
                            data["match_result"] = current_match
                        elif current_match is None:
                            self.last_match_id = None
                            data["match_result"] = None

                        return av.VideoFrame.from_ndarray(img, format="bgr24")
            else:
                class FaceProcessor(VideoProcessorBase):
                    def __init__(self):
                        self.cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                        )
                        self.last_match_id = None

                    def recv(self, frame):
                        data = _LIVE_DATA
                        img = frame.to_ndarray(format="bgr24")
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        faces = self.cascade.detectMultiScale(gray, 1.1, 4)

                        labels = []
                        current_match = None
                        for (x, y, w, h) in faces:
                            label = "Face"
                            r = data["rec"]
                            im = data["idmap"]
                            if r is not None and im:
                                roi = align_and_resize_face(gray, x, y, w, h)
                                lid, conf = r.predict(roi)
                                if conf < 50 and lid in im:
                                    target = im[lid]
                                    label = target
                                    if current_match is None:
                                        current_match = target
                            labels.append(label)

                        draw_face_boxes(img, faces, labels)

                        if current_match and current_match != self.last_match_id:
                            self.last_match_id = current_match
                            data["match_result"] = current_match
                        elif current_match is None:
                            self.last_match_id = None
                            data["match_result"] = None

                        return av.VideoFrame.from_ndarray(img, format="bgr24")

            factory_param = "video_transformer_factory" if WEBRTC_USE_TRANSFORM else "video_processor_factory"

            kw = {
                "key": "ptu-live-video",
                "mode": WebRtcMode.SENDRECV,
                factory_param: FaceProcessor,
                "rtc_configuration": {
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                },
                "media_stream_constraints": {
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}},
                    "audio": False,
                },
                "async_transform": True,
            }

            webrtc_streamer(**kw)

            st.markdown("---")
            st.subheader("Live Recognition Result")
            live_ph = st.empty()

            if _LIVE_DATA["match_result"]:
                student = lookup_student(_LIVE_DATA["match_result"], use_supabase)
                if student:
                    render_student_card(student, live_ph)
                else:
                    render_unknown_card(live_ph)
            else:
                live_ph.info("Waiting for face detection...")

elif choice == "Admin Panel":
    if st.session_state.logged_in:
        st.sidebar.success("Logged in as Admin")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

        admin_menu = ["Dashboard", "Register Student", "Edit / Delete Records", "API Settings"]
        admin_choice = st.sidebar.radio("Admin Functions", admin_menu)
        st.write("---")

        if admin_choice == "Dashboard":
            st.header("Dashboard")
            records = db.get_all_students() if use_supabase else get_local_students()
            if use_supabase:
                records = [(s["admission_id"], s["name"], s["roll_no"], s["department"], s["semester"], s.get("image_url","")) for s in records]
            if records:
                st.metric(label="Total Registered Students", value=len(records))
                grid = st.columns(3)
                for i, r in enumerate(records):
                    col = grid[i % 3]
                    with col:
                        with st.container(border=True):
                            src = r[5] if r[5] and r[5].startswith("http") else (
                                r[5] if r[5] and os.path.exists(r[5]) else
                                "https://cdn-icons-png.flaticon.com/512/3135/3135715.png")
                            st.image(src, width=110)
                            st.markdown(f"**{r[1]}**")
                            st.caption(f"ID: {r[0]} | Major: {r[3]}")
            else:
                st.info("No students registered yet.")

        elif admin_choice == "Register Student":
            st.header("Register New Student")
            with st.form("reg_form", clear_on_submit=True):
                adm_id = st.text_input("Admission ID *")
                s_name = st.text_input("Full Name *")
                r_no = st.text_input("Roll Number *")
                dept = st.selectbox("Department (Major)", MAJORS_LIST)
                seme = st.selectbox("Semester", SEMESTERS_LIST)
                uploaded_file = st.file_uploader("Upload Profile Image *", type=["jpg", "png", "jpeg"])
                if st.form_submit_button("Register Student", type="primary"):
                    if not (adm_id and s_name and r_no and uploaded_file):
                        st.error("Please fill all required fields.")
                    else:
                        with st.spinner("Registering..."):
                            if use_supabase:
                                img_bytes = uploaded_file.read()
                                image_url = db.upload_image(adm_id, img_bytes)
                                if image_url is None:
                                    st.error("Failed to upload image to Supabase.")
                                    st.stop()
                                ok, msg = db.add_student(adm_id, s_name, r_no, dept, seme, image_url)
                            else:
                                fp = os.path.join(IMG_FOLDER, f"{adm_id}.jpg")
                                save_local_student(adm_id, s_name, r_no, dept, seme,
                                                   file_bytes=uploaded_file.getbuffer(), file_path=fp)
                                ok, msg = True, "Student registered successfully!"
                            for k in ["sb_hash", "local_hash", "sb_recognizer", "local_recognizer", "sb_idmap", "local_idmap"]:
                                st.session_state.pop(k, None)
                            st.success(msg) if ok else st.error(msg)

        elif admin_choice == "Edit / Delete Records":
            st.header("Manage Student Records")
            if use_supabase:
                students = db.get_all_students()
                student_list = [(s["admission_id"], s["name"]) for s in students]
            else:
                student_list = [(r[0], r[1]) for r in get_local_students()]
            if student_list:
                selected = st.selectbox("Select student", [f"{s[0]} - {s[1]}" for s in student_list])
                target_id = selected.split(" - ")[0]
                curr_data = None
                if use_supabase:
                    s = db.get_student_by_id(target_id)
                    if s:
                        curr_data = (s["name"], s["roll_no"], s["department"], s["semester"], s.get("image_url",""))
                else:
                    for r in get_local_students():
                        if r[0] == target_id:
                            curr_data = (r[1], r[2], r[3], r[4], r[5])
                            break
                if curr_data:
                    ec, vc = st.columns([1.2, 1])
                    with vc:
                        st.markdown("**Current Photo**")
                        src = curr_data[4]
                        if src and src.startswith("http"):
                            st.image(src, width=150)
                        elif src and os.path.exists(src):
                            st.image(src, width=150)
                        else:
                            st.caption("No photo found.")
                    with ec:
                        with st.form("update_form"):
                            up_n = st.text_input("Full Name", value=curr_data[0])
                            up_r = st.text_input("Roll Number", value=curr_data[1])
                            up_d = st.selectbox("Department", MAJORS_LIST,
                                index=MAJORS_LIST.index(curr_data[2]) if curr_data[2] in MAJORS_LIST else 0)
                            up_s = st.selectbox("Semester", SEMESTERS_LIST,
                                index=SEMESTERS_LIST.index(curr_data[3]) if curr_data[3] in SEMESTERS_LIST else 0)
                            if st.form_submit_button("Save Changes", type="primary"):
                                if use_supabase:
                                    ok, msg = db.update_student(target_id, up_n, up_r, up_d, up_s)
                                else:
                                    import sqlite3
                                    conn = sqlite3.connect(DB_PATH)
                                    c = conn.cursor()
                                    c.execute("UPDATE students SET name=?, roll_no=?, department=?, semester=? WHERE admission_id=?",
                                              (up_n, up_r, up_d, up_s, target_id))
                                    conn.commit()
                                    conn.close()
                                    ok, msg = True, "Updated!"
                                for k in ["sb_hash", "local_hash", "sb_recognizer", "local_recognizer", "sb_idmap", "local_idmap"]:
                                    st.session_state.pop(k, None)
                                if ok:
                                    st.success(msg)
                                    st.rerun()
                                else:
                                    st.error(msg)
                    st.write("---")
                    st.markdown("### Danger Zone")
                    confirm = st.checkbox(f"I confirm delete {curr_data[0]} ({target_id})")
                    if st.button("Delete Student", type="secondary", disabled=not confirm):
                        if use_supabase:
                            ok, msg = db.delete_student(target_id)
                        else:
                            delete_local_student(target_id)
                            ok, msg = True, "Deleted!"
                        for k in ["sb_hash", "local_hash", "sb_recognizer", "local_recognizer", "sb_idmap", "local_idmap"]:
                            st.session_state.pop(k, None)
                        if ok:
                            st.success(msg)
                            st.rerun()
                        else:
                            st.error(msg)
            else:
                st.info("No records found.")

        elif admin_choice == "API Settings":
            st.header("Supabase API Settings")
            if not SUPABASE_AVAILABLE:
                st.error("Supabase package not installed. Add `supabase` to requirements.txt and redeploy.")
            else:
                with st.expander("Setup Guide", expanded=not bool(st.session_state.get("supabase_url"))):
                    st.markdown("""
                    1. Create a free project at [supabase.com](https://supabase.com)
                    2. In **SQL Editor**, run:
                    ```sql
                    CREATE TABLE students (
                        admission_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        roll_no TEXT NOT NULL,
                        department TEXT NOT NULL,
                        semester TEXT NOT NULL,
                        image_url TEXT
                    );
                    ALTER TABLE students ENABLE ROW LEVEL SECURITY;
                    CREATE POLICY "Allow all" ON students FOR ALL USING (true) WITH CHECK (true);
                    ```
                    3. In **Storage**, create bucket `registered_faces` (public)
                    4. Copy **Project URL** + **Anon Key** from Settings > API
                    """)
                with st.form("api_form"):
                    su = st.text_input("Supabase Project URL", value=st.session_state.get("supabase_url", ""),
                                       placeholder="https://xxxxx.supabase.co")
                    sk = st.text_input("Supabase Anon Key", value=st.session_state.get("supabase_key", ""),
                                       placeholder="eyJhbGciOiJIUzI1NiIs...", type="password")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        if st.form_submit_button("Save & Connect", type="primary"):
                            st.session_state["supabase_url"] = su.strip()
                            st.session_state["supabase_key"] = sk.strip()
                            for k in ["sb_hash", "sb_recognizer", "sb_idmap"]:
                                st.session_state.pop(k, None)
                            if su.strip() and sk.strip():
                                ok, msg = db.test_connection()
                            if ok:
                                st.success(msg)
                            else:
                                st.error(msg)
                    with c2:
                        if st.form_submit_button("Test Connection"):
                            st.session_state["supabase_url"] = su.strip()
                            st.session_state["supabase_key"] = sk.strip()
                            ok, msg = db.test_connection()
                            st.success(msg) if ok else st.error(msg)
                    with c3:
                        if st.form_submit_button("Disconnect"):
                            st.session_state["supabase_url"] = ""
                            st.session_state["supabase_key"] = ""
                            for k in ["sb_hash", "sb_recognizer", "sb_idmap"]:
                                st.session_state.pop(k, None)
                            st.rerun()
                st.write("---")
                st.subheader("Connection Status")
                if st.session_state.get("supabase_url"):
                    st.markdown("**Mode:** Cloud (Supabase)")
                    ok, msg = db.test_connection()
                    tag = "status-ok" if ok else "status-err"
                    st.markdown(f"**Status:** <span class='{tag}'>{'Connected' if ok else msg}</span>", unsafe_allow_html=True)
                else:
                    st.markdown("**Mode:** Local SQLite")
                    st.markdown("**Status:** <span class='status-ok'>Active</span>", unsafe_allow_html=True)

    else:
        st.subheader("Admin Login")
        pw = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if pw == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect Password!")
