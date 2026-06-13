import streamlit as st
import os
import cv2
import numpy as np
import base64
import hashlib
import requests

try:
    import supabase_db as db
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

HAS_FACE_RECOGNIZER = hasattr(cv2, 'face') and hasattr(cv2.face, 'LBPHFaceRecognizer_create')

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
        .block-container { padding: 1rem 0.5rem; }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
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

IS_CLOUD = not os.path.exists(DB_PATH) or os.access("/", os.W_OK) is False

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
    c.execute("CREATE TABLE IF NOT EXISTS students (admission_id TEXT PRIMARY KEY, name TEXT, roll_no TEXT, department TEXT, semester TEXT, image_path TEXT)")
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

def draw_face_boxes(opencv_img, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(opencv_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(opencv_img, "Face", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return opencv_img

def opencv_to_base64(img):
    _, buf = cv2.imencode('.jpg', img)
    return base64.b64encode(buf).decode()


def train_from_records(records, use_url=False):
    if not HAS_FACE_RECOGNIZER:
        return None, {}

    face_samples = []
    ids = []
    id_map = {}
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for index, (adm_id, img_source) in enumerate(records):
        try:
            if use_url and img_source and img_source.startswith("http"):
                resp = requests.get(img_source, timeout=10)
                file_bytes = np.asarray(bytearray(resp.content), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
            elif not use_url and img_source and os.path.exists(img_source):
                img = cv2.imread(img_source, cv2.IMREAD_GRAYSCALE)
            else:
                continue

            faces = face_cascade.detectMultiScale(img, 1.1, 4)
            for (x, y, w, h) in faces:
                aligned = align_and_resize_face(img, x, y, w, h)
                face_samples.append(aligned)
                ids.append(index)
                id_map[index] = adm_id
        except Exception:
            continue

    if len(face_samples) == 0:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(face_samples, np.array(ids))
    return recognizer, id_map


st.sidebar.title("🏫 PTU Systems")

use_supabase = bool(st.session_state.get("supabase_url") and SUPABASE_AVAILABLE)

if use_supabase:
    st.sidebar.success("☁️ Supabase: Connected")
elif IS_CLOUD:
    st.sidebar.warning("☁️ Cloud Mode (Connect Supabase for full features)")
else:
    st.sidebar.warning("📦 Mode: Local SQLite")

menu = ["📸 Face Scanner", "🔒 Admin Panel"]
choice = st.sidebar.selectbox("Navigation", menu)


if not HAS_FACE_RECOGNIZER:
    st.sidebar.error("⚠️ cv2.face not available. Recognition disabled.")


if choice == "📸 Face Scanner":
    st.header("👁 Real-Time Face Verification")
    st.caption("Take a photo or upload an image. The system detects faces and matches them against the database.")

    cam_col, data_col = st.columns([1.1, 1])

    with cam_col:
        st.subheader("📹 Camera Feed")

        if IS_CLOUD and not use_supabase:
            st.info("Camera requires Supabase connection in cloud mode. Please upload a photo below.")
            img_file = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="cloud_upload")
        else:
            img_file = st.camera_input("Look at the camera and click 'Take Photo'")
            if img_file is None:
                st.markdown("---")
                st.markdown("**Or upload a photo:**")
                img_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"], key="fallback_upload")

    with data_col:
        st.subheader("🔍 Scan Result")
        profile_placeholder = st.empty()

        if img_file is not None:
            file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
            opencv_img = cv2.imdecode(file_bytes, 1)
            gray = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                img_with_boxes = draw_face_boxes(opencv_img.copy(), faces)
                b64_img = opencv_to_base64(img_with_boxes)
                st.image(f"data:image/jpeg;base64,{b64_img}", caption=f"{len(faces)} face(s) detected", use_container_width=True)

                if not HAS_FACE_RECOGNIZER:
                    profile_placeholder.warning("Face detection works but recognition requires opencv-contrib. Install it with: `pip install opencv-contrib-python-headless`")
                else:
                    cache_key = "sb_hash" if use_supabase else "local_hash"
                    recognizer_key = "sb_recognizer" if use_supabase else "local_recognizer"
                    idmap_key = "sb_idmap" if use_supabase else "local_idmap"

                    if use_supabase:
                        current_hash = db.get_db_hash()
                    else:
                        current_hash = get_local_db_hash()

                    if st.session_state.get(cache_key) != current_hash:
                        if use_supabase:
                            students = db.get_all_students()
                            records = [(s["admission_id"], s.get("image_url", "")) for s in students]
                        else:
                            records = [(r[0], r[5]) for r in get_local_students()]

                        rec, idmap = train_from_records(records, use_url=use_supabase)
                        st.session_state[recognizer_key] = rec
                        st.session_state[idmap_key] = idmap
                        st.session_state[cache_key] = current_hash

                    recognizer = st.session_state.get(recognizer_key)
                    id_map = st.session_state.get(idmap_key, {})
                    matched_student = None

                    if recognizer is not None:
                        (x, y, w, h) = faces[0]
                        face_roi = align_and_resize_face(gray, x, y, w, h)
                        label_id, confidence = recognizer.predict(face_roi)

                        if confidence < 50 and label_id in id_map:
                            target_id = id_map[label_id]
                            if use_supabase:
                                s = db.get_student_by_id(target_id)
                                if s:
                                    matched_student = (s.get("admission_id"), s.get("name"), s.get("roll_no"),
                                                       s.get("department"), s.get("semester"), s.get("image_url", ""))
                            else:
                                rows = get_local_students()
                                for r in rows:
                                    if r[0] == target_id:
                                        matched_student = r
                                        break

                    if matched_student:
                        img_path = matched_student[5]
                        if img_path and img_path.startswith("http"):
                            img_src = img_path
                        elif img_path and os.path.exists(img_path):
                            with open(img_path, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode()
                            img_src = f"data:image/jpeg;base64,{b64}"
                        else:
                            img_src = "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"

                        profile_placeholder.markdown(f"""
                        <div class='profile-card-success'>
                            <img src='{img_src}' class='avatar-large'>
                            <div>
                                <h4 style='color:#16a34a; margin:0 0 4px 0;'>Verified Student</h4>
                                <h5 style='margin:0 0 4px 0; color:#0f172a;'><b>Name:</b> {matched_student[1]}</h5>
                                <p style='margin:0; font-size:13px; color:#334155;'><b>ID:</b> {matched_student[0]} | <b>Roll:</b> {matched_student[2]}</p>
                                <p style='margin:0; font-size:13px; color:#0284c7;'><b>Major:</b> {matched_student[3]}</p>
                                <p style='margin:0; font-size:12px; color:#64748b;'><b>Semester:</b> {matched_student[4]}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        profile_placeholder.markdown("""
                        <div class='profile-card-unknown'>
                            <img src='https://cdn-icons-png.flaticon.com/512/57/57708.png' class='avatar-large'>
                            <div>
                                <h4 style='color:#dc2626; margin:0 0 4px 0;'>Unknown Person</h4>
                                <p style='margin:0; font-size:13px; color:#7f1d1d;'>This person is not registered in the database.</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                profile_placeholder.error("No face detected in the image. Please try again.")
        else:
            profile_placeholder.info("Position your face in front of the camera and click 'Take Photo'.")


elif choice == "🔒 Admin Panel":
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

            if use_supabase:
                students = db.get_all_students()
                records = [(s["admission_id"], s["name"], s["roll_no"], s["department"], s["semester"], s.get("image_url","")) for s in students]
            else:
                records = get_local_students()

            if records:
                st.metric(label="Total Registered Students", value=len(records))
                grid = st.columns(3)
                for index, r in enumerate(records):
                    col = grid[index % 3]
                    with col:
                        with st.container(border=True):
                            img_src = r[5] if r[5] and r[5].startswith("http") else (
                                r[5] if r[5] and os.path.exists(r[5]) else
                                "https://cdn-icons-png.flaticon.com/512/3135/3135715.png"
                            )
                            st.image(img_src, width=110)
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
                                file_path = os.path.join(IMG_FOLDER, f"{adm_id}.jpg")
                                save_local_student(adm_id, s_name, r_no, dept, seme,
                                                   file_bytes=uploaded_file.getbuffer(), file_path=file_path)
                                ok, msg = True, "Student registered successfully!"

                            for k in ["sb_hash", "local_hash", "sb_recognizer", "local_recognizer", "sb_idmap", "local_idmap"]:
                                st.session_state.pop(k, None)

                            if ok:
                                st.success(f"Student {s_name} registered successfully!")
                            else:
                                st.error(f"Error: {msg}")

        elif admin_choice == "Edit / Delete Records":
            st.header("Manage Student Records")

            if use_supabase:
                students = db.get_all_students()
                student_list = [(s["admission_id"], s["name"]) for s in students]
            else:
                rows = get_local_students()
                student_list = [(r[0], r[1]) for r in rows]

            if student_list:
                options = [f"{s[0]} - {s[1]}" for s in student_list]
                selected = st.selectbox("Select student to manage", options)
                target_id = selected.split(" - ")[0]

                if use_supabase:
                    s = db.get_student_by_id(target_id)
                    curr_data = (s["name"], s["roll_no"], s["department"], s["semester"], s.get("image_url","")) if s else None
                else:
                    rows = get_local_students()
                    curr_data = None
                    for r in rows:
                        if r[0] == target_id:
                            curr_data = (r[1], r[2], r[3], r[4], r[5])
                            break

                if curr_data:
                    edit_col, view_col = st.columns([1.2, 1])
                    with view_col:
                        st.markdown("**Current Photo**")
                        img_src = curr_data[4]
                        if img_src and img_src.startswith("http"):
                            st.image(img_src, width=150)
                        elif img_src and os.path.exists(img_src):
                            st.image(img_src, width=150)
                        else:
                            st.caption("No photo found.")

                    with edit_col:
                        with st.form("update_form"):
                            st.markdown("### Edit Student Info")
                            up_name = st.text_input("Full Name", value=curr_data[0])
                            up_roll = st.text_input("Roll Number", value=curr_data[1])
                            up_dept = st.selectbox("Department", MAJORS_LIST,
                                index=MAJORS_LIST.index(curr_data[2]) if curr_data[2] in MAJORS_LIST else 0)
                            up_seme = st.selectbox("Semester", SEMESTERS_LIST,
                                index=SEMESTERS_LIST.index(curr_data[3]) if curr_data[3] in SEMESTERS_LIST else 0)
                            if st.form_submit_button("Save Changes", type="primary"):
                                if use_supabase:
                                    ok, msg = db.update_student(target_id, up_name, up_roll, up_dept, up_seme)
                                else:
                                    import sqlite3
                                    conn = sqlite3.connect(DB_PATH)
                                    c = conn.cursor()
                                    c.execute("UPDATE students SET name=?, roll_no=?, department=?, semester=? WHERE admission_id=?",
                                              (up_name, up_roll, up_dept, up_seme, target_id))
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
                    confirm = st.checkbox(f"I confirm I want to delete {curr_data[0]} ({target_id})")
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
            st.markdown("Configure your Supabase connection for cloud database storage.")

            if not SUPABASE_AVAILABLE:
                st.error("Supabase library not installed. Add `supabase` to requirements.txt")

            with st.expander("How to set up Supabase", expanded=not bool(st.session_state.get("supabase_url"))):
                st.markdown("""
                **Steps:**
                1. Go to [supabase.com](https://supabase.com) and create a free account
                2. Create a new project
                3. Go to **SQL Editor** and run this SQL to create the table:
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
                4. Go to **Storage** and create a bucket named `registered_faces` (set it to public)
                5. Copy your **Project URL** and **anon/public Key** from Settings > API
                6. Paste them below
                """)

            with st.form("api_form"):
                supabase_url = st.text_input(
                    "Supabase Project URL",
                    value=st.session_state.get("supabase_url", ""),
                    placeholder="https://xxxxx.supabase.co"
                )
                supabase_key = st.text_input(
                    "Supabase Anon Key",
                    value=st.session_state.get("supabase_key", ""),
                    placeholder="eyJhbGciOiJIUzI1NiIs...",
                    type="password"
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.form_submit_button("Save & Connect", type="primary"):
                        st.session_state["supabase_url"] = supabase_url.strip()
                        st.session_state["supabase_key"] = supabase_key.strip()
                        for k in ["sb_hash", "sb_recognizer", "sb_idmap"]:
                            st.session_state.pop(k, None)
                        if supabase_url.strip() and supabase_key.strip():
                            ok, msg = db.test_connection()
                            if ok:
                                st.success(msg)
                            else:
                                st.error(msg)
                        else:
                            st.warning("Please enter both URL and Key.")
                with col2:
                    if st.form_submit_button("Test Connection"):
                        st.session_state["supabase_url"] = supabase_url.strip()
                        st.session_state["supabase_key"] = supabase_key.strip()
                        ok, msg = db.test_connection()
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                with col3:
                    if st.form_submit_button("Disconnect (Use Local)"):
                        st.session_state["supabase_url"] = ""
                        st.session_state["supabase_key"] = ""
                        for k in ["sb_hash", "sb_recognizer", "sb_idmap"]:
                            st.session_state.pop(k, None)
                        st.info("Switched to local SQLite mode.")
                        st.rerun()

            st.write("---")
            st.subheader("Connection Status")
            if st.session_state.get("supabase_url"):
                st.markdown("**Mode:** Cloud (Supabase)")
                st.markdown(f"**URL:** `{st.session_state.get('supabase_url', '')}`")
                ok, msg = db.test_connection()
                if ok:
                    st.markdown("**Status:** <span class='status-ok'>Connected</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"**Status:** <span class='status-err'>Error - {msg}</span>", unsafe_allow_html=True)
            else:
                st.markdown("**Mode:** Local SQLite")
                st.markdown("**Status:** <span class='status-ok'>Active (Offline Mode)</span>", unsafe_allow_html=True)

    else:
        st.subheader("Admin Login")
        password_input = st.text_input("Enter Admin Password", type="password")
        if st.button("Login"):
            if password_input == ADMIN_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Incorrect Password!")
