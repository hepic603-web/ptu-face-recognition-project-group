import hashlib

try:
    import streamlit as st
    from supabase import create_client, Client
    SUPABASE_PKG_AVAILABLE = True
except Exception:
    SUPABASE_PKG_AVAILABLE = False
    st = None

BUCKET_NAME = "registered_faces"

def _check():
    if not SUPABASE_PKG_AVAILABLE:
        return None, "supabase package not installed"
    return True, None

def get_client():
    ok, err = _check()
    if not ok:
        return None
    url = st.session_state.get("supabase_url", "").strip()
    key = st.session_state.get("supabase_key", "").strip()
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None

def test_connection():
    ok, err = _check()
    if not ok:
        return False, err
    client = get_client()
    if client is None:
        return False, "Supabase URL or Key not configured."
    try:
        client.table("students").select("admission_id").limit(1).execute()
        return True, "Connected successfully!"
    except Exception as e:
        return False, f"Connection failed: {e}"

def ensure_bucket():
    client = get_client()
    if client is None:
        return
    try:
        buckets = client.storage.list_buckets()
        names = [b.name for b in buckets]
        if BUCKET_NAME not in names:
            client.storage.create_bucket(BUCKET_NAME, public=True)
    except Exception:
        pass

def upload_image(admission_id, file_bytes):
    client = get_client()
    if client is None:
        return None
    ensure_bucket()
    path = f"{admission_id}.jpg"
    try:
        client.storage.from_(BUCKET_NAME).upload(
            path=path, file=file_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        return client.storage.from_(BUCKET_NAME).get_public_url(path)
    except Exception:
        return None

def delete_image(admission_id):
    client = get_client()
    if client is None:
        return
    try:
        client.storage.from_(BUCKET_NAME).remove([f"{admission_id}.jpg"])
    except Exception:
        pass

def get_all_students():
    client = get_client()
    if client is None:
        return []
    try:
        resp = client.table("students").select("*").execute()
        return resp.data
    except Exception:
        return []

def get_student_by_id(admission_id):
    client = get_client()
    if client is None:
        return None
    try:
        resp = client.table("students").select("*").eq("admission_id", admission_id).execute()
        return resp.data[0] if resp.data else None
    except Exception:
        return None

def add_student(admission_id, name, roll_no, department, semester, image_url):
    client = get_client()
    if client is None:
        return False, "Supabase not connected."
    try:
        client.table("students").upsert({
            "admission_id": admission_id, "name": name, "roll_no": roll_no,
            "department": department, "semester": semester, "image_url": image_url
        }).execute()
        return True, "Student registered successfully!"
    except Exception as e:
        return False, str(e)

def update_student(admission_id, name, roll_no, department, semester):
    client = get_client()
    if client is None:
        return False, "Supabase not connected."
    try:
        client.table("students").update({
            "name": name, "roll_no": roll_no,
            "department": department, "semester": semester
        }).eq("admission_id", admission_id).execute()
        return True, "Student updated successfully!"
    except Exception as e:
        return False, str(e)

def delete_student(admission_id):
    client = get_client()
    if client is None:
        return False, "Supabase not connected."
    try:
        delete_image(admission_id)
        client.table("students").delete().eq("admission_id", admission_id).execute()
        return True, "Student deleted successfully!"
    except Exception as e:
        return False, str(e)

def get_db_hash():
    students = get_all_students()
    raw = str([(s.get("admission_id"), s.get("image_url")) for s in students])
    return hashlib.md5(raw.encode()).hexdigest()
