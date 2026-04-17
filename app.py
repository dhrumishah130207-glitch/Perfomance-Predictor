from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import csv, os, json
from datetime import datetime
from functools import wraps
import joblib

app = Flask(__name__)
app.secret_key = 'predictedu_secret_2025_nexus'

# ──────────────────────────────────────────
#  LOAD ML MODEL
# ──────────────────────────────────────────
MODEL_PATH = 'model.pkl'
model_ml = None
if os.path.isfile(MODEL_PATH):
    try:
        model_ml = joblib.load(MODEL_PATH)
        print("✅ ML model loaded")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}")
else:
    print("⚠️  model.pkl not found — run train_model.py first")

# ──────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────
CSV_FILE       = 'student_records.csv'
ADMIN_CSV_FILE = 'admin_all_records.csv'
ADMIN_PASSWORD = 'nexus2025'

CSV_HEADERS = [
    'Timestamp', 'Name', 'Gender', 'Age', 'Department',
    'Attendance(%)', 'Study Hours/Day', 'Prev Score(%)',
    'Assignments(%)', 'Sleep', 'Extracurricular', 'Parental Education',
    'Interests', 'Suggested Events',
    'Overall Subject %',
    'Predicted Score(%)', 'Grade', 'Performance Level'
]

# ──────────────────────────────────────────
#  UTILITIES
# ──────────────────────────────────────────
def calculate_overall_percentage(subjects):
    if not subjects:
        return 0
    total_obtained, total_max = 0, 0
    for sub in subjects:
        try:
            total_obtained += float(sub.get('marks', 0))
            total_max      += float(sub.get('total', 0))
        except:
            continue
    return round((total_obtained / total_max) * 100) if total_max else 0


def suggest_events(interests):
    event_map = {
        "Coding & Technology":      ["Hackathon", "App-a-thon", "AI Workshop"],
        "Music":                    ["Battle of Bands", "Open Mic", "Music Fest"],
        "Sports":                   ["Inter-College Tournament", "Annual Sports Meet", "Fitness Blitz"],
        "Art & Design":             ["Design Expo", "Graffiti Wall", "Rangoli Contest"],
        "Entrepreneurship":         ["Startup Pitch Day", "Business Plan Contest", "E-Summit"],
        "Science & Research":       ["Science Expo", "Research Showcase", "Quiz Bowl"],
        "Gaming & Esports":         ["Esports Cup", "LAN Gaming Fest", "BGMI League"],
        "Debate & Public Speaking": ["Debate Championship", "MUN", "Elocution Contest"],
        "Literature & Writing":     ["Creative Writing Contest", "Poetry Slam", "Story-telling Night"],
        "Photography & Film":       ["Photography Contest", "Short Film Festival", "Reel Challenge"],
        "Social Service & NGO":     ["Community Drive", "Awareness Campaign", "Blood Donation Camp"],
        "Dance & Performing Arts":  ["Dance Battle", "Drama Fest", "Nukkad Natak"],
    }
    events, seen = [], set()
    for interest in interests:
        for ev in event_map.get(interest, []):
            if ev not in seen:
                seen.add(ev)
                events.append(ev)
    return events[:4]


def write_to_csv(filepath, data, result, overall_pct):
    file_exists = os.path.isfile(filepath)
    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Timestamp':          datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Name':               data.get('name', 'Unknown'),
            'Gender':             data.get('gender', '-'),
            'Age':                data.get('age', '-'),
            'Department':         data.get('dept', '-'),
            'Attendance(%)':      data.get('attendance'),
            'Study Hours/Day':    data.get('study_hours'),
            'Prev Score(%)':      data.get('prev_score'),
            'Assignments(%)':     data.get('assignments'),
            'Sleep':              data.get('sleep'),
            'Extracurricular':    data.get('extracurricular'),
            'Parental Education': data.get('parental_edu'),
            'Interests':          ', '.join(data.get('interests', [])) if isinstance(data.get('interests'), list) else data.get('interests', ''),
            'Suggested Events':   ', '.join(data.get('events', [])),
            'Overall Subject %':  overall_pct,
            'Predicted Score(%)': result['score'],
            'Grade':              result['grade'],
            'Performance Level':  result['label'].split('—')[-1].strip()
        })


def predict_performance(data):
    subjects    = data.get('subjects', [])
    overall_pct = calculate_overall_percentage(subjects)

    att   = float(data.get('attendance',   75))
    study = float(data.get('study_hours',   4))
    prev  = overall_pct if overall_pct > 0 else float(data.get('prev_score', 65))

    math_score    = overall_pct if overall_pct > 0 else prev
    science_score = overall_pct if overall_pct > 0 else prev
    english_score = overall_pct if overall_pct > 0 else prev

    for sub in subjects:
        name = sub.get('name', '').lower()
        try:
            pct = round((float(sub.get('marks', 0)) / float(sub.get('total', 1))) * 100)
        except:
            pct = overall_pct
        if 'math'  in name:                                          math_score    = pct
        elif any(k in name for k in ['science','phy','chem','bio']): science_score = pct
        elif 'eng' in name:                                          english_score = pct

    # ── ML model prediction ──
    if model_ml is not None:
        features = [[study, att, prev, math_score, science_score, english_score]]
        score = round(float(model_ml.predict(features)[0]))
    else:
        # Fallback formula if model not loaded
        score = round(att * 0.28 + min(study / 8, 1) * 100 * 0.22 + prev * 0.25 + overall_pct * 0.25)

    score = min(100, max(0, score))

    if   score >= 85: grade, label = 'A', f"Predicted: {score}% — Excellent Performance"
    elif score >= 70: grade, label = 'B', f"Predicted: {score}% — Good Performance"
    elif score >= 55: grade, label = 'C', f"Predicted: {score}% — Average Performance"
    elif score >= 40: grade, label = 'D', f"Predicted: {score}% — Below Average"
    else:             grade, label = 'F', f"Predicted: {score}% — Failing Risk"

    def rate(val, t):
        return 'Excellent' if val >= t[0] else 'Good' if val >= t[1] else 'Moderate' if val >= t[2] else 'Low'

    breakdown = {
        'attendance': rate(att,   [90, 75, 60]),
        'study':      rate(study * 10, [70, 50, 30]),
        'prev':       rate(prev,  [85, 70, 55]),
        'assign':     rate(float(data.get('assignments', 70)), [90, 75, 60]),
        'lifestyle':  'Balanced' if data.get('sleep') == '7-8 hours' else 'Average'
    }

    return {
        'score':              score,
        'grade':              grade,
        'label':              label,
        'breakdown':          breakdown,
        'overall_percentage': overall_pct
    }


# ──────────────────────────────────────────
#  AUTH
# ──────────────────────────────────────────
def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('admin_logged_in'):
            return redirect(url_for('admin_login'))
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────────────────────
#  PUBLIC ROUTES
# ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    interests_raw = data.get('interests', [])
    data['interests'] = [i.strip() for i in interests_raw.split(';') if i.strip()] if isinstance(interests_raw, str) else interests_raw

    overall_pct = calculate_overall_percentage(data.get('subjects', []))
    result      = predict_performance(data)
    events      = suggest_events(data['interests'])

    result['events'] = events
    data['events']   = events

    write_to_csv(CSV_FILE,       data, result, overall_pct)
    write_to_csv(ADMIN_CSV_FILE, data, result, overall_pct)

    return jsonify(result)


@app.route('/records')
def records():
    rows = []
    if os.path.isfile(CSV_FILE):
        with open(CSV_FILE, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    return jsonify(rows)


@app.route('/clear_records', methods=['POST'])
def clear_records():
    if os.path.isfile(CSV_FILE):
        os.remove(CSV_FILE)
    return jsonify({'status': 'cleared'})


@app.route('/model/info')
def model_info():
    if model_ml is None:
        return jsonify({'status': '❌ Model not loaded', 'error': 'Run train_model.py first'})
    try:
        sample = round(float(model_ml.predict([[6, 80, 70, 75, 72, 74]])[0]), 2)
    except:
        sample = 'N/A'
    return jsonify({
        'model_type':       type(model_ml).__name__,
        'n_estimators':     getattr(model_ml, 'n_estimators', 'N/A'),
        'n_features':       getattr(model_ml, 'n_features_in_', 'N/A'),
        'feature_names':    ['Study_Hours', 'Attendance', 'Prev_Score', 'Math', 'Science', 'English'],
        'sample_prediction': sample,
        'status':           '✅ ML Model Active'
    })


# ──────────────────────────────────────────
#  ADMIN ROUTES
# ──────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin_panel():
    return render_template('admin.html')


@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        if request.form.get('password', '') == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            return redirect(url_for('admin_panel'))
        error = 'Invalid password'
    return render_template('admin_login.html', error=error)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin_logged_in', None)
    return redirect(url_for('admin_login'))


@app.route('/admin/records')
@admin_required
def admin_records():
    rows = []
    if os.path.isfile(ADMIN_CSV_FILE):
        with open(ADMIN_CSV_FILE, newline='', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
    return jsonify(rows)


# ──────────────────────────────────────────
#  RUN
# ──────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, port=5000)
