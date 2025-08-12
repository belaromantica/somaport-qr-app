import os
import re
import qrcode
import sqlite3
import uuid
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, jsonify, session, send_file
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename
import logging
import time
from flask_cors import CORS
import zipfile

# --- Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-for-somaport'
CORS(app, expose_headers=['X-Processing-Time'])

# --- Configuration des Chemins (Local vs. Serveur) ---
if 'PYTHONANYWHERE_DOMAIN' in os.environ:
    # Sur le serveur PythonAnywhere
    BASE_DIR = os.path.join('/home', os.environ.get('USER', 'pythonweb12'), 'somaport-qr-app')
    TEMP_DIR = '/tmp'
    DATABASE_FILE = os.path.join(BASE_DIR, 'somaport_qr.db')
else:
    # En local sur votre machine
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    DATABASE_FILE = os.path.join(BASE_DIR, 'somaport_qr_local.db')
    os.makedirs(TEMP_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 3 * 1024  # 3 KB


# --- Base de Données ---
def init_database():
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, original_filename TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP, processing_time REAL, 
            status TEXT, bon_delivrer TEXT, conteneur TEXT, permis_douane TEXT, 
            date_sortie TEXT, file_size INTEGER, error_message TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_database()


def log_file_processing(session_id, filename, status, **kwargs):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    if status == 'started':
        cursor.execute(
            "INSERT INTO files (session_id, original_filename, status, file_size) VALUES (?, ?, ?, ?)",
            (session_id, filename, status, kwargs.get('file_size', 0))
        )
    else:
        cursor.execute(
            """UPDATE files SET status = ?, processing_time = ?, bon_delivrer = ?, conteneur = ?, 
               permis_douane = ?, date_sortie = ?, error_message = ?
               WHERE id = (SELECT max(id) FROM files WHERE session_id = ? AND original_filename = ?)""",
            (status, kwargs.get('processing_time'), kwargs.get('bon'), kwargs.get('conteneur'),
             kwargs.get('permis'), kwargs.get('date_sortie'), kwargs.get('error_message'),
             session_id, filename)
        )
    conn.commit()
    conn.close()


# --- Fonctions de Traitement PDF (INCHANGÉES) ---
def extract_data_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            text = text.replace('\xa0', ' ').replace('\n', ' ').strip()
            patterns = {
                'bon': [r'N[°º]?\s*DU\s*BON\s*A\s*DELIVRER\s*[:\-]?\s*([A-Z0-9\-\/]+)',
                        r'BON\s*A\s*DELIVRER\s*[:\-]?\s*([A-Z0-9\-\/]+)'],
                'conteneur': [r'CONTENEUR\s*[:\-]?\s*([A-Z]{4}[0-9]{7})', r'\b([A-Z]{4}[0-9]{7})\b'],
                'permis': [r'NUM[ÉE]RO\s*DE\s*PERMIS\s*DE\s*DOUANE\s*[:\-]?\s*([0-9\/\-]+)',
                           r'PERMIS\s*DE\s*DOUANE\s*[:\-]?\s*([0-9\/\-]+)'],
                'date_sortie': [r'DATE\s*LIMITE\s*DE\s*SORTIE\s*LE\s*[:\-]?\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})',
                                r'Date\s*limite\s*de\s*sortie\s*le\s*[:\-]?\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})']
            }

            def extract(field_patterns):
                for pattern in field_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match: return match.group(1).strip()
                return "N/A"

            return {key: extract(value) for key, value in patterns.items()}
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
        return {key: "N/A" for key in patterns}


def generate_qr_code(data, output_path):
    qr_text = f"Bon à délivrer: {data['bon']}\nConteneur: {data['conteneur']}\nPermis de Douane: {data['permis']}\nDate limite sortie: {data['date_sortie']}"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
    qr.add_data(qr_text)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image.save(output_path)


def add_qr_to_pdf(pdf_path, qr_path, output_pdf_path):
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    last_page = reader.pages[-1]
    page_width = float(last_page.mediabox.width)
    page_height = float(last_page.mediabox.height)
    qr_display_size = 100
    x_pos = page_width - qr_display_size - 18
    y_pos = page_height - qr_display_size - 25
    packet = BytesIO()
    can = canvas.Canvas(packet, pagesize=(page_width, page_height))
    can.drawImage(qr_path, x_pos, y_pos, width=qr_display_size, height=qr_display_size, mask='auto')
    can.save()
    packet.seek(0)
    qr_pdf = PdfReader(packet)
    for i in range(len(reader.pages) - 1):
        writer.add_page(reader.pages[i])
    last_page.merge_page(qr_pdf.pages[0])
    writer.add_page(last_page)
    with open(output_pdf_path, 'wb') as f_out:
        writer.write(f_out)


# --- Routes Flask ---
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    session_id = get_session_id()
    if 'pdf_file' not in request.files: return jsonify({'error': 'Aucun fichier reçu'}), 400
    file = request.files['pdf_file']
    if file.filename == '': return jsonify({'error': 'Aucun fichier sélectionné'}), 400

    filename = secure_filename(file.filename)
    unique_id = str(uuid.uuid4())
    input_path = os.path.join(TEMP_DIR, f"{unique_id}_{filename}")
    output_path = os.path.join(TEMP_DIR, f"qr_{unique_id}_{filename}")
    qr_path = os.path.join(TEMP_DIR, f"qr_img_{unique_id}.png")

    try:
        start_time = time.time()
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        if file_size > MAX_FILE_SIZE:
            log_file_processing(session_id, filename, 'error', error_message='File size exceeded')
            return jsonify({'error': f'Le fichier dépasse la limite de {MAX_FILE_SIZE / 1024} KB.'}), 400

        log_file_processing(session_id, filename, 'started', file_size=file_size)

        data = extract_data_from_pdf(input_path)
        generate_qr_code(data, qr_path)
        add_qr_to_pdf(input_path, qr_path, output_path)

        processing_time = time.time() - start_time
        log_file_processing(session_id, filename, 'completed', processing_time=processing_time, **data)

        response = send_file(output_path, as_attachment=True, download_name=f'QR_{filename}')
        response.headers['X-Processing-Time'] = f"{processing_time:.2f}"
        return response

    except Exception as e:
        logger.error(f"Error in /upload: {e}", exc_info=True)
        log_file_processing(session_id, filename, 'error', error_message=str(e))
        return jsonify({'error': 'Une erreur interne est survenue.'}), 500
    finally:
        for path in [input_path, output_path, qr_path]:
            if os.path.exists(path): os.remove(path)


@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    session_id = get_session_id()
    memory_zip = BytesIO()
    with zipfile.ZipFile(memory_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        files = request.files.getlist('pdf_files')
        if not files: return jsonify({'error': 'Aucun fichier reçu'}), 400

        for file in files:
            if file.filename == '' or not file.filename.lower().endswith('.pdf'): continue

            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            input_path = os.path.join(TEMP_DIR, f"batch_{unique_id}_{filename}")
            output_path = os.path.join(TEMP_DIR, f"batch_qr_{unique_id}_{filename}")
            qr_path = os.path.join(TEMP_DIR, f"batch_qr_img_{unique_id}.png")

            try:
                file.save(input_path)
                if os.path.getsize(input_path) > MAX_FILE_SIZE: continue

                data = extract_data_from_pdf(input_path)
                generate_qr_code(data, qr_path)
                add_qr_to_pdf(input_path, qr_path, output_path)
                zf.write(output_path, f'QR_{filename}')
            except Exception as e:
                logger.error(f"Error in batch for file {filename}: {e}")
            finally:
                for path in [input_path, output_path, qr_path]:
                    if os.path.exists(path): os.remove(path)

    memory_zip.seek(0)
    return send_file(memory_zip, download_name='fichiers_QR.zip', as_attachment=True)


@app.route('/dashboard')
def dashboard():
    session_id = get_session_id()
    page = request.args.get('page', 1, type=int)
    per_page = 10
    filter_date = request.args.get('date', '')
    filter_status = request.args.get('status', '')

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        base_query = "FROM files WHERE session_id = ?"
        params = [session_id]

        if filter_date:
            base_query += " AND date(upload_time) = ?"
            params.append(filter_date)
        if filter_status:
            base_query += " AND status = ?"
            params.append(filter_status)

        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) {base_query}", tuple(params))
        total_items = cursor.fetchone()[0]
        total_pages = (total_items + per_page - 1) // per_page

        offset = (page - 1) * per_page
        query = f"""
            SELECT original_filename, strftime('%Y-%m-%d', upload_time), strftime('%H:%M:%S', upload_time), processing_time, status
            {base_query} ORDER BY upload_time DESC LIMIT ? OFFSET ?
        """
        params.extend([per_page, offset])
        cursor.execute(query, tuple(params))

        files = [
            {'filename': row[0], 'date': row[1], 'time': row[2],
             'processing_time': f"{row[3]:.2f}s" if row[3] is not None else "N/A", 'status': row[4]}
            for row in cursor.fetchall()
        ]
        conn.close()

        return jsonify({'files': files, 'total_pages': total_pages, 'current_page': page})
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}", exc_info=True)
        return jsonify({'error': 'Erreur de récupération des données'}), 500


@app.route('/stats')
def stats():
    session_id = get_session_id()
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM files WHERE session_id = ? AND status = 'completed'", (session_id,))
        total_completed = cursor.fetchone()[0]
        cursor.execute(
            "SELECT COUNT(*) FROM files WHERE session_id = ? AND status = 'completed' AND date(upload_time) = date('now')",
            (session_id,))
        today_completed = cursor.fetchone()[0]
        conn.close()
        return jsonify({'total_completed': total_completed, 'today_completed': today_completed})
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Erreur de récupération des statistiques'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
