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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-for-somaport'  # Change this in production
CORS(app)

# --- CONFIGURATION POUR PYTHONANYWHERE ---
# Utiliser le dossier /tmp pour les fichiers temporaires car il est accessible en écriture
UPLOAD_FOLDER = '/tmp'
PROCESSED_FOLDER = '/tmp'
QR_FOLDER = '/tmp'
# Utiliser le chemin absolu pour la base de données
DATABASE_FILE = '/home/pythonweb12/somaport-qr-app/somaport_qr.db'

ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MAX_BATCH_SIZE = 10


# La création des dossiers n'est plus nécessaire car /tmp existe déjà
# for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, QR_FOLDER]:
#     os.makedirs(folder, exist_ok=True)

def init_database():
    """Initialize SQLite database for file tracking"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT, original_filename TEXT,
            processed_filename TEXT, upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_time REAL, status TEXT DEFAULT 'pending', bon_delivrer TEXT,
            conteneur TEXT, permis_douane TEXT, date_sortie TEXT, file_size INTEGER,
            error_message TEXT
        )
    ''')
    conn.commit()
    conn.close()


init_database()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id():
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def log_file_processing(session_id, filename, status, **kwargs):
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()
    if status == 'started':
        cursor.execute('''
            INSERT INTO files (session_id, original_filename, status, file_size)
            VALUES (?, ?, ?, ?)
        ''', (session_id, filename, status, kwargs.get('file_size', 0)))
    else:
        cursor.execute('''
            UPDATE files SET status = ?, processing_time = ?, processed_filename = ?,
            bon_delivrer = ?, conteneur = ?, permis_douane = ?, date_sortie = ?, error_message = ?
            WHERE id = (SELECT max(id) FROM files WHERE session_id = ? AND original_filename = ?)
        ''', (status, kwargs.get('processing_time'), kwargs.get('processed_filename'),
              kwargs.get('bon'), kwargs.get('conteneur'), kwargs.get('permis'),
              kwargs.get('date_sortie'), kwargs.get('error_message'), session_id, filename))
    conn.commit()
    conn.close()


def extract_data_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            text = text.replace('\xa0', ' ').replace('\n', ' ').strip()
            logger.info(f"Extracted text length: {len(text)} characters")
            bon_patterns = [r'N[°º]?\s*DU\s*BON\s*A\s*DELIVRER\s*[:\-]?\s*([A-Z0-9\-\/]+)',
                            r'BON\s*A\s*DELIVRER\s*[:\-]?\s*([A-Z0-9\-\/]+)']
            conteneur_patterns = [r'CONTENEUR\s*[:\-]?\s*([A-Z]{4}[0-9]{7})', r'\b([A-Z]{4}[0-9]{7})\b']
            permis_patterns = [r'NUM[ÉE]RO\s*DE\s*PERMIS\s*DE\s*DOUANE\s*[:\-]?\s*([0-9\/\-]+)',
                               r'PERMIS\s*DE\s*DOUANE\s*[:\-]?\s*([0-9\/\-]+)']
            date_patterns = [r'DATE\s*LIMITE\s*DE\s*SORTIE\s*LE\s*[:\-]?\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})',
                             r'Date\s*limite\s*de\s*sortie\s*le\s*[:\-]?\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})']

            def extract_with_patterns(patterns):
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match: return match.group(1).strip()
                return "N/A"

            return extract_with_patterns(bon_patterns), extract_with_patterns(
                conteneur_patterns), extract_with_patterns(permis_patterns), extract_with_patterns(date_patterns)
    except Exception as e:
        logger.error(f"Error extracting data from PDF: {e}")
        return "N/A", "N/A", "N/A", "N/A"


def generate_qr_code(bon, conteneur, permis, date_sortie, output_path):
    qr_text = f"Bon à délivrer: {bon}\nConteneur: {conteneur}\nPermis de Douane: {permis}\nDate limite sortie: {date_sortie}"
    qr = qrcode.QRCode(version=1, error_correction=qrcode.constants.ERROR_CORRECT_H, box_size=10, border=4)
    qr.add_data(qr_text)
    qr.make(fit=True)
    qr_image = qr.make_image(fill_color="black", back_color="white")
    qr_image.save(output_path)
    logger.info(f"QR code generated successfully: {output_path}")


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
    logger.info(f"QR code added to PDF successfully: {output_pdf_path}")


@app.route('/')
def index():
    return render_template('index.html')


# --- FONCTION UPLOAD CORRIGÉE ET ROBUSTE ---
@app.route('/upload', methods=['POST'])
def upload():
    session_id = get_session_id()
    if 'pdf_file' not in request.files:
        return jsonify({'error': 'Aucun fichier reçu'}), 400
    file = request.files['pdf_file']
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Type de fichier non autorisé.'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
    output_path = ""
    qr_path = ""

    try:
        start_time = time.time()
        file.save(input_path)
        file_size = os.path.getsize(input_path)
        if file_size > MAX_FILE_SIZE:
            return jsonify({'error': 'Fichier trop volumineux.'}), 400

        logger.info(f"File uploaded: {filename}")
        log_file_processing(session_id, filename, 'started', file_size=file_size)

        bon, conteneur, permis, date_sortie = extract_data_from_pdf(input_path)

        qr_filename = f'qr_{session_id}_{filename}.png'
        qr_path = os.path.join(QR_FOLDER, qr_filename)
        generate_qr_code(bon, conteneur, permis, date_sortie, qr_path)

        output_filename = f'qr_{filename}'
        output_path = os.path.join(PROCESSED_FOLDER, f"{session_id}_{output_filename}")
        add_qr_to_pdf(input_path, qr_path, output_path)

        processing_time = time.time() - start_time
        log_file_processing(
            session_id, filename, 'completed', processing_time=processing_time,
            processed_filename=output_filename, bon=bon, conteneur=conteneur,
            permis=permis, date_sortie=date_sortie
        )

        memory_file = BytesIO()
        with open(output_path, 'rb') as f:
            memory_file.write(f.read())
        memory_file.seek(0)

        logger.info(f"Processing successful, sending file: {filename}")
        return send_file(
            memory_file, as_attachment=True,
            download_name=f'QR_{filename}', mimetype='application/pdf'
        )
    except Exception as e:
        logger.error(f"An error occurred during processing: {e}", exc_info=True)
        return jsonify({'error': 'Une erreur interne est survenue lors du traitement.'}), 500
    finally:
        # Nettoyage garanti des fichiers temporaires
        if os.path.exists(input_path): os.remove(input_path)
        if os.path.exists(output_path): os.remove(output_path)
        if os.path.exists(qr_path): os.remove(qr_path)


@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    session_id = get_session_id()
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        files = request.files.getlist('pdf_files')
        if not files or len(files) == 0: return jsonify({'error': 'Aucun fichier reçu'}), 400
        if len(files) > MAX_BATCH_SIZE: return jsonify({'error': f'Trop de fichiers. Maximum: {MAX_BATCH_SIZE}'}), 400

        for file in files:
            if file.filename == '' or not allowed_file(file.filename): continue
            input_path, output_path, qr_path = "", "", ""
            try:
                filename = secure_filename(file.filename)
                input_path = os.path.join(UPLOAD_FOLDER, f"batch_{session_id}_{filename}")
                file.save(input_path)
                if os.path.getsize(input_path) > MAX_FILE_SIZE: continue

                bon, conteneur, permis, date_sortie = extract_data_from_pdf(input_path)
                qr_filename = f'qr_batch_{session_id}_{filename}.png'
                qr_path = os.path.join(QR_FOLDER, qr_filename)
                generate_qr_code(bon, conteneur, permis, date_sortie, qr_path)

                output_filename = f'qr_{filename}'
                output_path = os.path.join(PROCESSED_FOLDER, f"batch_{session_id}_{output_filename}")
                add_qr_to_pdf(input_path, qr_path, output_path)

                zf.write(output_path, output_filename)
            except Exception as e:
                logger.error(f"Error processing file in batch: {e}")
            finally:
                if os.path.exists(input_path): os.remove(input_path)
                if os.path.exists(output_path): os.remove(output_path)
                if os.path.exists(qr_path): os.remove(qr_path)
    memory_file.seek(0)
    return send_file(memory_file, download_name='processed_files.zip', as_attachment=True)


@app.route('/history')
def history():
    session_id = get_session_id()
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT original_filename, status FROM files 
            WHERE session_id = ? ORDER BY upload_time DESC LIMIT 50
        ''', (session_id,))
        files = [{'filename': row[0], 'status': row[1]} for row in cursor.fetchall()]
        conn.close()
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': 'Erreur lors de la récupération de l\'historique'}), 500


@app.route('/stats')
def stats():
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM files WHERE status = "completed"')
        total_completed = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM files WHERE status = 'completed' AND date(upload_time) = date('now')")
        today_completed = cursor.fetchone()[0]
        cursor.execute('SELECT AVG(processing_time) FROM files WHERE status = "completed"')
        avg_processing_time = cursor.fetchone()[0] or 0
        conn.close()
        return jsonify({
            'total_completed': total_completed,
            'today_completed': today_completed,
            'avg_processing_time': round(avg_processing_time, 2)
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Erreur lors de la récupération des statistiques'}), 500


# Les autres routes (health, progress, errorhandlers) peuvent rester les mêmes
@app.errorhandler(404)
def not_found(e):
    return render_template('index.html'), 404  # Redirige vers l'accueil en cas de 404
