import os
import re
import qrcode
import sqlite3
import smtplib
import uuid
from io import BytesIO
from PIL import Image
from flask import Flask, render_template, request, send_from_directory, jsonify, flash, session, send_file
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from werkzeug.utils import secure_filename
import logging
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import threading
import time
from flask_cors import CORS
import zipfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
CORS(app)  # Enable CORS for all routes

# Configuration
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
QR_FOLDER = 'qr'
DATABASE_FILE = 'somaport_qr.db'
ALLOWED_EXTENSIONS = {'pdf'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
MAX_BATCH_SIZE = 10  # Maximum files in batch processing

# Create directories
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, QR_FOLDER]:
    os.makedirs(folder, exist_ok=True)


# Initialize database
def init_database():
    """Initialize SQLite database for file tracking"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    # Files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            original_filename TEXT,
            processed_filename TEXT,
            upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processing_time REAL,
            status TEXT DEFAULT 'pending',
            bon_delivrer TEXT,
            conteneur TEXT,
            permis_douane TEXT,
            date_sortie TEXT,
            file_size INTEGER,
            error_message TEXT
        )
    ''')

    # Processing queue table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processing_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_id INTEGER,
            status TEXT DEFAULT 'queued',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (file_id) REFERENCES files (id)
        )
    ''')

    conn.commit()
    conn.close()


init_database()


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_session_id():
    """Get or create session ID"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']


def log_file_processing(session_id, filename, status, **kwargs):
    """Log file processing to database"""
    conn = sqlite3.connect(DATABASE_FILE)
    cursor = conn.cursor()

    if status == 'started':
        cursor.execute('''
            INSERT INTO files (session_id, original_filename, status, file_size)
            VALUES (?, ?, ?, ?)
        ''', (session_id, filename, status, kwargs.get('file_size', 0)))
        file_id = cursor.lastrowid
    else:
        cursor.execute('''
            UPDATE files SET status = ?, processing_time = ?, processed_filename = ?,
            bon_delivrer = ?, conteneur = ?, permis_douane = ?, date_sortie = ?, error_message = ?
            WHERE session_id = ? AND original_filename = ?
        ''', (status, kwargs.get('processing_time'), kwargs.get('processed_filename'),
              kwargs.get('bon'), kwargs.get('conteneur'), kwargs.get('permis'),
              kwargs.get('date_sortie'), kwargs.get('error_message'), session_id, filename))
        file_id = None

    conn.commit()
    conn.close()
    return file_id


def extract_data_from_pdf(pdf_path):
    """Extract relevant data from PDF document with enhanced patterns"""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

            # Clean text
            text = text.replace('\xa0', ' ').replace('\n', ' ').strip()
            logger.info(f"Extracted text length: {len(text)} characters")

            # Enhanced patterns for better extraction
            bon_patterns = [
                r'N[°º]?\s*DU\s*BON\s*A\s*DELIVRER\s*[:\-]?\s*([A-Z0-9\-\/]+)',
                r'BON\s*A\s*DELIVRER\s*[:\-]?\s*([A-Z0-9\-\/]+)',
                r'BON\s*N[°º]?\s*[:\-]?\s*([A-Z0-9\-\/]+)',
                r'DELIVERY\s*ORDER\s*[:\-]?\s*([A-Z0-9\-\/]+)'
            ]

            conteneur_patterns = [
                r'CONTENEUR\s*[:\-]?\s*([A-Z]{4}[0-9]{7})',
                r'CONTAINER\s*[:\-]?\s*([A-Z]{4}[0-9]{7})',
                r'CONT[:\-]?\s*([A-Z]{4}[0-9]{7})',
                r'\b([A-Z]{4}[0-9]{7})\b'
            ]

            permis_patterns = [
                r'NUM[ÉE]RO\s*DE\s*PERMIS\s*DE\s*DOUANE\s*[:\-]?\s*([0-9\/\-]+)',
                r'PERMIS\s*DE\s*DOUANE\s*[:\-]?\s*([0-9\/\-]+)',
                r'PERMIS\s*N[°º]?\s*[:\-]?\s*([0-9\/\-]+)',
                r'CUSTOMS\s*PERMIT\s*[:\-]?\s*([0-9\/\-]+)'
            ]

            date_patterns = [
                r'DATE\s*LIMITE\s*DE\s*SORTIE\s*LE\s*[:\-]?\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})',
                r'Date\s*limite\s*de\s*sortie\s*le\s*[:\-]?\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})',
                r'SORTIE\s*LE\s*[:\-]?\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})',
                r'EXIT\s*DATE\s*[:\-]?\s*([0-9]{2}[-/][0-9]{2}[-/][0-9]{4})'
            ]

            def extract_with_patterns(patterns, field_name):
                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        result = match.group(1).strip()
                        logger.info(f"Found {field_name}: {result}")
                        return result
                logger.warning(f"No match found for {field_name}")
                return "N/A"

            bon = extract_with_patterns(bon_patterns, "bon")
            conteneur = extract_with_patterns(conteneur_patterns, "conteneur")
            permis = extract_with_patterns(permis_patterns, "permis")
            date_sortie = extract_with_patterns(date_patterns, "date_sortie")

            return bon, conteneur, permis, date_sortie

    except Exception as e:
        logger.error(f"Error extracting data from PDF: {e}")
        return "N/A", "N/A", "N/A", "N/A"


def generate_qr_code(bon, conteneur, permis, date_sortie, output_path):
    """Generate QR code with extracted data, no extra info"""
    try:
        # Only include the extracted data in the QR code text
        qr_text = f"""Bon à délivrer: {bon}
Conteneur: {conteneur}
Permis de Douane: {permis}
Date limite sortie: {date_sortie}"""

        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
            box_size=10,
            border=4,
        )
        qr.add_data(qr_text)
        qr.make(fit=True)

        qr_image = qr.make_image(fill_color="black", back_color="white")
        qr_image.save(output_path)

        logger.info(f"QR code generated successfully: {output_path}")
        return qr_text

    except Exception as e:
        logger.error(f"Error generating QR code: {e}")
        raise


def add_qr_to_pdf(pdf_path, qr_path, output_pdf_path):
    """Add QR code to the last page of PDF without extra modifications"""
    try:
        reader = PdfReader(pdf_path)
        writer = PdfWriter()

        # Load QR image
        qr_img = Image.open(qr_path)
        qr_width, qr_height = qr_img.size

        # Get last page dimensions
        last_page = reader.pages[-1]
        page_width = float(last_page.mediabox.width)
        page_height = float(last_page.mediabox.height)

        # Calculate QR code size and position (adjust as needed)
        # Based on the user's image, the QR code is in the top right corner.
        # I'll use a fixed size and position it relative to the top-right.
        qr_display_size = 100  # pixels, adjust as needed for desired size on PDF
        x_pos = page_width - qr_display_size - 18  # 30 points from right edge
        y_pos = page_height - qr_display_size - 25  # 30 points from top edge

        # Create overlay PDF with QR code
        packet = BytesIO()
        can = canvas.Canvas(packet, pagesize=(page_width, page_height))

        # Add QR code directly, no border or timestamp
        can.drawImage(qr_path, x_pos, y_pos, width=qr_display_size, height=qr_display_size, mask='auto')

        can.save()
        packet.seek(0)

        qr_pdf = PdfReader(packet)

        # Add all pages except the last one
        for i in range(len(reader.pages) - 1):
            writer.add_page(reader.pages[i])

        # Merge QR code with last page
        last_page.merge_page(qr_pdf.pages[0])
        writer.add_page(last_page)

        # Write output PDF
        with open(output_pdf_path, 'wb') as f_out:
            writer.write(f_out)

        logger.info(f"QR code added to PDF successfully: {output_pdf_path}")

    except Exception as e:
        logger.error(f"Error adding QR code to PDF: {e}")
        raise


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload and processing with enhanced features"""
    start_time = time.time()
    session_id = get_session_id()

    try:
        # Check if file was uploaded
        if 'pdf_file' not in request.files:
            logger.warning("No file uploaded")
            return jsonify({'error': 'Aucun fichier reçu'}), 400

        file = request.files['pdf_file']

        if file.filename == '':
            logger.warning("No file selected")
            return jsonify({'error': 'Aucun fichier sélectionné'}), 400

        if not allowed_file(file.filename):
            logger.warning(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Type de fichier non autorisé. Seuls les fichiers PDF sont acceptés.'}), 400

        # Secure filename
        filename = secure_filename(file.filename)
        if not filename:
            filename = 'document.pdf'

        # Save uploaded file
        input_path = os.path.join(UPLOAD_FOLDER, f"{session_id}_{filename}")
        file.save(input_path)

        # Check file size
        file_size = os.path.getsize(input_path)
        if file_size > MAX_FILE_SIZE:
            os.remove(input_path)
            return jsonify({'error': 'Fichier trop volumineux. Taille maximale: 16MB'}), 400

        logger.info(f"File uploaded successfully: {filename} ({file_size} bytes)")

        # Log processing start
        log_file_processing(session_id, filename, 'started', file_size=file_size)

        # Extract data from PDF
        bon, conteneur, permis, date_sortie = extract_data_from_pdf(input_path)

        # Generate QR Code
        qr_filename = f'qr_{session_id}_{filename}.png'
        qr_path = os.path.join(QR_FOLDER, qr_filename)
        qr_text = generate_qr_code(bon, conteneur, permis, date_sortie, qr_path)

        # Add QR to PDF
        output_filename = f'qr_{filename}'
        output_path = os.path.join(PROCESSED_FOLDER, f"{session_id}_{output_filename}")
        add_qr_to_pdf(input_path, qr_path, output_path)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Log successful processing
        log_file_processing(
            session_id, filename, 'completed',
            processing_time=processing_time,
            processed_filename=output_filename,
            bon=bon, conteneur=conteneur, permis=permis, date_sortie=date_sortie
        )

        # Clean up input file
        os.remove(input_path)

        logger.info(f"Processing completed successfully for: {filename} in {processing_time:.2f}s")

        # Return processed file
        return send_from_directory(
            PROCESSED_FOLDER,
            f"{session_id}_{output_filename}",
            as_attachment=True,
            download_name=f'QR_{filename}'
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing file: {e}")

        # Log error
        log_file_processing(
            session_id, filename if 'filename' in locals() else 'unknown',
            'error', processing_time=processing_time, error_message=str(e)
        )

        return jsonify({'error': 'Erreur lors du traitement du fichier. Veuillez réessayer.'}), 500


@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """Handle batch file upload and processing"""
    session_id = get_session_id()
    memory_file = BytesIO()

    try:
        files = request.files.getlist('pdf_files')

        if not files or len(files) == 0:
            return jsonify({'error': 'Aucun fichier reçu'}), 400

        if len(files) > MAX_BATCH_SIZE:
            return jsonify({'error': f'Trop de fichiers. Maximum: {MAX_BATCH_SIZE}'}), 400

        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file in files:
                if file.filename == '' or not allowed_file(file.filename):
                    continue

                try:
                    filename = secure_filename(file.filename)
                    if not filename:
                        filename = f'document_{uuid.uuid4()}.pdf'

                    input_path = os.path.join(UPLOAD_FOLDER, f"batch_{session_id}_{filename}")
                    file.save(input_path)

                    if os.path.getsize(input_path) > MAX_FILE_SIZE:
                        os.remove(input_path)
                        continue

                    bon, conteneur, permis, date_sortie = extract_data_from_pdf(input_path)
                    qr_filename = f'qr_batch_{session_id}_{filename}.png'
                    qr_path = os.path.join(QR_FOLDER, qr_filename)
                    generate_qr_code(bon, conteneur, permis, date_sortie, qr_path)

                    output_filename = f'qr_{filename}'
                    output_path = os.path.join(PROCESSED_FOLDER, f"batch_{session_id}_{output_filename}")
                    add_qr_to_pdf(input_path, qr_path, output_path)

                    zf.write(output_path, output_filename)

                    os.remove(input_path)
                    os.remove(output_path)
                    os.remove(qr_path)

                except Exception as e:
                    logger.error(f"Error processing file in batch: {e}")
                    continue

        memory_file.seek(0)
        return send_file(memory_file, download_name='processed_files.zip', as_attachment=True)

    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return jsonify({'error': 'Erreur lors du traitement par lot'}), 500


@app.route('/history')
def history():
    """Get file processing history for current session"""
    session_id = get_session_id()

    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT original_filename, status, upload_time, processing_time,
                   bon_delivrer, conteneur, permis_douane, date_sortie, file_size
            FROM files 
            WHERE session_id = ? 
            ORDER BY upload_time DESC
            LIMIT 50
        ''', (session_id,))

        files = []
        for row in cursor.fetchall():
            files.append({
                'filename': row[0],
                'status': row[1],
                'upload_time': row[2],
                'processing_time': row[3],
                'bon_delivrer': row[4],
                'conteneur': row[5],
                'permis_douane': row[6],
                'date_sortie': row[7],
                'file_size': row[8]
            })

        conn.close()
        return jsonify({'files': files})

    except Exception as e:
        logger.error(f"Error getting history: {e}")
        return jsonify({'error': 'Erreur lors de la récupération de l\'historique'}), 500


@app.route('/stats')
def stats():
    """Get processing statistics"""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()

        # Total files processed
        cursor.execute('SELECT COUNT(*) FROM files WHERE status = "completed"')
        total_completed = cursor.fetchone()[0]

        # Total files with errors
        cursor.execute('SELECT COUNT(*) FROM files WHERE status = "error"')
        total_errors = cursor.fetchone()[0]

        # Average processing time
        cursor.execute('SELECT AVG(processing_time) FROM files WHERE status = "completed"')
        avg_processing_time = cursor.fetchone()[0] or 0

        # Files processed today
        cursor.execute('''
            SELECT COUNT(*) FROM files 
            WHERE status = "completed" AND date(upload_time) = date('now')
        ''')
        today_completed = cursor.fetchone()[0]

        conn.close()

        return jsonify({
            'total_completed': total_completed,
            'total_errors': total_errors,
            'avg_processing_time': round(avg_processing_time, 2),
            'today_completed': today_completed
        })

    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Erreur lors de la récupération des statistiques'}), 500


@app.route('/health')
def health():
    """Enhanced health check endpoint"""
    try:
        # Check database connection
        conn = sqlite3.connect(DATABASE_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT 1')
        conn.close()

        # Check directories
        dirs_ok = all(os.path.exists(d) for d in [UPLOAD_FOLDER, PROCESSED_FOLDER, QR_FOLDER])

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'directories': 'ok' if dirs_ok else 'error',
            'version': '2.0.0'
        })

    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/progress/<task_id>')
def progress(task_id):
    """Get processing progress for a task"""
    # This would be implemented with a task queue system like Celery in production
    return jsonify({'progress': 100, 'status': 'completed'})


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'Fichier trop volumineux'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Page non trouvée'}), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Erreur interne du serveur'}), 500


#if __name__ == '__main__':
 #   app.run(debug=True, host='0.0.0.0', port=5000)
