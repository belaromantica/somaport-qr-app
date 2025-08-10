from flask import Flask, request, render_template, send_file
import os
from werkzeug.utils import secure_filename
from qr_facture import process_pdf  # importe ta fonction principale

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'pdf_file' not in request.files:
            return "❌ Aucun fichier reçu."
        file = request.files['pdf_file']
        if file.filename == '':
            return "❌ Aucun fichier sélectionné."

        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_path = os.path.join(UPLOAD_FOLDER, f"qr_{filename}")

        file.save(input_path)
        process_pdf(input_path, output_path)

        return send_file(output_path, as_attachment=True)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
