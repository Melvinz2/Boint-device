import sys
import os
sys.path.append('yolov8n')  # Add yolov8n directory to path

from flask import Flask, request, jsonify
import threading
import time
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from detect import YOLODetector  # Import our custom detector

app = Flask(__name__)

# Konfigurasi
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Dictionary untuk menyimpan hasil deteksi
detection_jobs = {}

# Buat folder uploads jika belum ada
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"[{datetime.now()}] Folder '{UPLOAD_FOLDER}' berhasil dibuat")

# Initialize YOLOv8 Detector
print(f"[{datetime.now()}] Menginisialisasi YOLOv8 Detector...")
detector = YOLODetector("yolov8n/hasil_traning_botol_kaleng/botol_kaleng_model2/weights/best.pt")

def allowed_file(filename):
    """Cek apakah file yang diupload adalah gambar yang valid"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_objects_background(image_path, job_id, filename, conf_threshold=0.5):
    """
    Fungsi untuk menjalankan deteksi objek di background thread
    """
    try:
        print(f"[{datetime.now()}] Memulai proses deteksi untuk {filename} (Job ID: {job_id[:8]}...)")
        
        # Update status ke processing
        detection_jobs[job_id]['status'] = 'processing'
        
        # Simulasi sedikit delay untuk demo
        time.sleep(0.5)
        
        # Jalankan deteksi menggunakan YOLODetector class
        detection_result = detector.detect_image(
            image_path=image_path,
            conf_threshold=conf_threshold,
            imgsz=640
        )
        
        unique_labels = detection_result['unique_labels']
        total_objects = detection_result['total_objects']
        confidences = detection_result['confidences']
        
        print(f"[{datetime.now()}] Deteksi selesai untuk {filename}.")
        print(f"[{datetime.now()}] Objek terdeteksi: {unique_labels} (Total: {total_objects})")
        
        # Update hasil deteksi
        detection_jobs[job_id] = {
            'status': 'completed',
            'results': unique_labels,
            'timestamp': datetime.now(),
            'filename': filename,
            'total_objects': total_objects,
            'unique_objects': len(unique_labels),
            'all_labels': detection_result['labels'],
            'confidences': confidences,
            'avg_confidence': sum(confidences) / len(confidences) if confidences else 0,
            'detection_details': detection_result
        }
        
    except Exception as e:
        print(f"[{datetime.now()}] Error dalam proses deteksi {filename}: {e}")
        detection_jobs[job_id] = {
            'status': 'error',
            'results': [],
            'error': str(e),
            'timestamp': datetime.now(),
            'filename': filename,
            'total_objects': 0,
            'unique_objects': 0
        }

@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Endpoint untuk menerima upload gambar dari Raspberry Pi
    """
    print(f"[{datetime.now()}] Menerima request upload dari {request.remote_addr}")
    
    try:
        # Cek apakah detector siap
        if detector.model is None:
            return jsonify({'error': 'Model YOLOv8 tidak tersedia'}), 503
        
        # Cek apakah ada file dalam request
        if 'image' not in request.files:
            print(f"[{datetime.now()}] Error: Tidak ada file 'image' dalam request")
            return jsonify({'error': 'Tidak ada file image'}), 400
        
        file = request.files['image']
        
        # Cek apakah file dipilih
        if file.filename == '':
            print(f"[{datetime.now()}] Error: Tidak ada file yang dipilih")
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
        # Get confidence threshold dari request (optional)
        conf_threshold = float(request.form.get('confidence', 0.5))
        if not (0.1 <= conf_threshold <= 1.0):
            conf_threshold = 0.5
        
        # Cek apakah file valid
        if file and allowed_file(file.filename):
            # Generate job ID unik
            job_id = str(uuid.uuid4())
            
            # Generate nama file yang aman
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            original_filename = secure_filename(file.filename)
            filename = f"{timestamp}_{original_filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            
            # Simpan file
            file.save(filepath)
            print(f"[{datetime.now()}] File berhasil disimpan: {filepath}")
            
            # Inisialisasi job status
            detection_jobs[job_id] = {
                'status': 'queued',
                'results': [],
                'timestamp': datetime.now(),
                'filename': filename,
                'confidence_threshold': conf_threshold
            }
            
            # Jalankan deteksi di background thread
            detection_thread = threading.Thread(
                target=detect_objects_background, 
                args=(filepath, job_id, filename, conf_threshold)
            )
            detection_thread.daemon = True
            detection_thread.start()
            
            print(f"[{datetime.now()}] Background thread deteksi dimulai untuk {filename} (Job ID: {job_id[:8]}...)")
            
            # Response dengan job_id
            return jsonify({
                'status': 'success',
                'message': 'File berhasil diupload, proses deteksi dimulai',
                'job_id': job_id,
                'filename': filename,
                'confidence_threshold': conf_threshold
            }), 200
            
        else:
            print(f"[{datetime.now()}] Error: Tipe file tidak diizinkan: {file.filename}")
            return jsonify({'error': 'Tipe file tidak diizinkan'}), 400
            
    except Exception as e:
        print(f"[{datetime.now()}] Error dalam upload_image: {e}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/api/result/<job_id>', methods=['GET'])
def get_result(job_id):
    """
    Endpoint untuk mengecek hasil deteksi berdasarkan job_id
    """
    print(f"[{datetime.now()}] Request hasil untuk Job ID: {job_id[:8]}...")
    
    if job_id not in detection_jobs:
        print(f"[{datetime.now()}] Job ID tidak ditemukan: {job_id[:8]}...")
        return jsonify({'error': 'Job ID tidak ditemukan'}), 404
    
    job_data = detection_jobs[job_id]
    
    # Prepare response
    response_data = {
        'job_id': job_id,
        'status': job_data['status'],
        'hasil': job_data['results'],
        'filename': job_data['filename'],
        'timestamp': job_data['timestamp'].isoformat(),
        'total_objects': job_data.get('total_objects', 0),
        'unique_objects': job_data.get('unique_objects', 0),
        'confidence_threshold': job_data.get('confidence_threshold', 0.5)
    }
    
    # Add additional info if completed
    if job_data['status'] == 'completed':
        response_data.update({
            'all_detected_labels': job_data.get('all_labels', []),
            'confidences': job_data.get('confidences', []),
            'average_confidence': round(job_data.get('avg_confidence', 0), 3)
        })
        print(f"[{datetime.now()}] Mengirim hasil untuk Job ID {job_id[:8]}...: {job_data['results']}")
    elif job_data['status'] == 'error':
        response_data['error'] = job_data.get('error', 'Unknown error')
    
    return jsonify(response_data), 200

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """
    Endpoint untuk melihat semua job yang ada
    """
    jobs_summary = {}
    active_jobs = 0
    completed_jobs = 0
    error_jobs = 0
    
    for job_id, job_data in detection_jobs.items():
        jobs_summary[job_id] = {
            'status': job_data['status'],
            'filename': job_data['filename'],
            'timestamp': job_data['timestamp'].isoformat(),
            'total_objects': job_data.get('total_objects', 0)
        }
        
        # Count status
        if job_data['status'] in ['queued', 'processing']:
            active_jobs += 1
        elif job_data['status'] == 'completed':
            completed_jobs += 1
        elif job_data['status'] == 'error':
            error_jobs += 1
    
    return jsonify({
        'total_jobs': len(detection_jobs),
        'active_jobs': active_jobs,
        'completed_jobs': completed_jobs,
        'error_jobs': error_jobs,
        'jobs': jobs_summary
    }), 200

@app.route('/api/model/info', methods=['GET'])
def get_model_info():
    """
    Endpoint untuk mendapatkan informasi model
    """
    if detector.model is None:
        return jsonify({'error': 'Model tidak tersedia'}), 503
    
    model_info = detector.get_model_info()
    return jsonify({
        'model_loaded': True,
        'model_info': model_info,
        'supported_classes': list(model_info['model_names'].values()) if model_info else []
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint untuk health check
    """
    active_jobs = len([j for j in detection_jobs.values() if j['status'] in ['queued', 'processing']])
    
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': detector.model is not None,
        'model_path': detector.model_path if detector.model else None,
        'active_jobs': active_jobs,
        'total_jobs': len(detection_jobs)
    }), 200

@app.route('/', methods=['GET'])
def home():
    """
    Endpoint root untuk info server
    """
    model_info = detector.get_model_info() if detector.model else None
    
    return jsonify({
        'message': 'Flask YOLOv8 Detection Server',
        'version': '2.0',
        'endpoints': {
            '/api/upload': 'POST - Upload gambar untuk deteksi',
            '/api/result/<job_id>': 'GET - Cek hasil deteksi',
            '/api/jobs': 'GET - List semua job',
            '/api/model/info': 'GET - Info model',
            '/api/files': 'GET - List uploaded files',
            '/api/files/cleanup': 'POST - Cleanup old files',
            '/api/files/storage': 'GET - Storage information',
            '/health': 'GET - Health check',
        },
        'model_status': 'loaded' if detector.model else 'not loaded',
        'supported_classes': list(model_info['model_names'].values()) if model_info else [],
        'model_path': detector.model_path
    })

@app.route('/api/files', methods=['GET'])
def list_uploaded_files():
    """
    Endpoint untuk melihat daftar file yang sudah diupload
    """
    try:
        files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath) and allowed_file(filename):
                    file_stat = os.stat(filepath)
                    files.append({
                        'filename': filename,
                        'size': file_stat.st_size,
                        'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
                    })
        
        # Sort by creation time (newest first)
        files.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'total_files': len(files),
            'files': files,
            'upload_folder': UPLOAD_FOLDER
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error listing files: {str(e)}'}), 500

@app.route('/api/files/cleanup', methods=['POST'])
def cleanup_old_files():
    """
    Endpoint untuk cleanup file lama (opsional)
    """
    try:
        days_old = request.json.get('days', 7) if request.is_json else 7
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        
        deleted_files = []
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    if os.path.getctime(filepath) < cutoff_time:
                        os.remove(filepath)
                        deleted_files.append(filename)
                        print(f"[{datetime.now()}] Deleted old file: {filename}")
        
        return jsonify({
            'status': 'success',
            'deleted_files': deleted_files,
            'deleted_count': len(deleted_files),
            'days_old': days_old
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error cleanup files: {str(e)}'}), 500

@app.route('/api/files/storage', methods=['GET'])
def get_storage_info():
    """
    Endpoint untuk informasi storage
    """
    try:
        total_files = 0
        total_size = 0
        
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                if os.path.isfile(filepath):
                    total_files += 1
                    total_size += os.path.getsize(filepath)
        
        # Get disk usage
        statvfs = os.statvfs(UPLOAD_FOLDER)
        disk_free = statvfs.f_frsize * statvfs.f_bavail
        disk_total = statvfs.f_frsize * statvfs.f_blocks
        disk_used = disk_total - disk_free
        
        return jsonify({
            'upload_folder': UPLOAD_FOLDER,
            'total_files': total_files,
            'total_size_bytes': total_size,
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'disk_usage': {
                'total_gb': round(disk_total / (1024**3), 2),
                'used_gb': round(disk_used / (1024**3), 2),
                'free_gb': round(disk_free / (1024**3), 2),
                'usage_percent': round((disk_used / disk_total) * 100, 1)
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error getting storage info: {str(e)}'}), 500

@app.route('/api/webcam/start', methods=['POST'])
def start_webcam_detection():
    """
    Endpoint untuk memulai deteksi webcam (untuk testing)
    Note: Ini hanya untuk testing di server, biasanya tidak digunakan dalam production
    """
    try:
        source = request.json.get('source', 0) if request.is_json else 0
        
        # Jalankan dalam thread terpisah
        webcam_thread = threading.Thread(
            target=detector.webcam_detection,
            args=(source,)
        )
        webcam_thread.daemon = True
        webcam_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': f'Webcam detection dimulai dari source {source}',
            'note': 'Tekan Ctrl+C di terminal server untuk menghentikan'
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'Error starting webcam: {str(e)}'}), 500

if __name__ == '__main__':
    print(f"[{datetime.now()}] Starting Flask YOLOv8 Detection Server v2.0...")
    print(f"[{datetime.now()}] Upload folder: {UPLOAD_FOLDER}")
    print(f"[{datetime.now()}] Model status: {'Loaded' if detector.model else 'Not loaded'}")
    
    if detector.model:
        model_info = detector.get_model_info()
        print(f"[{datetime.now()}] Model classes: {list(model_info['model_names'].values())}")
        print(f"[{datetime.now()}] Total classes: {model_info['num_classes']}")
    
    print(f"[{datetime.now()}] Server siap menerima request...")
    
    # Jalankan server Flask
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
