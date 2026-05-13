import os
import cv2
import time
import joblib
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
from skimage.feature import hog
from ultralytics import YOLO

app = Flask(__name__)

# --- CẤU HÌNH TOÀN CỤC ---
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class Config:
    filepath = None       
    input_type = None     
    current_model = 'hog'  
    state = 'stopped'    
    
    # === CẤU HÌNH YOLO ===
    yolo_model_path = 'best.pt' 
    yolo_conf = 0.5

    # === CẤU HÌNH HOG ===
    hog_step_size = 32
    hog_window_size = (128, 128)
    hog_orientations = 9
    hog_pixels_per_cell = (8, 8)
    hog_cells_per_block = (2, 2)
    hog_image_size = (64, 64)

config = Config()

# --- 1. LOAD MODEL YOLO ---
print(f"🔄 Đang tìm model YOLO tại: {config.yolo_model_path} ...")
if os.path.exists(config.yolo_model_path):
    try:
        yolo_model = YOLO(config.yolo_model_path)
        print(f"✅ Đã tải thành công model: {config.yolo_model_path}")
    except Exception as e:
        print(f"❌ Lỗi khi tải {config.yolo_model_path}: {e}")
        yolo_model = YOLO("yolov8n.pt")
else:
    print(f"⚠️ Không thấy file '{config.yolo_model_path}' cùng cấp với app.py")
    yolo_model = YOLO("yolov8n.pt")

# --- 2. LOAD MODEL SVM (HOG) ---
svm_clf = None
label_encoder = None
try:
    if os.path.exists('vehicle_svm_model.pkl') and os.path.exists('label_encoder.pkl'):
        svm_clf = joblib.load('vehicle_svm_model.pkl')
        label_encoder = joblib.load('label_encoder.pkl')
        print("✅ Đã tải model SVM & Label Encoder")
    else:
        print("⚠️ Thiếu file model SVM. Chức năng HOG sẽ không hoạt động.")
except Exception as e:
    print(f"❌ Lỗi khi tải SVM: {e}")

# --- CÁC HÀM XỬ LÝ ---
def extract_hog_features(image):
    # Đọc và chuyển về ảnh xám (HOG hiệu quả nhất trên ảnh xám)
    image_resized = cv2.resize(image, config.hog_image_size)

    # Trích xuất đặc trưng
    fd = hog(image_resized, 
             orientations=config.hog_orientations, # 9 khoảng hướng từ 0-180 độ
             pixels_per_cell=config.hog_pixels_per_cell, # Mỗi ô 8x8 pixels
             cells_per_block=config.hog_cells_per_block, #Khối 2x2 ô để chuẩn hóa
             visualize=False, 
             channel_axis=-1)
    return fd

def sliding_window(image, step_size, window_size):
    # Quét qua toàn bộ ảnh theo từng bước nhảy (Step Size)
    for y in range(0, image.shape[0] - window_size[1], step_size):
        for x in range(0, image.shape[1] - window_size[0], step_size):
            # Cắt lấy vùng ảnh tại vị trí (x, y)
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# Tách logic nhận diện ra hàm riêng để dùng chung cho ảnh và video
def process_frame(frame, fps):
    display_frame = frame.copy()
    
    # === 1. LOGIC HOG + SVM ===
    if config.current_model == 'hog':
        if svm_clf is not None:
            step = config.hog_step_size
            win_w, win_h = config.hog_window_size
            
            for (x, y, window) in sliding_window(frame, step, (win_w, win_h)):
                if window.shape[0] != win_h or window.shape[1] != win_w:
                    continue
                
                feat = extract_hog_features(window)
                prediction = svm_clf.predict([feat])[0]
                
                try:
                    raw_scores = svm_clf.decision_function([feat])[0]
                    score = np.max(raw_scores) if np.ndim(raw_scores) > 0 else raw_scores
                except:
                    score = 1.0 

                pred_label = label_encoder.inverse_transform([prediction])[0]
                
                if pred_label != 'neg' and score > 0.5:
                    cv2.rectangle(display_frame, (x, y), (x + win_w, y + win_h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"{pred_label}", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            cv2.putText(display_frame, "MISSING SVM MODEL", (50, 240), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

    # === 2. LOGIC YOLO ===
    elif config.current_model == 'yolo':
        # Dự đoán trực tiếp trên toàn bộ khung hình
        results = yolo_model.predict(source=frame, conf=config.yolo_conf, verbose=False)
        res = results[0]
        
        for box in res.boxes:
            # Lấy tọa độ khung bao (Bounding Box)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])  # Độ tin cậy của dự đoán
            label = res.names[cls] if res.names else str(cls) # Tên phương tiện (xe máy, xe hơi,...)
            
            # Vẽ khung bao màu cam (Orange) 
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            text = f"{label} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (x1, y1 - 20), (x1 + w, y1), (0, 165, 255), -1)
            cv2.putText(display_frame, text, (x1, y1 - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Nếu FPS = 0 (tức là ảnh tĩnh), hiển thị STATIC IMAGE
    fps_text = f"FPS: {fps:.1f}" if fps > 0 else "STATIC IMAGE"
    cv2.putText(display_frame, f"Mode: {config.current_model.upper()} | {fps_text}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
    return display_frame

# Đưa kết quả lê trình duyệt
def generate_frames():
    cap = None
    prev_time = 0
    
    # Tạo sẵn màn hình đen Standby
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank_frame, "STANDBY - WAITING FOR INPUT...", (120, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    ret, blank_buffer = cv2.imencode('.jpg', blank_frame)
    blank_bytes = blank_buffer.tobytes()
    
    last_frame_bytes = blank_bytes
    
    while True:
        if config.state == 'stopped' or config.filepath is None:
            if cap is not None:
                cap.release()
                cap = None
            last_frame_bytes = blank_bytes
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_frame_bytes + b'\r\n')
            time.sleep(0.5)
            continue

        # XỬ LÝ ẢNH TĨNH
        if config.input_type == 'image':
            frame = cv2.imread(config.filepath)
            if frame is not None:
                frame = cv2.resize(frame, (640, 480))
                # Gọi hàm nhận diện, truyền FPS = 0
                display_frame = process_frame(frame, 0)
                ret, buffer = cv2.imencode('.jpg', display_frame)
                last_frame_bytes = buffer.tobytes()
                
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_frame_bytes + b'\r\n')
            time.sleep(0.5) # Chờ 0.5s để tiết kiệm tài nguyên hệ thống
            continue

        # XỬ LÝ VIDEO
        elif config.input_type == 'video':
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(config.filepath)

            if config.state == 'replay':
                if cap is not None and cap.isOpened():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                config.state = 'playing'
                
            if config.state == 'paused':
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + last_frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue

            success, frame = cap.read()
            if not success:
                config.state = 'stopped' 
                continue

            frame = cv2.resize(frame, (640, 480))
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            # Chạy nhận diện
            display_frame = process_frame(frame, fps)

            ret, buffer = cv2.imencode('.jpg', display_frame)
            last_frame_bytes = buffer.tobytes() 
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + last_frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

# Xử lý tệp tin do người dùng tải lên
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)
        
        # Phân loại đầu vào dựa trên đuôi file
        ext = file.filename.rsplit('.', 1)[-1].lower()
        if ext in ['jpg', 'jpeg', 'png']:
            config.input_type = 'image'
        elif ext in ['mp4', 'avi', 'mov']:
            config.input_type = 'video'
        else:
            return "Unsupported file format"

        config.filepath = filepath
        config.state = 'playing'  
        print(f"🎬 Đã nhận {config.input_type} mới: {filepath}")
        return render_template('index.html')


#Nhận yêu cầu từ giao diện (thông qua AJAX) để thay đổi mô hình
@app.route('/update_config', methods=['POST'])
def update_config():
    data = request.json
    config.current_model = data.get('model', 'hog')
    config.hog_step_size = int(data.get('step_size', 32))
    config.yolo_conf = float(data.get('conf_thres', 0.5))
    print(f"⚙️ Cập nhật: Model={config.current_model}, Step={config.hog_step_size}, Conf={config.yolo_conf}")
    return jsonify({'status': 'success'})

# Đây là luồng dữ liệu chính hiển thị kết quả nhận diện lên màn hình Live Monitor.
@app.route('/video_control', methods=['POST'])
def video_control():
    data = request.json
    action = data.get('action') 
    
    if action == 'play':
        config.state = 'playing'
    elif action == 'pause':
        config.state = 'paused'
    elif action == 'stop':
        config.state = 'stopped'
    elif action == 'replay':
        config.state = 'replay'
        
    print(f"⏯️ Trạng thái thay đổi thành: {config.state.upper()}")
    return jsonify({'status': 'success', 'state': config.state})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)