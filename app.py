import os
import numpy as np
import cv2
from flask import Flask, render_template, request
from mtcnn.mtcnn import MTCNN
from tensorflow.keras.models import load_model

app = Flask(__name__, static_folder='static')

# Tải lại mô hình từ tệp my_model.h5
loaded_model = load_model("my_model.h5")

# Tạo một đối tượng MTCNN
mtcnn = MTCNN()

# Khởi tạo từ điển để ánh xạ tên người nổi tiếng sang các giá trị số nguyên
class_dict = {'Anushka_Sharma': 0, 
              'Barack_Obama': 1, 
              'Bill_Gates': 2, 
              'Dalai_Lama': 3, 
              'Indira_Nooyi': 4, 
              'Melinda_Gates': 5, 
              'Narendra_Modi': 6, 
              'Sundar_Pichai': 7, 
              'Vikas_Khanna': 8, 
              'Virat_Kohli': 9}

# Tạo một từ điển ngược (inverted dictionary) để ánh xạ các giá trị số nguyên thành tên người nổi tiếng
inv_dict = {v: k for k, v in class_dict.items()}

# Định nghĩa hàm để dự đoán người nổi tiếng từ hình ảnh
def predict_celebrity(image):
    # Sử dụng MTCNN để cắt khuôn mặt từ hình ảnh
    faces = mtcnn.detect_faces(image)
    
    # Kiểm tra xem có khuôn mặt được phát hiện trong hình ảnh hay không
    if faces:
        # Lấy tọa độ của khuôn mặt đầu tiên
        x, y, w, h = faces[0]['box']
        
        # Cắt khuôn mặt từ hình ảnh gốc
        face_img = image[y:y+h, x:x+w]
        
        # Định dạng lại kích thước của khuôn mặt thành 32x32
        scalled_face_img = cv2.resize(face_img, (32, 32))
        
        # Thực hiện dự đoán bằng mô hình
        img_for_prediction = scalled_face_img.reshape(1, 32, 32, 3).astype(float) / 255.0
        
        prediction = loaded_model.predict(img_for_prediction)
        predicted_class = np.argmax(prediction)
        
        # Trả về tên người nổi tiếng dự đoán
        return predicted_class
    else:
        return None

# Định nghĩa route cho trang chủ
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Kiểm tra xem có tệp hình ảnh được tải lên hay không
        if 'image' not in request.files:
            return render_template('index.html', error='Không có tệp hình ảnh được tải lên.')
        
        file = request.files['image']
        
        # Kiểm tra xem tên tệp hợp lệ
        if file.filename == '':
            return render_template('index.html', error='Tên tệp không hợp lệ.')
        
        # Kiểm tra định dạng tệp
        if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
            return render_template('index.html', error='Chỉ hỗ trợ các tệp JPG và PNG.')
        
        # Đọc hình ảnh từ tệp đã tải lên
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        
        # Dự đoán người nổi tiếng từ hình ảnh
        predicted_class = predict_celebrity(image)
        
        if predicted_class is not None:
            # Trả về tên người nổi tiếng dự đoán
            celebrity_name = inv_dict[predicted_class]
            return render_template('index.html', celebrity=celebrity_name)
        else:
            return render_template('index.html', error='Không tìm thấy khuôn mặt trong hình ảnh.')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
