from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import os
from scipy.spatial.distance import cosine
import librosa
app = Flask(__name__)

# 配置 SQL Server 資料庫的連接 URI
app.config['SQLALCHEMY_DATABASE_URI'] = 'mssql+pyodbc://sa:jim93329@localhost/account_voice?driver=ODBC+Driver+17+for+SQL+Server'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# 初始化 SQLAlchemy
db = SQLAlchemy(app)

# 創建資料庫模型
class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(50), unique=True, nullable=False)
    voice_features = db.Column(db.LargeBinary, nullable=False)

# 創建資料庫表
with app.app_context():
    db.create_all()

# 提取 MFCC 特徵的函數
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    return mfcc_mean

# 註冊功能
@app.route('/')
def home():
    return render_template('index.html')
# 註冊功能
@app.route('/register', methods=['POST'])
def register():
    if 'audio' not in request.files or 'user_id' not in request.form:
        return jsonify({"success": False, "message": "No audio or user_id provided"}), 400

    audio_file = request.files['audio']
    user_id = request.form['user_id']

    # 將音頻文件保存並提取聲音特徵
    audio_path = os.path.join('temp_register_audio.wav')
    audio_file.save(audio_path)
    mfcc_features = extract_mfcc(audio_path)

    # 檢查用戶是否已經註冊
    existing_user = User.query.filter_by(user_id=user_id).first()
    if existing_user:
        return jsonify({"success": False, "message": "User already registered"}), 400

    # 將聲音特徵存儲到資料庫
    new_user = User(user_id=user_id, voice_features=mfcc_features.tobytes())
    db.session.add(new_user)
    db.session.commit()

    return jsonify({"success": True, "message": "User registered successfully"})

# 聲音驗證功能
@app.route('/verify-voice', methods=['POST'])
def verify_voice():
    if 'audio' not in request.files or 'user_id' not in request.form:
        return jsonify({"success": False, "message": "No audio or user_id provided"}), 400

    audio_file = request.files['audio']
    user_id = request.form['user_id']

    # 將音頻文件保存並提取聲音特徵
    audio_path = os.path.join('temp_audio.wav')
    audio_file.save(audio_path)
    mfcc_features = extract_mfcc(audio_path)

    # 從資料庫中檢索用戶
    user = User.query.filter_by(user_id=user_id).first()
    if user:
        stored_features = np.frombuffer(user.voice_features, dtype=np.float32)
        similarity = 1 - cosine(stored_features, mfcc_features)

        if similarity > 0.8:  # 假設相似度超過 0.8 視為匹配
            return jsonify({"success": True, "message": "Voice verification successful"})
        else:
            return jsonify({"success": False, "message": "Voice verification failed"})

    return jsonify({"success": False, "message": "User not found"})

if __name__ == '__main__':
    app.run(debug=True)
