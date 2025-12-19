from flask import Flask, render_template, redirect, request, url_for, send_file, send_from_directory, flash
from flask import jsonify, json
from werkzeug.utils import secure_filename
import datetime
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from models import db, User
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import face_recognition
from torch.autograd import Variable
import time
import uuid
import sys
import traceback
import logging
import zipfile
from torch import nn
import torch.nn.functional as F
from torchvision import models
from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import LinearSegmentedColormap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the absolute path for the upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Uploaded_Files')
FRAMES_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'frames')
GRAPHS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'graphs')
DATASET_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Admin', 'datasets')

# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(GRAPHS_FOLDER, exist_ok=True)
os.makedirs(DATASET_FOLDER, exist_ok=True)

# Ensure folders have proper permissions
os.chmod(FRAMES_FOLDER, 0o755)
os.chmod(GRAPHS_FOLDER, 0o755)
os.chmod(DATASET_FOLDER, 0o755)

video_path = ""
detectOutput = []

app = Flask("__main__", template_folder="templates", static_folder="static")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure secret key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Initialize SQLAlchemy
db.init_app(app)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Create all database tables
with app.app_context():
    db.create_all()

# Dataset comparison accuracies
DATASET_ACCURACIES = {
    'Our Model': None,
    'FaceForensics++': 85.1,
    'DeepFake Detection Challenge': 82.3,
    'DeeperForensics-1.0': 80.7
}

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match")

        user = User.query.filter_by(email=email).first()
        if user:
            return render_template('signup.html', error="Email already exists")

        user = User.query.filter_by(username=username).first()
        if user:
            return render_template('signup.html', error="Username already exists")

        new_user = User(username=username, email=email)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()

        login_user(new_user)
        return redirect(url_for('homepage'))

    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('homepage'))
        else:
            return render_template('login.html', error="Invalid email or password")

    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('homepage'))

def generate_confidence_graph(confidence, predicted_label):
    try:
        # Bigger graph size
        fig, ax = plt.subplots(figsize=(15, 15))
        fig.patch.set_facecolor('#f7f6f2')  # Match background color
        ax.set_facecolor('#f7f6f2')

        # Colors
        real_color = '#2ecc71'  # green
        fake_color = '#e74c3c'  # red
        
        # Labels always in fixed order
        labels = ['Real', 'Fake']

        # Real vs Fake percentage mapping
        # Colors should always be: Real = green, Fake = red
        if predicted_label.upper() == "REAL":
            sizes = [confidence, 100 - confidence]
        else:
            sizes = [100 - confidence, confidence]
        
        # Colors always in order: [Real (green), Fake (red)]
        colors = [real_color, fake_color]

        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(
            sizes,
            explode=explode,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            pctdistance=0.85,
            labeldistance=1.1,
            startangle=90,
            shadow=False,
            textprops={'color': '#1f2e39', 'fontsize': 24, 'weight': 'bold'},  # Dark text, larger font
            wedgeprops={'edgecolor': '#ebe7dc', 'linewidth': 3}
        )

        # Style the percentage text
        for autotext in autotexts:
            autotext.set_color('#1f2e39')
            autotext.set_fontsize(22)
            autotext.set_weight('bold')
        
        # Style the labels (Real/Fake)
        for text in texts:
            text.set_color('#1f2e39')
            text.set_fontsize(26)
            text.set_weight('bold')
            text.set_fontfamily('Inter, sans-serif')

        ax.set_title("Confidence Score", color="#1f2e39", fontsize=32, pad=30, weight='bold')

        # Save bigger image
        unique_id = str(uuid.uuid4()).split('-')[0]
        graph_filename = f'confidence_{unique_id}.png'
        graph_path = os.path.join(GRAPHS_FOLDER, graph_filename)

        plt.savefig(graph_path, dpi=450, bbox_inches='tight', facecolor='#f7f6f2', edgecolor='none')
        plt.close()

        return f'graphs/{graph_filename}'

    except Exception as e:
        logger.error(f"Error generating confidence graph: {e}")
        traceback.print_exc()
        return None



class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size*seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm,_ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))

def extract_frames(video_path, num_frames=16):
    frames = []
    frame_paths = []
    unique_id = str(uuid.uuid4()).split('-')[0]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error opening video file")
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise Exception("Video file appears to be empty")
        
    interval = total_frames // num_frames
    
    count = 0
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % interval == 0 and frame_count < num_frames:
            faces = face_recognition.face_locations(frame)
            if len(faces) == 0:
                continue
                
            try:
                top, right, bottom, left = faces[0]
                face_frame = frame[top:bottom, left:right, :]
                frame_path = os.path.join(FRAMES_FOLDER, f'frame_{unique_id}_{frame_count}.jpg')
                cv2.imwrite(frame_path, face_frame)
                frame_paths.append(os.path.basename(frame_path))
                frames.append(face_frame)
                frame_count += 1
                logger.info(f"Extracted frame {frame_count}: {os.path.basename(frame_path)}")
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {str(e)}")
                continue
                
        count += 1
        if frame_count >= num_frames:
            break
            
    cap.release()
    
    if len(frames) == 0:
        raise Exception("No faces detected in the video")
        
    return frames, frame_paths

def predict(model, img, path='./'):
    try:
        with torch.no_grad():
            fmap, logits = model(img.to())
            params = list(model.parameters())
            weight_softmax = model.linear1.weight.detach().cpu().numpy()
            logits = F.softmax(logits, dim=1)
            _, prediction = torch.max(logits, 1)
            confidence = logits[:, int(prediction.item())].item()*100
            # Get probabilities for both classes (Real=0, Fake=1)
            real_prob = logits[0][0].item()*100
            fake_prob = logits[0][1].item()*100
            logger.info(f'Prediction confidence: {confidence}%')
            return [int(prediction.item()), confidence, real_prob, fake_prob]
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        traceback.print_exc()
        raise

class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0,a)
        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top,right,bottom,left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def detectFakeVideo(videoPath):
    start_time = time.time()
    
    try:
        im_size = 112
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size,im_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)
        ])
        
        path_to_videos = [videoPath]
        video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)
        model = Model(2)
        path_to_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model/df_model.pt')
        
        if not os.path.exists(path_to_model):
            raise Exception("Model file not found")
            
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()
        
        prediction = predict(model, video_dataset[0], './')
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        
        return prediction, processing_time
    except Exception as e:
        logger.error(f"Error in detectFakeVideo: {str(e)}")
        traceback.print_exc()
        raise

def get_datasets():
    datasets = []
    for item in os.listdir(DATASET_FOLDER):
        if item.endswith('.zip'):
            path = os.path.join(DATASET_FOLDER, item)
            stats = os.stat(path)
            datasets.append({
                'name': item,
                'size': stats.st_size,
                'upload_date': datetime.datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    return datasets

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/')
def homepage():
    return render_template('home.html')

@app.route('/admin')
@login_required
def admin():
    datasets = get_datasets()
    return render_template('admin.html', datasets=datasets)

@app.route('/admin/upload', methods=['POST'])
@login_required
def admin_upload():
    if 'dataset' not in request.files:
        return jsonify({'success': False, 'error': 'No file uploaded'})
        
    dataset = request.files['dataset']
    if dataset.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
        
    if not dataset.filename.lower().endswith('.zip'):
        return jsonify({'success': False, 'error': 'Invalid file format. Please upload ZIP files only.'})
        
    try:
        filename = secure_filename(dataset.filename)
        filepath = os.path.join(DATASET_FOLDER, filename)
        dataset.save(filepath)
        
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.testzip()
            
        logger.info(f"Dataset uploaded successfully: {filename}")
        return jsonify({
            'success': True,
            'message': 'Dataset uploaded successfully',
            'dataset': {
                'name': filename,
                'size': os.path.getsize(filepath),
                'upload_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        })
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Error uploading dataset: {str(e)}")
        return jsonify({'success': False, 'error': f'Error uploading dataset: {str(e)}'})

@app.route('/detect', methods=['GET', 'POST'])
@login_required
def detect():
    if request.method == 'GET':
        return render_template('detect.html')
    if request.method == 'POST':
        if 'video' not in request.files:
            return render_template('detect.html', error="No video file uploaded")
            
        video = request.files['video']
        if video.filename == '':
            return render_template('detect.html', error="No video file selected")
            
        if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return render_template('detect.html', error="Invalid file format. Please upload MP4, AVI, or MOV files.")
            
        video_filename = secure_filename(video.filename)
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        video.save(video_path)
        
        try:
            logger.info(f"Processing video: {video_filename}")
            
            frames, frame_paths = extract_frames(video_path)
            
            if not frames:
                raise Exception("No frames could be extracted from the video")
            
            prediction, processing_time = detectFakeVideo(video_path)
            
            if prediction[0] == 0:
                output = "REAL"
            else:
                output = "FAKE"
            confidence = prediction[1]
            real_prob = prediction[2]
            fake_prob = prediction[3]
            
            logger.info(f"Video prediction: {output} with confidence {confidence}%")
            
            confidence_image = generate_confidence_graph(confidence,output)
            if not confidence_image:
                raise Exception("Failed to generate confidence graph")
                
            
            data = {
                'output': output, 
                'confidence': confidence,
                'real_probability': round(real_prob, 2),
                'fake_probability': round(fake_prob, 2),
                'frames': frame_paths,
                'processing_time': round(processing_time, 2),
                'confidence_image': confidence_image,
            }
            
            logger.info(f"Sending response data: {data}")
            data = json.dumps(data)
            
            os.remove(video_path)
            return render_template('detect.html', data=data)
            
        except Exception as e:
            if os.path.exists(video_path):
                os.remove(video_path)
            error_msg = str(e)
            logger.error(f"Error processing video: {error_msg}")
            traceback.print_exc()
            return render_template('detect.html', error=f"Error processing video: {error_msg}")

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)