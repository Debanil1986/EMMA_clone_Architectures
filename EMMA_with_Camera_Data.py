#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/Debanil1986/Waymo_indivitualProject/blob/main/EMMA_with_Camera_Data.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# In[1]:


import torch
import torch.nn as nn
import cv2
import numpy as np
import torch
from torchvision.models import resnet50


class CustomCNN(nn.Module):
    def __init__(self, output_dim=512):
        super(CustomCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Reduce spatial size by half
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Further reduce spatial size
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # Global average pooling to output 1x1 feature map
        )
        self.fc = nn.Linear(256, output_dim)  # Fully connected layer to reduce to output_dim

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)  # Flatten the spatial dimensions
        x = self.fc(x)
        return x

# In[2]:


from torchvision.models.detection import fasterrcnn_resnet50_fpn


def load_pretrained_object_detector():
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model

object_detector = load_pretrained_object_detector()  # Load the model at the start of your script or main function

# In[1]:


from tqdm import tqdm
import os
from flask import Flask, request, redirect, url_for,jsonify
from werkzeug.utils import secure_filename
import asyncio

cnn_feature_dim = 512
intent_dim = 10
historical_state_dim = 4
hidden_size = 512
resized_width, resized_height = 640, 480


class EMMA:
    def __init__(self, cnn_feature_dim, intent_dim, historical_state_dim,hidden_size):
        super(EMMA, self).__init__()
        self.cnn_feature_dim = 512  # Desired output feature size from CNN
        self.cnn = CustomCNN(output_dim=self.cnn_feature_dim)  # Use the custom CNN
        self.rnn_input_size = cnn_feature_dim + intent_dim + historical_state_dim
        self.rnn = torch.nn.LSTM(input_size=self.rnn_input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 2)
        
    def state_dict(self):
        state_dict = {
        'cnn.conv_layers.0.weight': self.cnn.conv_layers[0].weight,
        'cnn.conv_layers.0.bias': self.cnn.conv_layers[0].bias,
        'cnn.conv_layers.3.weight': self.cnn.conv_layers[3].weight,
        'cnn.conv_layers.3.bias': self.cnn.conv_layers[3].bias,
        'cnn.conv_layers.6.weight': self.cnn.conv_layers[6].weight,
        'cnn.conv_layers.6.bias': self.cnn.conv_layers[6].bias,
        'fc.weight': self.fc.weight,
        'fc.bias': self.fc.bias
        }
        return state_dict

    def preprocess_frame(self, frame):
        """Resize and normalize the frame."""
        # Example preprocessing: resize and normalize
        resized_frame = cv2.resize(frame, (resized_width, resized_height))
        normalized_frame = resized_frame / 255.0
        return normalized_frame

    def predict(self, frame, intents, historical_states):
        """Make a prediction using the preprocessed frame."""
        # Convert frame to a batch format (batch size 1)
        camera_frames = frame

        camera_frames_tensor = torch.tensor(camera_frames, dtype=torch.float32)

        batch_size, T, W, H, C = camera_frames_tensor.shape
        cnn_out = self.cnn(camera_frames_tensor.view(-1, C, H, W))  # Reshape and pass through CNN
        cnn_out = cnn_out.view(batch_size, T, -1)

        # Combine CNN output with intents and historical_states
        # Here you might need to encode intents and concatenate
        intents_tensor = torch.tensor(intents, dtype=torch.float32)  # Shape: (batch, time, intent_dim)
        historical_states_tensor = torch.tensor(historical_states, dtype=torch.float32)  # Shape: (batch, time, state_dim)

        combined_features = torch.cat((cnn_out, intents_tensor, historical_states_tensor), dim=-1)

        rnn_out, _ = self.rnn(combined_features)
        output = self.fc(rnn_out)
        return output

    def process_video(self, video_path):
        """Extract frames from a video and process them with the model."""
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            preprocessed_frame = self.preprocess_frame(frame)
            intents = np.random.rand(1, 1, intent_dim)  # Random intents
            historical_states = np.random.rand(1, 1, historical_state_dim)
            output = self.predict(preprocessed_frame, intents, historical_states)
            print(output)  # Print or further process the output
            # Display frame
            cv2.imshow('Video Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def preprocess_frame(frame):
    """Resize and normalize the frame."""
    # Resize the frame to the required input size of the model
    resized_frame = cv2.resize(frame, (resized_width, resized_height))  # Example resize
    # Normalize the frame if necessary
    normalized_frame = resized_frame / 255.0
    preprocessed_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)  # Add time dimension
    return preprocessed_frame

def draw_lane_overlay(frame, lane_points):
    """Draws a semi-transparent lane overlay."""
    overlay = frame.copy()
    cv2.fillPoly(overlay, [np.array(lane_points, np.int32)], (0, 255, 0))
    alpha = 0.4  # Transparency factor.
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def adjust_lane_points(frame_width, frame_height):
    # Example adjustment, these points should be dynamically calculated based on actual lane detection
    return [
        (frame_width * 0.4, frame_height),  # Bottom left
        (frame_width * 0.6, frame_height),  # Bottom right
        (frame_width * 0.55, frame_height * 0.7),  # Top right
        (frame_width * 0.45, frame_height * 0.7)   # Top left
    ]

def preprocess_frame_for_torch(frame):
    """Preprocess the frame for PyTorch model input."""
    # Convert frame to RGB (PyTorch models expect RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize the frame to the required input size of the model
    resized_frame = cv2.resize(frame_rgb, (resized_width, resized_height))  # Example resize
    # Normalize the frame to 0-1
    normalized_frame = resized_frame / 255.0
    # Convert to tensor
    tensor_frame = torch.from_numpy(normalized_frame).float()
    # Rearrange dimensions to (C, H, W) from (H, W, C)
    tensor_frame = tensor_frame.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    return tensor_frame


async def process_video(video_path, output_video_path, model, intent_dim, historical_state_dim):
    cap = cv2.VideoCapture(video_path)
    lengthOfFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened():
        print("Error: Unable to open the video.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))
    pbar = tqdm(total=lengthOfFrames, unit="frames")
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        preprocessed_frame = preprocess_frame(frame)
        preprocessed_tensor = preprocess_frame_for_torch(frame)
        with torch.no_grad():
          detection_output = object_detector(preprocessed_tensor)
          detections = detection_output[0]

        # Draw detections with high confidence scores

        labels = detections['labels'].cpu().numpy()
        boxes = detections['boxes'].cpu().numpy()
        scores = detections['scores'].cpu().numpy()
        scale_width = frame_width / resized_width
        scale_height = frame_width / resized_height
        for label, box, score in zip(labels, boxes, scores):
            if score > 0.9:  # Threshold can be adjusted
                x1, y1, x2, y2 = map(int, box)

                x1 = int(x1 * scale_width)
                y1 = int(y1 * scale_height)
                x2 = int(x2 * scale_width)
                y2 = int(y2 * scale_height)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Car: {score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        intents = np.random.rand(1, 1, intent_dim)
        historical_states = np.random.rand(1, 1, historical_state_dim)
        output = model.predict(preprocessed_frame, intents, historical_states)

        lane_points = adjust_lane_points(frame_width, frame_height)
        draw_lane_overlay(frame, lane_points)

        out.write(frame)
        pbar.update(1)
        await asyncio.sleep(0)

    cap.release()
    print(f"Output video saved to {output_video_path}")
    pbar.close()
    out.release()
    # cv2.destroyAllWindows()



app = Flask(__name__)
UPLOAD_FOLDER = './'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/convert-video-to-base64',methods=['POST'])
def convert_video_to_base64():
    emma_model = EMMA(cnn_feature_dim, intent_dim, historical_state_dim, hidden_size)  # Initialize the EMMA model
    print(request.files)
    if 'video' not in request.files:
            return 'No file part', 400
    file = request.files['video']
    if file and allowed_file(file.filename):
        filename = secure_filename("input.mp4")
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    video_path ='input.mp4'
    output_video_path = 'emma_processed_videos.mp4'
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_in_executor(None, lambda: asyncio.run(process_video(video_path, output_video_path, emma_model, intent_dim, historical_state_dim)))
    
    

    return jsonify({"message": "Processing started"}), 202

if __name__ == "__main__":
    app.run(debug=False,port=3000)

# In[ ]:



