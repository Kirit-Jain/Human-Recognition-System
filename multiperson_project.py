#importing the libraries used
from ultralytics import YOLO
import time
import torch
import cv2
import numpy as np
import os
import open_clip
from sentence_transformers import util
from PIL import Image
from deep_sort.deep_sort import DeepSort
#from skimage.metrics import structural_similarity as ssim

def SaveBoundingImage(box, frame, track_id):
    x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
    crop_img = frame[y1:y2, x1:x2]  # Crop the image using the bounding box
    track_folder = f"tracked_images/track_id_{track_id}"
    os.makedirs(track_folder, exist_ok=True)
    img_path = os.path.join(track_folder, f"{time.time()}.jpg")
    cv2.imwrite(img_path, crop_img)  # Save the cropped image to the folder

#Setting GPU/CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_img, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model_img.to(device)

def imageEncoder(img):
    #checking if the iage is valid
    if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
        raise ValueError("The image has zero width or height")
    
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model_img.encode_image(img1)
    return img1

def generateScore(enc_img1, enc_img2):
    #test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    #data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    #img1 = imageEncoder(image1)
    #img2 = imageEncoder(image2)
    cos_scores = util.pytorch_cos_sim(enc_img1, enc_img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score
    
def CompareImages(enc_img1, enc_img2):
    # image processing model
    #print(f"similarity Score: ", round(generateScore(image1, image2), 2))
    return generateScore(enc_img1, enc_img2)

def FindBestMatchingTrackID(crop_img, unique_track_ids, track_cache):
    best_match_id = None
    highest_similarity = 0.0
    
    try:
        crop_img_enc = imageEncoder(crop_img)
    except ValueError as e: 
        print(f"Error in encoding crop_img: {e}")
        return None
    
    for track_id in unique_track_ids:
        track_folder = f"tracked_images/track_id_{track_id}"
        for img_name in os.listdir(track_folder):
            img_path = os.path.join(track_folder, img_name)
            if(img_path not in track_cache):
                saved_img = cv2.imread(img_path)
                if saved_img is None or saved_img.size == 0:
                    print(f"Skipped zero-sized image: {img_path}")
                    continue
                track_cache[img_path] = imageEncoder(saved_img)
            similarity = CompareImages(crop_img_enc, track_cache[img_path])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match_id = track_id

    return best_match_id if highest_similarity > 0.65 else None



#Loading the model
model = YOLO("yolov8n.pt")

#Initializing weights
deep_sort_weight = 'E:/multiperson_project/Tracking-and-counting-Using-YOLOv8-and-DeepSORT/deep_sort/deep/checkpoint/ckpt.t7'
tracker = DeepSort(model_path = deep_sort_weight, max_age = 70)

#Taking the video feed from webcam
cap = cv2.VideoCapture(0)

#Get the Video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_fps = cap.get(cv2.CAP_PROP_FPS)

#defining VideoWriter Object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_path = 'output/output.mp4'
out = cv2.VideoWriter(output_path, fourcc, frame_fps, (frame_width, frame_height))

#initializing variable
frames = []
unique_track_ids = set()
track_cache = {}
i = 0
counter, fps, elapsed = 0, 0, 0
start_time = time.perf_counter()

#Staring the while loop to starting the detecting process
while cap.isOpened():
    ret, frame = cap.read()
    
    if ret:
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()
        
        results = model(frame, device='cpu', classes = 0, conf = 0.8)
        class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        for result in results:
            boxes = result.boxes #box object
            probs = result.probs #classification class probablity
            cls = boxes.cls.tolist()
            xyxy = boxes.xyxy
            conf = boxes.conf
            xywh = boxes.xywh
            for class_index in cls:
                class_name = class_names[int(class_index)]
            
        pred_cls = np.array(cls)
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        boxes_xywh = xywh
        boxes_xywh = xywh.cpu().numpy()
        boxes_xywh = np.array(boxes_xywh, dtype=float)
        
        tracks = tracker.update(boxes_xywh, conf, og_frame)
        
        for track in tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr() #Get the coordinates of bounding box
            w = x2 - x1 #width of bounding box
            h = y2 - y1 #height of the bounding box
            
            red_color = (0, 0, 255)
            blue_color = (255, 0, 0)
            green_color = (0, 255, 0)
            
            color_id = track_id % 3
            if color_id == 0:
                color = red_color
            elif color_id == 1:
                color = blue_color
            else:
                color = green_color
                
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1+w), int(y2+h)), color, 2)
            
            text_color = (0, 0, 0)
            cv2.putText(og_frame, f"{class_name}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
            
            unique_track_ids.add(track_id)
                
            
            if(os.path.exists(f"tracked_images/track_id_{track_id}")):
                # Save the bounding box image and update track ID if necessary
                crop_img = og_frame[int(y1):int(y2), int(x1):int(x2)]
                if crop_img.size == 0 or crop_img.shape[0] == 0 or crop_img.shape[1] == 0:  #<----------------
                    print(f"Skipped zero-sized crop image.")  #<----------------
                    continue  # Skip processing this image if it's zero-sized
                    
                    
                if unique_track_ids:
                    matched_track_id = FindBestMatchingTrackID(crop_img, unique_track_ids, track_cache)
                    if matched_track_id is not None:
                        track_id = matched_track_id
                        SaveBoundingImage((x1, y1, x2, y2), og_frame, track_id)
            else:
                SaveBoundingImage((x1, y1, x2, y2), og_frame, track_id) 
        
        frames.append(og_frame)
        
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))
        
        cv2.imshow("Video", og_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
             break
        
cap.release()
out.release()
cv2.destroyAllWindows()
