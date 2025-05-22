This project is a real-time human recognition and tracking system built using YOLOv8, Deep SORT, and CLIP-based image similarity matching. The system captures live webcam footage, detects and tracks individuals, and identifies previously seen individuals using visual embeddings.

## 🔧 Features

- 🧠 **Human Detection** using YOLOv8  
- 🔁 **Multi-person tracking** using Deep SORT algorithm  
- 📸 **Saves bounding-box cropped images** for each tracked person  
- 🖥️ **Real-time webcam input** with dynamic bounding box drawing  
- 🎯 **Intelligent track ID reassignment** based on image similarity  
- 📂 **Outputs annotated video** and stores tracking images for analysis  

🚀 How It Works
    Captures live video feed via webcam.
    Detects humans using YOLOv8 (confidence threshold ≥ 0.8).
    Tracks each person using Deep SORT and assigns a unique ID.
    Saves cropped images of each person into folders (tracked_images/track_id_X).
    If the same person is re-detected, it compares with saved images using CLIP embeddings and reassigns the correct ID.
    Annotates frames with bounding boxes and track IDs, and writes to the output video.

    Output Video: https://shorturl.at/LLBJ5
