
# Underwater Fish Species Classification using YOLOv8

## 📌 Project Overview
This project implements an **Underwater Fish Species Classification System** using **YOLOv8**, a state-of-the-art object detection model.  
The system detects fish in underwater images and classifies them into **13 different fish species**.

The trained deep learning model is integrated into an **offline web application** where users can upload an image and instantly view:
- Detected fish
- Bounding boxes
- Fish species name
- Confidence score

This project was developed as a **minor academic project**.

---

## 🎯 Objectives
- Detect fish in underwater images
- Classify fish into multiple species
- Train a YOLOv8 multi-class detection model
- Evaluate model performance using standard metrics
- Deploy the model in an **offline web application**

---

## 🧠 Technology Stack
- **Programming Language:** Python 3.10
- **Deep Learning Framework:** PyTorch
- **Model Architecture:** YOLOv8 (Ultralytics)
- **Web Framework:** Streamlit
- **Development Platform:** Google Colab (training), Local PC (deployment)

### Libraries Used
- ultralytics
- torch, torchvision
- streamlit
- pillow
- opencv-python

---

## 📂 Dataset Information
- **Source:** Kaggle (Roboflow Dataset)
- **Dataset Name:** Fish Detection
- **Dataset Link:**  
  https://www.kaggle.com/datasets/zehraatlgan/fish-detection

### Fish Species (13 Classes)
1. AngelFish  
2. BlueTang  
3. ButterflyFish  
4. ClownFish  
5. GoldFish  
6. Gourami  
7. MorishIdol  
8. PlatyFish  
9. RibbonedSweetlips  
10. ThreeStripedDamselfish  
11. YellowCichlid  
12. YellowTang  
13. ZebraFish  

---

## 📁 Project Structure
```

fish_offline_app/
│
├── app.py            # Streamlit web application
├── best.pt           # Trained YOLOv8 model weights
├── dataset/          # Local dataset (optional)
│   ├── train/
│   ├── valid/
│   └── test/
├── README.md         # Project documentation
└── venv/             # Python virtual environment

````

---

## ⚙️ Model Training Details
- **Base Model:** yolov8n.pt (pretrained on COCO)
- **Training Type:** Transfer Learning
- **Epochs:** 10
- **Image Size:** 640 × 640
- **Number of Classes:** 13
- **Loss Functions:** YOLOv8 default (box, cls, dfl)

### 📊 Final Training Metrics
- **Precision:** ~0.86
- **Recall:** ~0.87
- **mAP@50:** ~0.91
- **mAP@50–95:** ~0.66

These metrics indicate strong detection and classification performance.

---

## 🌐 Web Application (Offline)
The trained YOLOv8 model is deployed using **Streamlit** as an offline web application.

### Features
- Upload underwater fish image (JPG / PNG)
- Detect fish in the image
- Display bounding boxes
- Show fish species name and confidence score

### Run the App
```bash
streamlit run app.py
````

The application runs locally at:

```
http://localhost:8501
```

---

## 🧪 Testing the Application

You can test the application using:

* Images from the dataset (`dataset/train/images`)
* Underwater fish images from Google Images
* Free images from Unsplash or Pexels
* Screenshots from underwater fish videos

---

## ➕ How to Add a New Dataset (Future Extension)

This project is designed to be **extendable**.
If someone wants to add a **new fish dataset or new species**, follow these steps.

---

### Step 1: Prepare the New Dataset

The dataset must be in **YOLO format**:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each label file must contain:

```
<class_id> <x_center> <y_center> <width> <height>
```

---

### Step 2: Update `data.yaml`

Example for adding **2 new species**:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 15
names:
  - AngelFish
  - BlueTang
  - ButterflyFish
  - ClownFish
  - GoldFish
  - Gourami
  - MorishIdol
  - PlatyFish
  - RibbonedSweetlips
  - ThreeStripedDamselfish
  - YellowCichlid
  - YellowTang
  - ZebraFish
  - LionFish
  - Shark
```

---

### Step 3: Retrain the Model

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=10,
    imgsz=640
)
```

---

### Step 4: Replace Model in Web App

After training:

* Download the new `best.pt`
* Replace the old `best.pt` in `fish_offline_app`
* Restart Streamlit

The web app will automatically support new species.

---

## 🚀 Future Enhancements

* Real-time video detection
* Fish counting and tracking
* Cloud deployment
* Mobile application integration
* Species population analytics

---

## 👨‍🎓 Academic Declaration

This project was developed strictly for **academic and educational purposes**.
All datasets and tools used are **open-source** and publicly available.

---

## 📜 License

This project is intended for educational use only.

---

## 🙌 Acknowledgements

* Ultralytics YOLOv8
* Kaggle Datasets
* Roboflow
* Google Colab
* Open-source Python community

```

---
