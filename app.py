from flask import Flask, request, render_template, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import torchvision.transforms as T
import requests


app = Flask(__name__)

# ตั้งค่าที่เก็บไฟล์ที่อัปโหลด
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ตาราง HSV สำหรับแถบสีของตัวต้านทาน
# COLOUR_TABLE = [
#      # [lower_bound_HSV, upper_bound_HSV, color_name, tolerance, box_color]
#     [(0, 10, 0), (0, 0, 45), "BLACK", 0, (0, 0, 0)],
#     [(0, 105, 40), (25, 200, 70), "BROWN", 1, (0, 51, 102)],
#     [(0, 88, 120), (30, 255, 255), "RED", 2, (0, 0, 255)],
#     [(0, 140, 120), (10, 255, 255), "ORANGE", 3, (0, 128, 255)],
#     [(10, 86, 200), (84,170, 255), "YELLOW", 4, (0, 255, 255)],
#     [(45, 38, 80), (76, 255, 255), "GREEN", 5, (0, 255, 0)],
#     [(83, 36, 155), (134, 255, 255), "BLUE", 6, (255, 0, 0)],
#     [(270, 30, 35), (280, 255, 255), "PURPLE", 7, (255, 0, 125)],
#     [(0, 0, 50), (179, 45, 90), "GRAY", 8, (128, 128, 128)],
#     [(0, 0, 175), (179, 15, 250), "WHITE", 9, (255, 255, 255)],
# ]

COLOUR_TABLE = [                           
                [(0, 0, 0), (179, 255, 30)  , "BLACK"  , 0 , (0,0,0)],#done
                [(0, 100, 100), (15, 255, 255), "RED", 2, (0, 0, 255)],
                [(0, 60, 40), (15, 255, 200)  , "BROWN"  , 1 , (0,51,102)],
                #[(0, 90, 80)    , (10, 255, 100)  , "RED"    , 2 , (0,0,255)],
                #[(8, 140, 120)   , (16, 255, 255)  , "ORANGE" , 3 , (0,128,255)],
                [(13, 110, 120), (23, 255, 255), "ORANGE", 3, (0, 128, 255)],
                #[(20, 145, 0)   , (29, 255, 255)  , "YELLOW" , 4 , (0,255,255)],
                [(24,110, 120)   , (35, 255, 255)  , "YELLOW" , 4 , (0,255,255)   ],
                # [(30, 40, 40)    , (70, 255, 255)   , "GREEN"  , 5 , (0,255,0)    ],
                [(45, 38, 80), (70, 255, 255), "GREEN", 5, (0, 255, 0)],
                [(110, 50, 50)    , (139, 255, 255)  , "BLUE"   , 6 , (255,0,0)     ],
                [(140, 40, 0) , (150, 255, 255) , "PURPLE" , 7 , (255,0,127)   ],
                [(0, 0, 50)     , (179, 200, 200)   , "GRAY"   , 8 , (128,128,128) ],
                [(0, 200, 200)     , (179, 255, 255)  , "WHITE"  , 9 , (255,255,255) ],
                #[(0, 0, 110)     , (179, 30, 250)  , "WHITE"  , 9 , (255,255,255) ],
                [(24,110, 120)   , (35, 255, 255)  , "GOLD" , 10 , (0,255,255)   ]
                ]

# โมเดล RCNN
MODEL_PATH = "resistor_detector_onlyBody.pth"
BANDS_MODEL_PATH = "bandsV2_detector.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

def load_bands_model():
    model = fasterrcnn_resnet50_fpn(pretrained=False, num_classes=2)  
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    model.to(device)
    model.eval()
    return model

rcnn_model = load_model()
bands_model = load_bands_model()

# ฟังก์ชันใช้โมเดล RCNN เพื่อตรวจจับตัวต้านทาน
def detect_resistors_rcnn(image):
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    img_tensor = transform(image).to(device).unsqueeze(0)  # เพิ่ม batch dimension
    with torch.no_grad():
        predictions = rcnn_model(img_tensor)

    # คัดกรองผลลัพธ์โดยใช้ threshold ของความมั่นใจ
    threshold = 0.5
    boxes = [
        box.cpu().numpy().astype(int) for box, score in zip(predictions[0]["boxes"], predictions[0]["scores"]) if score > threshold
    ]

    resistor_images = []
    for box in boxes:
        x1, y1, x2, y2 = box
        cropped_resistor = image[y1:y2, x1:x2]  # ตัดตัวต้านทานออกจากภาพ
        resistor_images.append(cropped_resistor)

    return resistor_images

def process_image(file_path):
    try:
        original_image = cv2.imread(file_path)
        if original_image is None:
            return [{"error": "Invalid image file"}]
        
        # ใช้ RCNN แยกตัวต้านทาน
        extracted_resistors = detect_resistors_rcnn(original_image)

        results = []
        for resistor in extracted_resistors:
            
            color_bands, annotated_image,image_ = find_resistor_bands(resistor)
            resistance_value = calculate_resistance(color_bands)

            _, encoded_resistor = cv2.imencode('.jpg', image_)
            _, encoded_annotated = cv2.imencode('.jpg', annotated_image)

            results.append({
                "bands": color_bands,
                "resistance": resistance_value,
                "resistor": base64.b64encode(encoded_resistor).decode('utf-8'),
                "annotated": base64.b64encode(encoded_annotated).decode('utf-8')
            })

        return results
    except Exception as e:
        return [{"error": str(e)}]
    
def filter_bands_by_column(band_candidates):
    # เรียงแถบสีจากซ้ายไปขวาตาม x_min
    band_candidates.sort(key=lambda b: b[0])

    filtered_bands = []
    while band_candidates:
        first_band = band_candidates.pop(0)
        x_min1, y_min1, width1, height1, color_name1, color_box1, score1 = first_band

        same_column = [first_band]

        # ตรวจสอบแถบสีที่เหลือว่าซ้อนกับแถบแรกมากกว่า 50% หรือไม่
        for band in band_candidates[:]:  # ใช้ copy ของ list เพื่อลบค่าได้
            x_min2, _, width2, _, _, _, _ = band
            x_max1 = x_min1 + width1
            x_max2 = x_min2 + width2

            # คำนวณอัตราการซ้อนกันของคอลัมน์
            overlap = max(0, min(x_max1, x_max2) - max(x_min1, x_min2))
            min_width = min(width1, width2)

            if overlap / min_width > 0.9:  # ซ้อนกันเกิน 50%
                same_column.append(band)
                band_candidates.remove(band)

        # เลือกแถบสีที่มีค่า score สูงสุดในคอลัมน์เดียวกัน
        best_band = max(same_column, key=lambda b: b[6])
        filtered_bands.append(best_band)

    return filtered_bands

def find_resistor_bands(resistor_info):
    hsv = cv2.cvtColor(resistor_info, cv2.COLOR_BGR2HSV)
    
    transform = T.Compose([T.ToTensor()])
    image_rgb = cv2.cvtColor(resistor_info, cv2.COLOR_BGR2RGB)
    img_tensor = transform(image_rgb).to(device).unsqueeze(0)

    with torch.no_grad():
        predictions = bands_model(img_tensor)

    pred_boxes = predictions[0]["boxes"].cpu().numpy()
    pred_scores = predictions[0]["scores"].cpu().numpy()
    threshold = 0.5

    band_candidates = []

    for i, score in enumerate(pred_scores):
        if score > threshold:
            x_min, y_min, x_max, y_max = map(int, pred_boxes[i])
            band_region = hsv[y_min:y_max, x_min:x_max]

            h, w, _ = band_region.shape
            reshaped = band_region.reshape((h * w, 3))
            reshaped_int = reshaped.astype(np.uint8)
            reshaped_tuples = [tuple(pix) for pix in reshaped_int]

            unique_colors, counts = np.unique(reshaped_tuples, axis=0, return_counts=True)
            dominant_color = unique_colors[np.argmax(counts)]

            matched = False
            for color in COLOUR_TABLE:
                lower_bound = np.array(color[0])
                upper_bound = np.array(color[1])
                
                if all(lower_bound <= dominant_color) and all(dominant_color <= upper_bound):
                    band_candidates.append((x_min, y_min, x_max - x_min, y_max - y_min, color[2], color[4], score))
                    matched = True
                    break
            
            if not matched:
                band_candidates.append((x_min, y_min, x_max - x_min, y_max - y_min, "Unknown Color", (128, 128, 128), score))

    # กรองแถบสีที่อยู่ตำแหน่งเดียวกัน
    band_candidates.sort(key=lambda b: (b[0], -b[6]))
    filtered_bands = filter_bands_by_column(band_candidates)

    #แก้แถบลำดับที่ 4 ถ้าอยู่ขวาสุด ให้เป็น GOLD
    if len(filtered_bands) >= 4:
        # หาตำแหน่ง x ของแถบลำดับที่ 4 และหาค่า x สูงสุด
        x_of_4th_band = filtered_bands[3][0]
        x_rightmost = max(filtered_bands, key=lambda b: b[0])[0]

        # ถ้าแถบลำดับที่ 4 เป็นแถบที่อยู่ขวาสุด
        if x_of_4th_band == x_rightmost:
            # หาและแทนที่แถบนั้นด้วยสี GOLD
            for i in range(len(filtered_bands)):
                if filtered_bands[i][0] == x_of_4th_band:
                    for color in COLOUR_TABLE:
                        if color[2].upper() == "GOLD":
                            gold_band = (
                                filtered_bands[i][0],  # x
                                filtered_bands[i][1],  # y
                                filtered_bands[i][2],  # w
                                filtered_bands[i][3],  # h
                                "GOLD",                # color name
                                color[4],              # color BGR
                                filtered_bands[i][6]   # score
                            )
                            filtered_bands[i] = gold_band
                            break
                    break

    bands = [b[4] for b in filtered_bands]

    output_image = hsv
    output_image2 = resistor_info.copy()
    for x, y, w, h, color_name, color_box, _ in filtered_bands:
        cv2.rectangle(output_image, (x, y), (x + w, y + h), color_box, 2)
        cv2.putText(output_image, color_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)
        
    for x, y, w, h, color_name, color_box, _ in filtered_bands:
        cv2.rectangle(output_image2, (x, y), (x + w, y + h), color_box, 2)
        cv2.putText(output_image2, color_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_box, 1)

    return bands, output_image, output_image2

def calculate_resistance(color_bands):
    try:
        # ใช้เฉพาะ 3 แถบแรกสำหรับคำนวณค่า
        color_bands_calc = color_bands[:3]

        # คำนวณค่าหลักสำคัญจากแถบสีที่ 1 และ 2
        significant_figures = "".join(
            str(next(color[3] for color in COLOUR_TABLE if color[2].upper() == band.upper()))
            for band in color_bands_calc[:2]
        )

        # คำนวณตัวคูณจากแถบที่ 3
        multiplier = next(
            10 ** color[3] for color in COLOUR_TABLE if color[2].upper() == color_bands_calc[2].upper()
        )

        # คำนวณค่าความต้านทาน
        resistance_value = int(significant_figures) * multiplier

        # แปลงค่าให้อยู่ในหน่วยที่เหมาะสม
        if resistance_value >= 1_000_000:
            resistance_formatted = f"{int(resistance_value / 1_000_000)} MΩ" if resistance_value % 1_000_000 == 0 else f"{resistance_value / 1_000_000:.2f} MΩ"
        elif resistance_value >= 1_000:
            resistance_formatted = f"{int(resistance_value / 1_000)} KΩ" if resistance_value % 1_000 == 0 else f"{resistance_value / 1_000:.2f} KΩ"
        else:
            resistance_formatted = f"{resistance_value} Ω"

        # เพิ่มค่าความคลาดเคลื่อน ±5%
        resistance_with_tolerance = f"{resistance_formatted} ±5%"

        return resistance_with_tolerance

    except Exception as e:
        return "Unable to calculate resistance"


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        results = process_image(file_path)
        return jsonify(results)
    
    return render_template('upload.html')

@app.route('/selectcolor')
def selectcolor():
    return render_template('selectcolor.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
    # app.run(debug=True)
