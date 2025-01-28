# Automatic Number Plate Recognition (ANPR) System

This project implements an Automatic Number Plate Recognition (ANPR) system using YOLO for license plate detection and PaddleOCR for optical character recognition (OCR). The system processes video input to detect license plates and extract their alphanumeric content.

## Features
- Real-time license plate detection using YOLO.
- OCR to extract and validate alphanumeric text from detected license plates.
- Filters and cleans extracted text to ensure accuracy.
- Visualizes detected plates and recognized text on the video feed.

## Requirements
- Python 3.7+
- Libraries:
  - `ultralytics`
  - `cv2` (OpenCV)
  - `cvzone`
  - `math`
  - `re`
  - `os`
  - `paddleocr`
  - `numpy`

Install dependencies using:
```bash
pip install ultralytics opencv-python-headless cvzone paddleocr numpy
```

## How It Works
1. **License Plate Detection**: YOLO model detects license plates in the video frames.
2. **Text Extraction**: Detected regions are passed to PaddleOCR to recognize alphanumeric text.
3. **Text Cleaning**: Extracted text is validated to remove invalid characters and ensure proper formatting.
4. **Visualization**: Bounding boxes and recognized text are displayed on the video feed.

## File Structure
- `model_weight/license_plate_detector.pt`: Pre-trained YOLO model for license plate detection.
- `data/carLicence4.mp4`: Example input video.
- Main script: Contains the implementation for detection and OCR.

## Usage
1. Place the YOLO model weights in the `model_weight` directory.
2. Add your input video to the `data` directory.
3. Run the script:
   ```bash
   python main.py
   ```
4. Press `q` to quit the application.

## Customization
- Replace `model_weight/license_plate_detector.pt` with your own trained YOLO model for different regions or license plate formats.
- Modify the `ocr` function to adjust OCR confidence thresholds and text cleaning rules.

## Notes
- The system uses PaddleOCR in CPU mode by default. Modify the `use_gpu` parameter to enable GPU acceleration if available.
- Ensure that the input video resolution is adequate for license plate detection.

## License
This project is for educational and non-commercial purposes.

