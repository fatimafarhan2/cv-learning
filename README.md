# Computer Vision Learning Project

A comprehensive learning repository for OpenCV fundamentals and image processing techniques using Python.

## 📋 Overview

This project contains practical examples and implementations of core computer vision concepts including image manipulation, video processing, filtering, edge detection, and contour analysis.

## 📁 Project Structure

```
cv/
├── cv1.ipynb              # Main notebook with comprehensive CV examples
├── data/                  # Directory for sample images and videos
│   ├── bird.png
│   ├── dogs2.jpg
│   ├── birds.jpeg
│   ├── bear.jpg
│   ├── freelancer.jpg
│   ├── basketball.jpg
│   ├── whiteboard.jpg
│   ├── h2.jpg
│   ├── birdies.jpg
│   └── bird_out.png       # Output images
└── README.md              # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions


### 1. Clone or download this repository:

```bash
git clone https://github.com/fatimafarhan2/cv-learning.git
cd cv-learning
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
```

3. Activate the virtual environment:
   - **Windows:**
   ```bash
   .venv\Scripts\activate
   ```
   - **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. Install required packages:
```bash
pip install opencv-python numpy jupyter
```

## 📦 Dependencies

- **OpenCV** (4.13.0.90+) - Computer vision library
- **NumPy** (2.4.2+) - Numerical computing
- **Jupyter** - Interactive notebook environment

## 📚 Topics Covered

### 1. **Image Basics**
   - Reading images from files
   - Writing images to disk
   - Displaying images with OpenCV

### 2. **Video & Webcam Processing**
   - Reading video files frame by frame
   - Real-time webcam capture
   - Video playback and display

### 3. **Image Resizing**
   - Using `cv2.resize()` to adjust image dimensions
   - Preserving or modifying aspect ratios

### 4. **Colorspace Conversions**
   - BGR to RGB conversion
   - Grayscale conversion
   - HSV colorspace for color detection
   - Applications: `cv2.cvtColor()`

### 5. **Image Filtering & Blurring**
   - Averaging blur
   - Gaussian blur
   - Median blur
   - Bilateral filtering
   - Noise reduction techniques

### 6. **Thresholding**
   - **Global Thresholding:** Single threshold value for entire image
   - **Adaptive Thresholding:** Dynamic thresholding based on neighborhood
   - Binary image generation
   - Object separation from background

### 7. **Edge Detection**
   - Canny edge detection
   - Morphological operations (dilation, erosion)
   - Edge enhancement techniques

### 8. **Drawing Shapes**
   - Lines, rectangles, circles
   - Text annotation
   - Custom graphics overlay

### 9. **Contour Analysis**
   - Finding contours in binary images
   - Contour area calculation
   - Bounding rectangle extraction
   - Object detection and localization

## 🚀 Quick Start

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `cv1.ipynb` in your browser

3. Run cells sequentially to see each concept in action

## 📖 Usage Examples

### Reading and Displaying an Image
```python
import cv2
import os

image_path = os.path.join('.', 'data', 'bird.png')
img = cv2.imread(image_path)
cv2.imshow('image', img)
cv2.waitKey(0)
```

### Converting Colorspaces
```python
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
```

### Applying Filters
```python
img_blur = cv2.GaussianBlur(img, (7, 7), 0)
img_median = cv2.medianBlur(img, 7)
```

### Edge Detection
```python
edges = cv2.Canny(img, 100, 200)
dilated = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8))
```

## 💡 Key Concepts

### Image Representation
- Images are represented as NumPy arrays
- Shape format: `(height, width, channels)` for color images
- Channels in OpenCV: BGR (not RGB)

### Kernel Size & Parameters
- Larger kernel sizes produce stronger effects
- For blurring: kernel size must be odd numbers (3, 5, 7, etc.)
- For thresholding: experiment with threshold values and block sizes

### Performance Tips
- Resize images for faster processing
- Use appropriate colorspaces for specific tasks
- Apply morphological operations to improve results

## 🔧 Tips for Learning

1. **Experiment:** Change parameter values to see their effects
2. **Visualize:** Always display intermediate results
3. **Document:** Add comments explaining your code
4. **Practice:** Apply techniques to your own images
5. **Iterate:** Combine multiple techniques for better results

## 📝 Notes

- The wait time for webcam processing is slightly higher than video playback due to hardware interaction
- Global thresholding may not work well for uneven lighting; use adaptive thresholding in such cases
- Always convert images to grayscale before applying thresholding or edge detection
- Use HSV colorspace for robust color-based detection rather than RGB/BGR

## 🐛 Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Image not displaying | Ensure image path is correct and file exists |
| Black image output | Check colorspace conversion (BGR vs RGB) |
| Poor edge detection | Adjust Canny thresholds (lower/upper values) |
| Slow performance | Resize image or reduce processing operations |

## 📚 Resources

- [OpenCV Documentation](https://docs.opencv.org/)
- [NumPy Documentation](https://numpy.org/doc/)
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

## 🎓 Next Steps

- Implement feature detection (SIFT, SURF, ORB)
- Explore object detection (Cascade Classifiers, YOLO)
- Learn image segmentation techniques
- Study video tracking methods
- Implement deep learning models for vision tasks

## 📄 License

This project is for educational purposes.

## ✍️ Author

Computer Vision Learning Project

---

