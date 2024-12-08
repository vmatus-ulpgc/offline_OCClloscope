
# Signal Extraction for SCCP-PAM in Optical Camera Communication

This document provides a focused explanation of the **signal extraction process** in the SCCP-PAM-based OCC system. It includes key code snippets and their explanation to help you understand the processing steps.

---

## **1. Setup and Initialization**

Before processing, the program loads the video file and initializes the required parameters:

```python
import numpy as np
import cv2
import collections
import matplotlib.pyplot as plt

# Video file and parameters
filename, fps, x, y = "20241206_40m\20241206_40m_ag10.0_ss500.h264", 100, 322, 290
path = "C:\Users\Y6082772\Desktop\2025iet_oe-occpam"
cap = cv2.VideoCapture(os.path.join(path, filename))

# Define the Region of Interest (ROI) around the transmitter
x1, y1 = x - 10, y - 10
x2, y2 = x + 10, y + 10
```

- **ROI**: The `x, y` coordinates pinpoint the transmitter's center. A 20x20 pixel window is extracted for processing.
- **Video Capture**: `cv2.VideoCapture` loads the `.h264` video file for frame-by-frame analysis.

---

## **2. Signal Buffer Initialization**

To store the extracted signal values from each frame:

```python
buffer_size = 512
signal_buff = collections.deque(maxlen=buffer_size)
signal_buff.extend(np.zeros(buffer_size))  # Initialize with zeros
```

- **Purpose**: The `deque` acts as a circular buffer to hold recent signal intensity values.
- **Size**: The buffer length (`512`) determines the number of frames retained for analysis and plotting.

---

## **3. Frame Processing**

Each frame is processed to extract intensity values:

### **3.1 Cropping the Frame to ROI**

```python
good_frame, frame = cap.read()
if good_frame:
    frame = frame[y1:y2, x1:x2, :]  # Crop the frame to the ROI
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
```

- **Cropping**: Focuses only on the small window containing the LED signal.
- **Grayscale Conversion**: Reduces complexity by analyzing intensity values only.

### **3.2 Pixel Intensity Calculation**

```python
px_val = frame_gray[y, x]  # Intensity at the central pixel
nine_px_val = np.mean([
    frame_gray[y-1, x-1], frame_gray[y-1, x], frame_gray[y-1, x+1],
    frame_gray[y, x-1], frame_gray[y, x], frame_gray[y, x+1],
    frame_gray[y+1, x-1], frame_gray[y+1, x], frame_gray[y+1, x+1]
])
val = np.power(nine_px_val / 255.0, gamma)  # Normalize and apply gamma correction
signal_buff.append(val)
```

- **Intensity Extraction**:
  - `px_val`: Single-pixel intensity at the LED location.
  - `nine_px_val`: Average intensity over a 3x3 pixel grid for robustness.
- **Gamma Correction**: Compensates for non-linear intensity variations.

---

## **4. Live Plotting**

The extracted signal is visualized in real-time:

```python
plt.ion()  # Enable interactive plotting
fig, ax = plt.subplots(figsize=(9, 3))
signal_y = np.array(signal_buff)
line1, = ax.plot(np.linspace(0, buffer_size, buffer_size), signal_y, '.-', linewidth=0.8)

# Update the plot with new signal values
line1.set_ydata(signal_y)
fig.canvas.draw()
fig.canvas.flush_events()
```

- **Purpose**: Displays the variation in received signal intensity over frames.
- **Interactive Mode**: Updates the plot dynamically as new frames are processed.

---

## **5. Threshold Calculation and Symbol Decoding**

To identify symbols from the signal levels:

```python
from sklearn.mixture import GaussianMixture as GM

# Fit a Gaussian Mixture Model to classify intensity levels
pam_levels = 8  # Number of PAM levels
gmm = GM(pam_levels)
gmm.fit(signal_y.reshape(-1, 1))
means = np.sort(gmm.means_.flatten())

# Calculate thresholds between levels
thresholds = [(means[i] + means[i + 1]) / 2 for i in range(len(means) - 1)]

# Overlay thresholds on the plot
for threshold in thresholds:
    ax.axhline(threshold, color='red', linestyle='--', linewidth=0.8)
```

- **Gaussian Mixture Model**: Clusters intensity values into `pam_levels` (e.g., 8 levels for SCCP-PAM).
- **Thresholds**: Midpoints between adjacent clusters, used for symbol decoding.

---

## **6. Display the Region of Interest (Optional)**

For debugging, the PSF of the LED can be visualized:

```python
psf_span = 3
psf = frame_gray[y - psf_span:y + psf_span + 1, x - psf_span:x + psf_span + 1]
plt.imshow(psf, cmap='gray', extent=[-psf_span, psf_span, -psf_span, psf_span])
plt.colorbar()
plt.title("Point Spread Function")
plt.show()
```

- **Purpose**: Verifies the LED projection and signal quality.

---

## **7. Histogram of Signal Levels**

To validate symbol distribution:

```python
plt.hist(signal_y, bins=100)
plt.xlabel("Signal Intensity")
plt.ylabel("Counts")
for threshold in thresholds:
    plt.axvline(threshold, color='red', linestyle='--', linewidth=0.8)
plt.show()
```

- **Histogram**: Visualizes the distribution of received signal intensities.
- **Overlay Thresholds**: Highlights boundaries for symbol detection.

---

## **Complete Signal Extraction Workflow**

```python
while True:
    good_frame, frame = cap.read()
    if not good_frame:
        break
    
    frame = frame[y1:y2, x1:x2, :]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    nine_px_val = np.mean([frame_gray[y-1, x-1], frame_gray[y-1, x], frame_gray[y-1, x+1],
                           frame_gray[y, x-1], frame_gray[y, x], frame_gray[y, x+1],
                           frame_gray[y+1, x-1], frame_gray[y+1, x], frame_gray[y+1, x+1]])
    val = np.power(nine_px_val / 255.0, gamma)
    signal_buff.append(val)
    
    # Update live plot
    signal_y = np.array(signal_buff)
    line1.set_ydata(signal_y)
    fig.canvas.draw()
    fig.canvas.flush_events()

cap.release()
```

---

## Key Outputs:
1. **Real-Time Signal Plot**: Shows intensity variation over frames.
2. **Histogram with Thresholds**: Confirms effective symbol separation.
3. **PSF Visualization**: Verifies ROI alignment and signal quality.

---

Let me know if you need further clarifications or enhancements!
