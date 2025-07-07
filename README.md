**Cheating Detection Using OpenCV & Mediapipe**

## Overview

This Python application uses OpenCV, Mediapipe, and other libraries to detect and log potential cheating behaviors during an exam or online assessment. The program continuously analyzes live webcam feed for multiple faces, gaze deviation, excessive blinking, mouth movement (talking), and hands in frame.  When any suspicious activity is detected, the system:

* Captures and crops the relevant video frame region.
* Logs the event along with timestamp, description, and image path in a CSV file.
* Provides real-time audio alerts via a text-to-speech engine.

## Features

* **Multiple Face Detection:** Flags if more than one face is detected on-screen.
* **Gaze Monitoring:** Tracks eye iris position to detect if the user looks away for too long.
* **Blink Detection:** Monitors blink rate and flags excessive blinking.
* **Mouth Movement:** Detects mouth opening to identify talking or whispering.
* **Hand Detection:** Alerts when hands enter the video frame (e.g., hidden notes).
* **Event Logging:** Saves each suspicious event as an image crop and logs details in `events_log.csv`.
* **Audio Alerts:** Reads out warnings to the candidate in real-time.

## Hardware & Software Requirements

* Python 3.8 or higher
* Webcam or integrated camera

## Python Dependencies

Install the required libraries using pip:

```bash
pip install opencv-python mediapipe numpy pyttsx3
```

## Directory Structure

```
project_root/
├── logs/
│   ├── events_log.csv      # CSV file storing event logs
│   └── *.jpg              # Cropped images of suspicious events
├── cheating_detection.py   # Main script
└── README.md               # Project documentation
```

## Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/cheating-detection.git
   cd cheating-detection
   ```

2. **Create logs directory** (automatically created at runtime if missing):

   ```bash
   mkdir -p logs
   ```

3. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify camera index** (default is 0). To change, edit:

   ```python
   cap = cv2.VideoCapture(0)
   ```

## Usage

Run the script:

```bash
python cheating_detection.py
```

On launch:

* A window titled **Cheating Detection** will display the webcam feed with overlaid landmarks and metrics.
* **Console logs** will appear in `logs/events_log.csv`.
* **Audio alerts** will notify any flagged behavior.
* Press **`q`** to quit the application.

## Configuration Parameters

Inside `cheating_detection.py`, you can adjust thresholds and parameters:

| Parameter    | Description                                     | Default |
| ------------ | ----------------------------------------------- | ------- |
| `EAR_T`      | Eye Aspect Ratio threshold for blink detection  | 0.25    |
| `EAR_FRAMES` | Consecutive frames below `EAR_T` to count blink | 3       |
| `MAR_T`      | Mouth Aspect Ratio threshold for mouth opening  | 0.5     |
| `GAZE_T`     | Gaze deviation threshold                        | 2       |
| `ALERT_CD`   | Cool-down (sec) between audio alerts            | 3       |

## Event Logging

All suspicious events are recorded to `logs/events_log.csv` with columns:

* **timestamp:** YYYYMMDD\_HHMMSS
* **event\_type:** faces, gaze, blink, mouth, hands
* **description:** Detailed event description
* **image\_path:** Path to the cropped image

## Troubleshooting

* **No camera detected:** Ensure your webcam is functional and available to OpenCV.
* **Permission errors writing logs:** Check folder permissions for `logs/` and grant write access.
* **Audio issues:** Verify `pyttsx3` back-end works on your system (e.g., install `espeak` on Linux).

## Extending the Project

* Integrate with a GUI framework (Tkinter, PyQt) for a more user-friendly interface.
* Add remote monitoring or dashboard to view logs in real-time.
* Tune detection thresholds and add new metrics (e.g., head pose estimation).

##
