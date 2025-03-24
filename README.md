# Robot Vision System

A real-time video processing application that receives video frames over UDP and performs object detection to identify red and blue balls, as well as green robots.

## Features

- UDP-based video frame reception and reconstruction
- Multi-threaded architecture for non-blocking video processing
- HSV color space filtering for robust object detection
- Detects and visualizes:
  - Red balls (using dual threshold to handle hue wrapping)
  - Blue balls
  - Green robots (square detection)

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- NumPy
- Socket networking support

## How It Works

1. The application listens for UDP packets on port 8080
2. Incoming video frames are reconstructed from multiple packets
3. HSV color space filtering identifies objects based on color profiles
4. Contour detection and shape analysis differentiate between balls and robots
5. The processed frame is displayed with visual markers and labels

## Usage

Run the application:

```bash
python3 receiver.py
