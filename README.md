# 🚗 Automatic Vehicle Speed Monitoring System using YOLOv8

This project implements an end-to-end **vehicle speed detection system** using **YOLOv8 object detection** and **OpenCV**, built for use in traffic surveillance and monitoring over-speeding violations. Designed to work with video footage, it detects moving vehicles, tracks them across frames, and estimates their speed in km/h.

---

## 📌 Features

- 🎯 Real-time vehicle detection using YOLOv8 (Ultralytics)
- 📦 Integrated ByteTrack for object ID tracking
- ⚡ Speed estimation in km/h based on pixel displacement and time
- 📹 Annotated output video with live speed overlays
- ✅ Works fully in **Google Colab** or local Python environments

---

## 🧠 How It Works

1. **YOLOv8** detects vehicles frame-by-frame.
2. **ByteTrack** assigns unique IDs to each vehicle.
3. **Speed is estimated** using:
   - Pixel displacement between frames
   - Frame rate and real-world calibration distance

Speed (km/h) = `(pixel_distance / calibration_pixel_ref) * real_world_distance / time_elapsed * 3.6`

---


