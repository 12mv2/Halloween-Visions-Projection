# Maintenance Guide

**Version:** vB10-final
**Last Updated:** 2025-10-19

---

## Updating the Detection Model

### Model Requirements
- Format: YOLO classification model (.pt file)
- Framework: Ultralytics 8.0.196+ compatible
- Input: 224×224 RGB images
- Output: Classification probabilities
- Classes: Binary (hand/not_hand) or multi-class

### Replace Model

```bash
cd ~/Documents/MLVisionsProjects/Halloween-Visions-Projection
source venv/bin/activate

# Backup current model
cp Colin1.pt Colin1_backup_$(date +%Y%m%d).pt

# Copy new model
cp /path/to/new_model.pt Colin1.pt

# Test
python3 test_gpu_validation.py
```

### Validate Performance
Check for:
- Model loads without errors
- GPU inference works (cuda:0)
- FPS ≥15 (preferably ≥25)
- NaN count = 0
- Classes detected correctly

### Update Service (if using systemd)

```bash
sudo nano /etc/systemd/system/halloween.service

# Update model path if needed
ExecStart=.../python3 simple_projection.py --model new_model.pt ...

sudo systemctl daemon-reload
sudo systemctl restart halloween.service
```

---

##Updating Videos

### Video Requirements
- Format: MP4 (H.264 recommended)
- Resolution: Any (scales automatically), 1080p recommended
- Duration: 3-10 seconds for seamless loops
- Location: videos/ directory

### Replace Videos

```bash
cd ~/Documents/MLVisionsProjects/Halloween-Visions-Projection/videos

# Backup current videos
cp sleeping_face.mp4 sleeping_face_backup.mp4
cp angry_face.mp4 angry_face_backup.mp4

# Copy new videos
cp /path/to/new_idle.mp4 sleeping_face.mp4
cp /path/to/new_scare.mp4 angry_face.mp4

# Test
cd ..
python3 simple_projection.py --source 0 --conf 0.5
```

### Custom Video Names

```bash
python3 simple_projection.py \
    --video-sleep custom_idle.mp4 \
    --video-scare custom_scare.mp4 \
    --source 0
```

---

## Adjusting Parameters

### Confidence Threshold

Controls sensitivity of hand detection:

```bash
# More sensitive (detects more easily)
python3 simple_projection.py --conf 0.3

# Less sensitive (requires clearer detection)
python3 simple_projection.py --conf 0.9

# Default (balanced)
python3 simple_projection.py --conf 0.7
```

### Scare Duration

Edit simple_projection.py:

```python
# Line ~52
self.scare_duration = 2.0  # Change to desired seconds
```

### Camera Selection

```bash
# Default camera
python3 simple_projection.py --source 0

# Alternate camera
python3 simple_projection.py --source 1
```

---

## Troubleshooting

### Camera Not Working

```bash
# Try alternate camera
python3 simple_projection.py --source 1

# Check permissions (macOS)
# System Preferences → Privacy & Security → Camera

# List available cameras
ls /dev/video*  # Linux
```

### Low Detection Accuracy

1. Improve lighting conditions
2. Lower confidence threshold: --conf 0.3
3. Retrain model with better dataset
4. Check camera focus and positioning

### Poor Performance

```bash
# Run benchmark
python3 test_gpu_validation.py

# Check for:
# - FPS < 15: Model too complex for hardware
# - High NaN rate: GPU issues
# - Errors: Check logs
```

### Videos Not Playing

```bash
# Verify files exist
ls -lh videos/*.mp4

# Check format compatibility
ffmpeg -i videos/sleeping_face.mp4  # Should show H.264

# Re-encode if needed
ffmpeg -i input.mp4 -c:v libx264 -c:a copy output.mp4
```

### System Service Issues

```bash
# Check status
sudo systemctl status halloween.service

# View logs
sudo journalctl -u halloween.service -f

# Restart
sudo systemctl restart halloween.service

# Stop
sudo systemctl stop halloween.service
```

---

## Performance Tuning

### Monitor Performance

```bash
# GPU stats (Jetson)
tegrastats

# System resources
htop

# Application logs
tail -f inference_diagnostics.log
```

### Optimize for Low-End Hardware

1. Lower camera resolution
2. Reduce model complexity (use nano variant)
3. Increase confidence threshold
4. Consider TensorRT optimization (see PERFORMANCE_ENHANCEMENT_RECOMMENDATIONS.md)

### Optimize for Multi-Camera

1. Implement TensorRT FP16 (see docs/PERFORMANCE_ENHANCEMENT_RECOMMENDATIONS.md)
2. Use batched inference
3. Consider threading for camera capture

---

## Jetson Xavier NX Specific

### Service Management

```bash
# Enable auto-start
sudo systemctl enable halloween.service

# Disable auto-start
sudo systemctl disable halloween.service

# Check auto-start status
systemctl is-enabled halloween.service
```

### Power Modes

```bash
# Maximum performance
sudo nvpmodel -m 0

# Balanced (default)
sudo nvpmodel -m 2

# Check current mode
sudo nvpmodel -q
```

### Thermal Monitoring

```bash
# Real-time stats
tegrastats

# Check temperatures
cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

---

## Backup and Restore

### Backup System

```bash
# Backup models
cp Colin1.pt ~/backups/Colin1_$(date +%Y%m%d).pt

# Backup videos
cp videos/*.mp4 ~/backups/videos/

# Backup configuration
cp simple_projection.py ~/backups/
cp /etc/systemd/system/halloween.service ~/backups/
```

### Restore System

```bash
# Restore model
cp ~/backups/Colin1_YYYYMMDD.pt Colin1.pt

# Restore videos
cp ~/backups/videos/*.mp4 videos/

# Restore service
sudo cp ~/backups/halloween.service /etc/systemd/system/
sudo systemctl daemon-reload
```

---

## Log Management

### View Logs

```bash
# Application logs
cat inference_diagnostics.log

# System service logs
sudo journalctl -u halloween.service --since "1 hour ago"

# All logs from today
sudo journalctl -u halloween.service --since today
```

### Clear Logs

```bash
# Clear application log
> inference_diagnostics.log

# Clear systemd logs
sudo journalctl --vacuum-time=7d  # Keep last 7 days
```

---

## Updating Dependencies

### Update Python Packages

```bash
source venv/bin/activate

# Update specific package
pip install --upgrade ultralytics

# Update all packages
pip install --upgrade -r requirements.txt
```

**Warning:** Test thoroughly after updates. Major version changes may break compatibility.

---

## Common Maintenance Tasks

**Weekly:**
- Check system logs for errors
- Monitor performance metrics
- Verify camera and display functionality

**Monthly:**
- Update video content (keep fresh)
- Review and archive logs
- Check for software updates

**Seasonal:**
- Backup models and configuration
- Clean dust from camera and projector
- Update documentation

---

For performance optimization recommendations, see [docs/PERFORMANCE_ENHANCEMENT_RECOMMENDATIONS.md](../docs/PERFORMANCE_ENHANCEMENT_RECOMMENDATIONS.md)
