# SAM Hold Labeling - Quick Start Guide

## 🚀 Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install segment-anything torch torchvision
```

### 2. Download SAM Model
```bash
# Download SAM ViT-B (recommended for prototyping)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/
```

### 3. Start Server
```bash
cd webapp
uvicorn main:app --reload --port 8000
```

### 4. Open Workflow
Visit: **http://localhost:8000/workflow**

---

## 📹 Workflow Steps (Unified)

### Step 1: Upload Video & Start Segmentation (1-2 min)
1. Click "Choose File" and select a climbing video.
2. (Optional) Pick Hold Color & Route Difficulty from the dropdowns.
3. A preview of the first frame appears.
4. Click "🔍 Start Segmentation" — this launches SAM on the first frame for hold labeling.

### Step 2: Label Holds with SAM (10-20 min)
1. In the SAM labeler, click on a segment and choose the hold type from the dropdown.
2. Repeat for all holds you want to keep. Skip non-hold segments.
3. Save/finish so the labels are available back in the workflow page.

**Keyboard Shortcuts:**
- `A` - Previous frame
- `D` - Next frame
- `1-6` - Quick assign hold type (1=crimp, 2=sloper, etc.)
- `ESC` - Deselect segment

**Tips:**
- Label at least **30 holds per type** for good results
- Don't worry about labeling every segment - skip non-holds
- Click "Save Progress" regularly
- Export when done labeling

### Step 3: Key Frame Selection (Required)
1. Back in the workflow page, use the frame selector to choose key frames (or run the selector if already trained).
2. Save to the training pool when ready.

### Step 4: Train YOLO (30-60 min)
1. Click "🚀 Train YOLO (All Pool)" to train on all labeled holds in the pool.
2. Monitor progress in "Training Jobs". When status is COMPLETED, the model is ready.

### Step 5: Upload to GCS (30 sec)
1. When training shows COMPLETED, click "Upload Model".
2. Copy the GCS URI for deployment.

---

## 🎯 Expected Results

### Frame Extraction
- Input: 2-minute climbing video
- Output: **15-25 frames** (climbing-only, diverse poses)
- Location: `data/frames/video_name/`

### SAM Segmentation
- Per frame: **20-50 segments** auto-detected
- Includes: holds, person, wall features, background objects
- You label only the holds

### Labeling Session
- Target: **30-100 labeled holds** total
- Time: **15-30 minutes** for 20 frames
- Output: JSON session file + YOLO dataset

### YOLO Training
- Dataset split: 70% train / 15% val / 15% test
- Training time: **30-60 minutes** (100 epochs, CPU)
- Training time: **5-10 minutes** (100 epochs, GPU)
- Output: `runs/hold_type/train/weights/best.pt`

### Model Performance (after sufficient data)
- mAP50: **0.7-0.9** (good hold detection)
- Inference: **30-60 FPS** on GPU
- Inference: **5-10 FPS** on CPU

---

## ⚡ Common Workflows

### Workflow A: Single Video Training
```
1. Upload video → Extract frames (motion+pose)
2. Create session → Label 20 frames (30 min)
3. Export → Train YOLO (50 epochs)
4. Test model → If good, upload to GCS
```

### Workflow B: Multi-Video Dataset
```
1. Video 1 → Extract → Label (Session 1)
2. Video 2 → Extract → Label (Session 2)
3. Video 3 → Extract → Label (Session 3)
4. Export all sessions → Combined dataset
5. Train YOLO (100 epochs) → Upload
```

### Workflow C: Incremental Training
```
1. Start with base model (yolov8n.pt)
2. Label 20 frames → Train 50 epochs → Test
3. Add 20 more frames → Train 50 more epochs
4. Repeat until performance satisfactory
5. Final upload to GCS
```

---

## 🐛 Troubleshooting

### SAM Not Working
```bash
# Check if SAM checkpoint exists
ls models/sam_vit_b_01ec64.pth

# If missing, download again
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -P models/
```

### Training Fails with OOM
```python
# Reduce batch size in Step 3
batch_size = 8  # or even 4

# Or use smaller image size
imgsz = 416  # instead of 640
```

### Frames Not Extracting
```bash
# Check video file format
ffmpeg -i your_video.mp4

# Convert if needed
ffmpeg -i input.mov -c:v libx264 output.mp4
```

### Labeling UI Not Loading
```bash
# Check session was created
curl http://localhost:8000/api/labeling/sessions

# If empty, SAM segmentation might be in progress
# Wait 1-2 minutes and refresh
```

---

## 📊 Progress Checklist

- [ ] Server running at http://localhost:8000
- [ ] SAM checkpoint downloaded
- [ ] Video uploaded and frames extracted
- [ ] Labeling session created
- [ ] At least 30 holds labeled (10+ per type)
- [ ] Dataset exported to YOLO format
- [ ] Training started (job ID obtained)
- [ ] Training completed (check status)
- [ ] Model uploaded to GCS

---

## 🎓 Best Practices

### Labeling Quality
✅ **DO:**
- Label clear, visible holds
- Include variety of hold types
- Label from different wall angles
- Include both hands and feet holds

❌ **DON'T:**
- Label blurry or occluded holds
- Label holds behind the climber
- Skip entire hold types
- Label too quickly without thinking

### Training Efficiency
✅ **DO:**
- Start with 50 epochs, then increase
- Monitor validation loss
- Use GPU if available
- Save model checkpoints

❌ **DON'T:**
- Train with <20 images
- Use excessive batch size (OOM)
- Stop training prematurely
- Forget to validate on test set

---

## 📞 Support

If you encounter issues:
1. Check `docs/SAM_LABELING_GUIDE.md` for detailed documentation
2. Review API logs in terminal
3. Check browser console for JavaScript errors
4. Verify all dependencies are installed

---

## 🎉 Success Indicators

You're on the right track if:
- ✅ Frame extraction completes in <1 minute
- ✅ SAM segments 20-50 objects per frame
- ✅ Labeling takes ~1-2 minutes per frame
- ✅ Training shows decreasing loss
- ✅ Validation mAP > 0.5 after 50 epochs
- ✅ Final model size ~6-25 MB

---

**Happy Labeling! 🧗‍♂️**
