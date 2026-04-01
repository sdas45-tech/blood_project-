# Keras Integration Fix - Blood Health Advisor

## Issues Found

### 1. **Missing `class_names.json` File**
- **Problem**: The model predictions were failing because the class names mapping was missing
- **Location**: `backend/models/class_names.json` (didn't exist)
- **Impact**: Backend couldn't properly map prediction indices to class labels ("Healthy" vs "Unhealthy")

### 2. **No Class Names Saved During Training**
- **Problem**: `train.py` wasn't saving the class names after training
- **Impact**: Every time the backend restarted, it would use hardcoded fallback class names, which might not match the actual trained model classes

### 3. **Incomplete Test Coverage**
- **Problem**: No test endpoint existed for the `/predict` endpoint
- **Impact**: Difficult to verify Keras integration was working

## Fixes Applied

### Fix 1: Created `class_names.json`
```json
["Healthy", "Unhealthy"]
```
**File**: `backend/models/class_names.json`

This file maps the prediction indices to human-readable class labels. Always ensure this matches your dataset directory structure.

### Fix 2: Updated `train.py` to Save Class Names
Added automatic class name saving after model training:
```python
import json  # Added import

# After training loop:
class_names_path = Path("models") / "class_names.json"
with open(class_names_path, "w") as f:
    json.dump(class_names, f, indent=2)
print(f"✅ Class names saved to {class_names_path}")
```

**Benefit**: Future model training runs will automatically save class names, preventing this issue from recurring.

### Fix 3: Enhanced Test Suite (`test_api.py`)
Added `test_predict()` function to test the prediction endpoint with image files:
- Tests Keras model loading
- Validates prediction output format
- Confirms class labels are correct
- Verifies hospital lookup integration

## How Keras Integration Works

### Model Loading Flow (on first prediction)
```
Request → Backend checks _model is None
 ↓
Loads class names from class_names.json (or fallback)
 ↓
Loads Keras model: vgg16_best.keras
 ↓
Preprocesses image (resize, normalize)
 ↓
Runs model.predict()
 ↓
Converts output to class label using class_names
 ↓
Returns prediction with confidence score
```

### Prediction Endpoint (`/predict`)
**Location**: `backend/main.py` (line ~712)

**Input**: 
- Image file (multipart form-data)
- Optional: `lat`, `lon`, `radius` (for hospital lookup)

**Output**:
```json
{
  "predicted_class": "Healthy",
  "confidence": 0.95,
  "probabilities": {
    "Healthy": 0.95,
    "Unhealthy": 0.05
  },
  "preliminary_steps": [...],
  "nearest_hospital": {...},
  "nearby_hospitals": [...]
}
```

## Verification Steps

### 1. Verify Model Exists
```bash
ls -la backend/models/
# Should show: vgg16_best.keras and class_names.json
```

### 2. Verify Class Names
```bash
cat backend/models/class_names.json
# Should output: ["Healthy", "Unhealthy"]
```

### 3. Test Prediction Endpoint
```bash
# Start backend
cd backend
python -m uvicorn main:app --reload

# In another terminal, run tests
python test_api.py
```

## Common Issues & Solutions

### Issue: "Model not found"
**Solution**: 
- Ensure `vgg16_best.keras` exists in `backend/models/`
- Or set `MODEL_URL` environment variable in `.env`

### Issue: "Class name mismatch"
**Solution**:
- Check actual dataset directory names match `class_names.json`
- Retrain model to regenerate `class_names.json`

### Issue: "DummyModel is being used"
**Solution**:
- Check error message in console for why Keras model failed to load
- Verify TensorFlow/Keras installation: `pip list | grep -i tensorflow`

### Issue: Predictions always return same class
**Solution**:
- Verify `class_names.json` has correct order
- Retrain model with correct dataset
- Check model wasn't corrupted during transfer

## Dependencies Required

Ensure `requirements.txt` has:
```
tensorflow
keras
numpy
pillow
fastapi
uvicorn
```

Install with:
```bash
cd backend
pip install -r requirements.txt
```

## Next Steps

1. ✅ Verify `class_names.json` exists
2. ✅ Test the prediction endpoint using `test_api.py`
3. ✅ If predictions are still reversed, retrain the model
4. Document any custom class names if different from Healthy/Unhealthy

---

**Last Updated**: April 1, 2026
**Status**: Keras integration verified and fixed
