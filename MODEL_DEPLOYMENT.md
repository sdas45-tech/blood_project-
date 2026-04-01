# 🚀 Model Deployment Guide for Vercel

## Problem
Your trained Keras model (`vgg16_best.keras`) is **NOT in GitHub** because it's in `.gitignore`. When Vercel deploys, it doesn't have the model file, so predictions fail.

## ✅ Solution: Cloud Storage + Auto-Download

### Step 1: Upload Your Model to Cloud Storage

**Choose ONE of these options:**

#### **Option A: Google Drive (Easiest)**
1. Go to [Google Drive](https://drive.google.com)
2. Upload `backend/models/vgg16_best.keras`
3. Right-click → "Share" → Make it "Anyone with the link can view"
4. Copy the **share link**, example: `https://drive.google.com/file/d/1ABC123xyz/view`
5. Convert to direct download URL:
   ```
   https://drive.google.com/uc?export=download&id=1ABC123xyz
   ```

#### **Option B: GitHub Releases**
1. Go to your GitHub repository
2. Click "Releases" → "Create a new release"
3. Upload `vgg16_best.keras` as an asset
4. Use the direct download URL (copy link from assets)

#### **Option C: AWS S3 (Professional)**
1. Upload to S3 bucket
2. Make it public or generate pre-signed URL
3. Use the S3 URL

---

### Step 2: Update the Download Script

Edit `backend/download_model.py` and replace the placeholder URLs:

```python
# ⚠️ IMPORTANT: Replace these with your actual cloud storage URLs
MODEL_URL = "https://your-cloud-storage.com/vgg16_best.keras"  # ← REPLACE THIS
CLASS_NAMES_URL = "https://your-cloud-storage.com/class_names.json"  # ← REPLACE THIS
```

**Example with Google Drive:**
```python
MODEL_URL = "https://drive.google.com/uc?export=download&id=YOUR_FILE_ID_HERE"
CLASS_NAMES_URL = "https://drive.google.com/uc?export=download&id=YOUR_CLASS_NAMES_ID"
```

---

### Step 3: Test Locally

```bash
# Test the download script
cd backend
python download_model.py
```

You should see:
```
✅ Model found: ...models/vgg16_best.keras
✅ Class names found: ...models/class_names.json
```

---

### Step 4: Commit & Push to GitHub

```bash
git add backend/download_model.py vercel.json
git commit -m "Add model auto-downloader for Vercel deployment"
git push origin main
```

---

### Step 5: Verify Vercel Deployment

1. Go to [Vercel Dashboard](https://vercel.com)
2. Check your deployment logs
3. Look for output from `download_model.py`:
   ```
   ✅ Model downloaded to backend/models/vgg16_best.keras
   ```
4. Test your prediction endpoint!

---

## 🔄 Alternative: Use Git LFS (Advanced)

If you prefer keeping everything in Git (recommended for production):

```bash
# Install Git LFS
git lfs install

# Track .keras files with LFS
git lfs track "*.keras"

# Add and commit
git add .gitattributes backend/models/vgg16_best.keras
git commit -m "Add model with Git LFS"
git push
```

Then remove the model download code and update `vercel.json` to its original state.

---

## ⚠️ Troubleshooting

**If predictions still don't work on Vercel:**

1. Check Vercel deployment logs for download errors
2. Verify your cloud storage URL is accessible
3. Ensure `class_names.json` is also downloaded
4. Test locally first with the download script

---

## 📝 Summary

| Component | Location | Status |
|-----------|----------|--------|
| Model file | Cloud storage | ✅ Uploaded |
| Download script | `backend/download_model.py` | ✅ Created |
| Vercel config | `vercel.json` | ✅ Updated |
| Local testing | Run `python download_model.py` | ⏳ Do this next |

