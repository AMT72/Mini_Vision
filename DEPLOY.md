# MiniVision — Team Deploy Guide !

## What you need
- The code: this GitHub repo (UI branch)
- The model weights: download from [this link] → you get two files:
  - `epoch60.pt` (damage model)
  - `parts_best.pt` (parts detector)

---

## Deploy on Hugging Face Spaces (free)

### 1. Upload model weights to HF Hub (once)
```bash
pip install huggingface_hub
huggingface-cli login          # use your HF token

huggingface-cli repo create minivision-models --type model
huggingface-cli upload YOUR_HF_USERNAME/minivision-models epoch60.pt --repo-type model
huggingface-cli upload YOUR_HF_USERNAME/minivision-models parts_best.pt --repo-type model
```

### 2. Create the Space
1. Go to https://huggingface.co/new-space
2. Name: `minivision` | SDK: **Docker** | Visibility: Public

### 3. Push the code to the Space
```bash
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/minivision
cp -r /path/to/this/repo/* minivision/
cd minivision
git add . && git commit -m "deploy" && git push
```

### 4. Set environment variables
In the Space → **Settings → Variables and Secrets**, add:

| Key | Value |
|-----|-------|
| `HF_MODEL_REPO` | `YOUR_HF_USERNAME/minivision-models` |
| `PORT` | `7860` |

The app downloads the weights automatically on first startup.

### 5. Update the landing page link
In `index.html` and `index_en.html`, replace:
```
YOUR_HF_USERNAME-minivision.hf.space
```
with your actual Space URL (shown in the Space page after deploy).

Commit and push to GitHub.

---

## Run locally (for testing)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Put the two .pt files in:
#   models/CSK_Model/weights/epoch60.pt
#   models/Parts Detector (YOLO26)/weights/best.pt

python app.py
# → open http://localhost:5000
```
