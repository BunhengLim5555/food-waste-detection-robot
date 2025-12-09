# GitHub Upload Guide

Follow these steps to upload your Food Waste Detection Robot project to GitHub.

## Step 1: Create a New GitHub Repository

1. Go to [GitHub.com](https://github.com) and log in
2. Click the **"+"** icon in the top-right corner
3. Select **"New repository"**
4. Fill in the details:
   - **Repository name**: `food-waste-detection-robot`
   - **Description**: "Real-time food waste detection robot using SSD MobileNet V2 and Raspberry Pi 4"
   - **Visibility**: Choose **Public** or **Private**
   - **DO NOT** check "Initialize this repository with a README" (we already have one)
5. Click **"Create repository"**

## Step 2: Connect Your Local Repository to GitHub

Open Git Bash or terminal in your project directory and run:

```bash
cd "c:\Users\Bunheng Lim\Downloads\food_waste_ssd_project\food_waste_ssd\food_waste_ssd"

# Add your GitHub repository as remote
git remote add origin https://github.com/YOUR_USERNAME/food-waste-detection-robot.git

# Verify the remote was added
git remote -v
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 3: Push Your Code to GitHub

```bash
# Push to GitHub (main branch)
git push -u origin master
```

If you're using a newer version of Git that uses `main` instead of `master`:

```bash
# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

## Step 4: Verify Upload

1. Go to your GitHub repository: `https://github.com/YOUR_USERNAME/food-waste-detection-robot`
2. You should see all your files uploaded
3. The README.md will automatically display on the repository homepage

## Alternative: Upload via GitHub Desktop

If you prefer a GUI:

1. Download and install [GitHub Desktop](https://desktop.github.com/)
2. Open GitHub Desktop
3. Click **"Add"** > **"Add Existing Repository"**
4. Browse to: `C:\Users\Bunheng Lim\Downloads\food_waste_ssd_project\food_waste_ssd\food_waste_ssd`
5. Click **"Publish repository"**
6. Choose repository name and visibility
7. Click **"Publish"**

## Important Notes

### Files Included in Upload

✅ **Included:**
- All Python scripts
- TFLite model (6.5 MB)
- Dataset (TFRecord files)
- Label map
- Training config
- Documentation (README, TRAINING_GUIDE)
- Requirements files

❌ **Excluded (via .gitignore):**
- Virtual environments (.venv/)
- Training checkpoints (too large)
- Full TensorFlow models directory
- Temporary files
- Large CUDA/cuDNN installers

### Dataset Size

Your TFRecord files might be large. If GitHub rejects the push due to file size:

**Option 1: Use Git LFS (Large File Storage)**
```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "*.tfrecord"
git lfs track "*.tflite"

# Add .gitattributes
git add .gitattributes

# Commit and push
git commit -m "Add Git LFS tracking"
git push
```

**Option 2: Store Dataset Elsewhere**
- Upload dataset to Google Drive or Dropbox
- Add download link to README.md
- Remove TFRecords from git:
  ```bash
  git rm --cached data/*.tfrecord
  echo "data/*.tfrecord" >> .gitignore
  git commit -m "Remove large TFRecord files"
  ```

## Troubleshooting

### Error: "remote: error: File too large"

If you get this error:

```bash
# Remove large files from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch data/train.tfrecord data/test.tfrecord data/valid.tfrecord" \
  --prune-empty --tag-name-filter cat -- --all

# Force push
git push -u --force origin master
```

Then upload dataset separately to Google Drive.

### Error: "Authentication failed"

1. **Using HTTPS**: Generate a Personal Access Token
   - Go to GitHub Settings > Developer settings > Personal access tokens
   - Click "Generate new token (classic)"
   - Select scopes: `repo`, `workflow`
   - Copy the token
   - Use token as password when pushing

2. **Using SSH**: Set up SSH keys
   ```bash
   ssh-keygen -t ed25519 -C "your_email@example.com"
   cat ~/.ssh/id_ed25519.pub
   ```
   - Copy the SSH key
   - Add to GitHub: Settings > SSH and GPG keys > New SSH key
   - Change remote to SSH:
     ```bash
     git remote set-url origin git@github.com:YOUR_USERNAME/food-waste-detection-robot.git
     ```

### Error: "Repository not found"

Make sure:
1. You created the repository on GitHub
2. The repository name matches exactly
3. You replaced `YOUR_USERNAME` with your actual username

## After Upload

### 1. Add Topics/Tags

On your GitHub repository page:
- Click the gear icon next to "About"
- Add topics: `robotics`, `object-detection`, `tensorflow`, `raspberry-pi`, `computer-vision`, `food-waste`, `ssd-mobilenet`, `tflite`

### 2. Add License

Create a LICENSE file:
```bash
# Add MIT License (or your choice)
curl https://raw.githubusercontent.com/licenses/license-templates/master/templates/mit.txt > LICENSE
git add LICENSE
git commit -m "Add MIT License"
git push
```

### 3. Create Releases

Tag your first version:
```bash
git tag -a v1.0.0 -m "First release: Food Waste Detection Robot"
git push origin v1.0.0
```

### 4. Enable GitHub Pages (Optional)

To create a project website:
1. Go to repository Settings > Pages
2. Source: Deploy from branch `main` or `master`
3. Folder: `/ (root)`
4. Save
5. Your site will be at: `https://YOUR_USERNAME.github.io/food-waste-detection-robot`

## Sharing Your Project

Share your repository URL:
```
https://github.com/YOUR_USERNAME/food-waste-detection-robot
```

Add to your portfolio, resume, or LinkedIn!

## Future Updates

When you make changes:

```bash
# Stage changes
git add .

# Commit with message
git commit -m "Add new feature or fix bug"

# Push to GitHub
git push
```

## Need Help?

- [GitHub Documentation](https://docs.github.com/)
- [Git Basics](https://git-scm.com/book/en/v2/Getting-Started-Git-Basics)
- [Git LFS](https://git-lfs.github.com/)

---

**Good luck with your submission!** 🚀
