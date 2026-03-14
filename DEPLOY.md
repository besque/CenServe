# Push to GitHub & Make Download Link Work

Your repo: **https://github.com/besque/CenServe**

---

## 1. Push your code to GitHub

From the project root (`CenServe` folder):

```powershell
cd "c:\Users\gagan\OneDrive\Desktop\make a thing\censerve\CenServe"

git status
git add .
git commit -m "Add marketing site, run.py launcher, resource_path for PyInstaller"
git push origin main
```

If your branch is `master` instead of `main`:

```powershell
git push origin master
```

---

## 2. Build the Windows exe (so the download has something to attach)

Install PyInstaller and build from project root:

```powershell
cd "c:\Users\gagan\OneDrive\Desktop\make a thing\censerve\CenServe"
pip install pyinstaller
```

Then run the build (one line, or split across lines with backtick in PowerShell):

```powershell
pyinstaller --onefile --noconsole --name CenServe --hidden-import insightface --hidden-import easyocr --hidden-import ultralytics --hidden-import nudenet --add-data "censerve/models;censerve/models" --add-data "censerve/web/static;censerve/web/static" run.py
```

The exe will be at: **`dist\CenServe.exe`**

If the build fails with a missing module, add it with another `--hidden-import <module>` and rebuild.

---

## 3. Create a GitHub Release and attach the exe

1. Open: **https://github.com/besque/CenServe/releases**
2. Click **“Draft a new release”**.
3. **Choose a tag:** type `v1.0.0` (create new tag).
4. **Release title:** e.g. `CenServe v1.0 — Windows`
5. **Description:** optional (e.g. “First release. Requires OBS Virtual Camera.”).
6. **Attach the exe:** drag and drop `dist\CenServe.exe` into the “Assets” area (or use “Attach binaries”).
7. Click **“Publish release”**.

After publishing, the download URL will be:

**https://github.com/besque/CenServe/releases/download/v1.0.0/CenServe.exe**

The marketing site (`site/index.html`) already points to this URL, so the download button will work.

---

## 4. Deploy the marketing site (optional)

To host the landing page on GitHub Pages:

**Option A — Separate repo**

1. Create a new repo, e.g. `censerve-site`.
2. Copy `site/index.html` into the root of that repo.
3. In that repo: **Settings → Pages → Source:** Deploy from branch **main** (or **master**), root.
4. Site URL will be: `https://besque.github.io/censerve-site`

**Option B — Same repo, `docs` folder**

1. In `CenServe`, create a folder `docs`.
2. Copy `site/index.html` to `docs/index.html`.
3. Commit and push:
   ```powershell
   mkdir docs
   copy site\index.html docs\index.html
   git add docs\index.html
   git commit -m "Add GitHub Pages site"
   git push origin main
   ```
4. **Settings → Pages → Source:** Deploy from branch **main**, folder **/docs**.
5. Site URL: `https://besque.github.io/CenServe`

---

## Quick checklist

- [ ] Code pushed: `git push origin main` (or `master`)
- [ ] `dist\CenServe.exe` built with PyInstaller
- [ ] New release created at https://github.com/besque/CenServe/releases with tag **v1.0.0**
- [ ] `CenServe.exe` attached to that release
- [ ] Download link works: https://github.com/besque/CenServe/releases/download/v1.0.0/CenServe.exe
