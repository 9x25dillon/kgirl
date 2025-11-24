# COMPILATION AND UPLOAD INSTRUCTIONS

## Quick Start (For Claude CLI)

```bash
# 1. Download all files from this conversation
# Already done if you're reading this!

# 2. Compile the LaTeX document
cd ~/path/to/files
pdflatex Cognitive_Renewal_Dynamics_FINAL.tex
bibtex Cognitive_Renewal_Dynamics_FINAL
pdflatex Cognitive_Renewal_Dynamics_FINAL.tex
pdflatex Cognitive_Renewal_Dynamics_FINAL.tex

# 3. Check the output
open Cognitive_Renewal_Dynamics_FINAL.pdf  # Mac
# or
xdg-open Cognitive_Renewal_Dynamics_FINAL.pdf  # Linux
```

---

## Step-by-Step Instructions

### STEP 1: Install LaTeX (if not already installed)

**Mac:**
```bash
brew install --cask mactex
# or download from: https://www.tug.org/mactex/
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install texlive-full
```

**Windows:**
Download MiKTeX from: https://miktex.org/download

---

### STEP 2: Compile the Document

The LaTeX file requires **three passes** to properly resolve references:

```bash
# First pass: Generate auxiliary files
pdflatex Cognitive_Renewal_Dynamics_FINAL.tex

# Second pass: Process bibliography
bibtex Cognitive_Renewal_Dynamics_FINAL

# Third pass: Resolve all references
pdflatex Cognitive_Renewal_Dynamics_FINAL.tex

# Fourth pass: Final cleanup
pdflatex Cognitive_Renewal_Dynamics_FINAL.tex
```

**Expected output:**
- `Cognitive_Renewal_Dynamics_FINAL.pdf` (your final paper!)
- Various auxiliary files (.aux, .bbl, .blg, .log, .out)

**Common issues:**

1. **Missing packages:** LaTeX will tell you which package is missing. Install via:
   ```bash
   # Mac
   sudo tlmgr install <package-name>
   
   # Linux
   sudo apt-get install texlive-<package-name>
   ```

2. **Bibliography errors:** Make sure you ran `bibtex` AFTER the first `pdflatex` run

3. **Hyperref warnings:** Safe to ignore most hyperref warnings about destinations

---

### STEP 3: Review the PDF

**Check these sections carefully:**

- [ ] Abstract is clear and compelling
- [ ] All equations render correctly
- [ ] Table 1 (mapping) is properly formatted
- [ ] References are properly cited
- [ ] No "??" marks (means missing reference)
- [ ] Page breaks don't split equations awkwardly
- [ ] Appendices are readable

**Quick visual check:**
- Paper should be 20-24 pages
- Clean, professional appearance
- No obvious typos or formatting glitches

---

### STEP 4: Make Final Edits (if needed)

**Common edits:**

1. **Add your email:**
   Find line 20-ish: `\href{mailto:your.email@example.com}{your.email@example.com}`
   Replace with your actual email

2. **Add your academia.edu profile:**
   Find line ~400: `\textbf{Preprint:} Available at \url{https://www.academia.edu/your-profile}`
   Replace with your actual URL

3. **Update date if needed:**
   Line 25: `\date{November 3, 2025\\`
   Change to upload date if different

4. **Customize acknowledgments:**
   Section 8 if you want to add collaborators

After edits, recompile:
```bash
pdflatex Cognitive_Renewal_Dynamics_FINAL.tex
```

---

### STEP 5: Upload to Academia.edu

**Login to academia.edu:**
1. Go to https://www.academia.edu
2. Log in to your account (or create one)

**Upload the paper:**
1. Click "Upload" button (top right)
2. Select `Cognitive_Renewal_Dynamics_FINAL.pdf`

**Fill in metadata:**

**Title:**
```
Cognitive Renewal Dynamics: An Application of the Kernel Renewal Condition to Neural Coherence
```

**Abstract:**
```
Consciousness exhibits a paradox: continuity emerges from constant change. This paper resolves this paradox by extending Halldór G. Halldórsson's Kernel Renewal Condition into cognitive neuroscience, demonstrating that neural coherence operates as a sequential–invariant renewal loop where identity arises not from persistence but from rhythmic proportion. Using EEG coherence dynamics, the model predicts measurable relationships between coupling elasticity, coherence recovery, and awareness states, providing testable hypotheses for meditation effects, learning rates, and cognitive flexibility.
```

**Keywords:** (copy from paper)
```
Neural Coherence, Phase Synchronization, Kernel Renewal Condition, Consciousness Dynamics, EEG Analysis, Cognitive Flexibility, Meditation Neuroscience, Learning Rate, Invariant Field, Sequential Processing, Memory Consolidation, Attention Networks
```

**Research Interests:** (select from dropdown)
- Neuroscience
- Cognitive Science  
- Consciousness Studies
- Dynamical Systems
- Computational Neuroscience
- Meditation Research

**Co-authors:**
- Add Halldór G. Halldórsson if he has an academia.edu account
- Or mention in acknowledgments

**License:**
- Recommended: "Creative Commons Attribution 4.0" (CC BY)
- Allows others to build on your work with proper citation

---

### STEP 6: Promote (Optional but Recommended)

**Post the plain language summary:**
1. Open `CRD_Plain_Language_Summary.md`
2. Copy the text
3. Post on:
   - Twitter (see thread template in `CRD_Graphics_and_Social.md`)
   - LinkedIn (see professional post template)
   - Reddit r/neuroscience, r/cogsci (see post template)

**Add link to your paper everywhere:**
Once uploaded, academia.edu gives you a URL like:
```
https://www.academia.edu/XXXXXXX/Cognitive_Renewal_Dynamics
```

Update all your posts with this link.

**Email to interested parties:**
- Halldór G. Halldórsson (your collaborator)
- Neuroscience labs working on consciousness
- Meditation research centers
- Anyone who expressed interest

---

### STEP 7: Track Engagement

Academia.edu provides analytics:
- Views per day
- Downloads
- Citations (eventually)
- Geographic distribution

Check monthly to see:
- Which sections get read most (heatmap)
- Who's citing your work
- Collaboration requests

---

## TROUBLESHOOTING

### LaTeX won't compile

**Error: `! LaTeX Error: File 'xyz.sty' not found.`**

**Solution:**
```bash
# Mac
sudo tlmgr install xyz

# Linux  
sudo apt-get install texlive-xyz

# Windows (MiKTeX)
# Open MiKTeX Console → Packages → Search → Install
```

---

### PDF looks wrong

**Issue:** Equations are cut off

**Solution:** Add `\allowdisplaybreaks` before `\begin{document}`

**Issue:** References show as "??"

**Solution:** Run bibtex, then pdflatex twice more

**Issue:** Hyperlinks don't work

**Solution:** Make sure you have `\usepackage{hyperref}` in preamble (already included)

---

### Academia.edu issues

**Issue:** Upload rejected

**Possible reasons:**
- File too large (limit is 100MB, yours should be <5MB)
- Wrong file type (must be PDF)
- Copyright issue (make sure it's your original work)

**Solution:** Check file size, ensure PDF format, verify licensing

---

## FILE MANIFEST

After compilation, you should have:

**Essential files:**
- `Cognitive_Renewal_Dynamics_FINAL.pdf` ← **UPLOAD THIS**
- `Cognitive_Renewal_Dynamics_FINAL.tex` (source)
- `CRD_Plain_Language_Summary.md` (for promotion)
- `CRD_Graphics_and_Social.md` (for social media)
- `COMPILATION_INSTRUCTIONS.md` (this file)

**Generated files (can delete after successful compilation):**
- `*.aux`, `*.log`, `*.out`, `*.bbl`, `*.blg`

**Optional but recommended:**
- Graphical abstract image (create using specs in `CRD_Graphics_and_Social.md`)

---

## POST-UPLOAD CHECKLIST

- [ ] PDF compiled successfully
- [ ] Reviewed final PDF for errors
- [ ] Uploaded to academia.edu
- [ ] Added proper metadata (title, abstract, keywords)
- [ ] Selected research interests
- [ ] Set license (CC BY recommended)
- [ ] Posted plain language summary on social media
- [ ] Emailed collaborators with link
- [ ] Added paper to your CV/website
- [ ] Celebrated! 🎉

---

## FUTURE UPDATES

If you want to update the paper later:

1. Edit the .tex file
2. Increment version: `\date{November 3, 2025 (v2)}` 
3. Add note: `\textit{This version corrects...}`
4. Recompile
5. Upload new PDF to academia.edu as "new version"

Academia.edu keeps version history, so readers can see evolution.

---

## NEED HELP?

**LaTeX issues:**
- Stack Exchange: https://tex.stackexchange.com
- Overleaf documentation: https://www.overleaf.com/learn

**Academia.edu issues:**
- Help center: https://support.academia.edu

**Content questions:**
- Reach out to me (Randy Lynn): your.email@example.com
- Or discuss with Halldór G. Halldórsson

---

**Good luck! This is important work. Get it out there.** 🚀
