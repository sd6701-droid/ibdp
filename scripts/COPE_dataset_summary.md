# COPE dataset — structure summary

Root (BigPurple): `/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential`
Owner: `ys1001:shenlab`. Everything below is **PHI-confidential** — keep it on the cluster unless you have IRB/HIPAA cover for a local copy.

Derived from the `find`/`du` dump in `COPE-structure.txt.md` (2026-08-10 snapshot).

## 1. Top level: 3 timepoints + 4 task-extract folders

| Folder | Size | Files | What it is |
|---|---|---|---|
| `6_Months/` | 976 G | 2,075 | Raw 6-month visit sessions, one dir per subject (157 subjects) |
| `12_Months/` | 999 G | 2,050 | Raw 12-month visit sessions, one dir per subject (155 subjects) |
| `42_Months/` | 126 G | 16,239 | 42-month visit — **not** subject-first; organized by modality (fNIRS, tasks, HR, notes) |
| `Arm_Restraint_6mo/` | 84 G | 140 | Flat folder of cropped arm-restraint clips (≈1/subject) |
| `Freeplay_6mo/` | 194 G | 131 | Flat folder of cropped free-play clips |
| `MAAP_Cropped_Videos/` | 61 G | 157 | Flat folder of cropped MAAP task clips (+ `MAAP Cropped Videos Redo/`) |
| `final-datasets/` | 257 K | 4 | Curated/derived tabular outputs — **look here first** |

Total ≈ **2.4 TB / ~20.8 k files**. Note the file-count inversion: 6/12-month are few huge video files; 42-month is many small neuroimaging/log files.

## 2. Data categories

| Category | Extensions (counts) | Where |
|---|---|---|
| **Video** | `.mov` 3,160, `.mp4` 1,425 | 6/12_Months subject dirs; `42_Months/Task Videos & Notes/`; the 3 cropped-task folders |
| **Visit notes** | `.pdf` 371 (+ `.docx`→pdf) | one per subject-visit, e.g. `109_12_E2_Notes.pdf` |
| **Surveys** | `.pdf` (7 Microbiome, 1 Stool) | 12_Months only |
| **fNIRS** | `.snirf` 924, `.nirs` 849, `.wl1`/`.wl2` 866 each, `.hdr` 1,075, `.tri` 748, `.mat` 1,081 | `42_Months/fNIRS/` (+ `Test fNIRS/`); NIRx/Aurora raw + Satori/Homer processed (`Eric Project/Satori Processed/...`) |
| **Task logs** | `.csv` 1,495, `.json` 2,342, `.txt` 907 | `42_Months/{Jumble CSVs, Jumble PIPs, Puzzles CSVs, QUILS}` |
| **Physiology** | — | `42_Months/Heart Rate/` |
| **Consent / admin** | — | `42_Months/AV consents/` (highest-sensitivity folder) |
| **Analysis + backups** | `.zip` 1,072, R code | `42_Months/{R Scripts, DELL Laptop Backup FIles, Task Laptop Backup Files}` |
| **Junk** | `.DS_Store` 957 | everywhere — exclude in any glob |

Notes-file naming: `{ID}_{age}_{E#}_Notes.pdf`. `E#` = session/examiner block — **E2 dominates (234 files)**, E1 = 20, E3 = 17. Formatting is inconsistent (spaces, `.docx.pdf`, `Notes_E2` vs `E2_Notes`, `421/439_42_notes.pdf` — an ID mismatch worth flagging to the data manager).

## 3. How it all joins

**Subject ID is the only join key.** Four ID namespaces coexist — likely recruitment waves/sites:

- numeric (`5`, `109`, `651`) — 136
- `M#####` (`M21052`) — 142
- `N###` (`N194`) — 22
- `T####` (`T1836`) — 12

**Longitudinal overlap (6 vs 12 months):** 103 subjects in both, 54 only-6mo, 52 only-12mo. So the balanced longitudinal cohort is ~103.

**42-month linkage:** at least 88 subjects have visit notes there; 26 of those also appear in 6_Months and 34 in 12_Months (lower bound — the `find -maxdepth 2` never listed the 42-month subject dirs, so re-run deeper to get the true count).

**Raw → derived chain:**

```
6_Months/<ID>/*.mov            (full session, raw)
        │  cropped per paradigm
        ├──► Arm_Restraint_6mo/<ID>*.mov      ~140 clips
        ├──► Freeplay_6mo/<ID>*.mov           ~131 clips
        └──► MAAP_Cropped_Videos/<ID>*.mov    ~157 clips  ── ML-ready layer
                                                    │
6_Months/<ID>/<ID>_6_E2_Notes.pdf  ── QC/labels ────┘
final-datasets/                    ── merged subject-level table
```

For the video-description pipeline in this repo, the three cropped-task folders + `final-datasets/` are the practical entry point; the per-subject trees are the raw archive.

## 4. Fetch a file locally (PHI — use an encrypted disk, delete after)

Direct `scp` from your laptop (BigPurple is reachable directly — no jump host needed):

```bash
R=sd6701@bigpurple.nyumc.org
D=/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential

# one visit-notes PDF
scp "$R:$D/12_Months/109/109_12_E2_Notes.pdf" ~/Downloads/

# a path containing spaces — quote twice (local shell + remote shell)
scp "$R:'$D/42_Months/Task Videos & Notes/109/109_42_notes.pdf'" ~/Downloads/

# a survey PDF
scp "$R:'$D/12_Months/109/109 _12_Microbiome_Survey.pdf'" ~/Downloads/

# a whole subject dir, notes/docs only (no multi-GB video)
rsync -avz --include='*/' --include='*.pdf' --include='*.docx' --include='*.csv' \
      --exclude='*' "$R:$D/12_Months/109/" ~/Downloads/cope_109_12/
```

Docs are stored as `.docx.pdf` (already converted); true `.docx` originals mostly live under `42_Months/Task Videos & Notes/`.

## 5. Commands to go deeper (run on the cluster)

```bash
D=/gpfs/data/shenlab/data/pediatric_psychiatry/COPE_Videos_PHI_confidential

# 0. the small curated tables — cheapest, highest-value first look
ls -la "$D/final-datasets"; head -3 "$D/final-datasets/"*.csv

# 1. what a typical subject dir actually holds
ls -la "$D/6_Months/109" "$D/12_Months/109"

# 2. the 42-month subject list that maxdepth=2 missed
ls "$D/42_Months/Task Videos & Notes" | sort > /tmp/s42.txt; wc -l /tmp/s42.txt

# 3. full subject x timepoint coverage matrix
for t in 6_Months 12_Months; do ls "$D/$t" > /tmp/$t.ids; done
comm -12 <(sort /tmp/6_Months.ids) <(sort /tmp/12_Months.ids) | wc -l

# 4. video inventory: name, size, duration (needs ffprobe module)
find "$D/Freeplay_6mo" -iname '*.mov' -printf '%s\t%p\n' | sort -rn | head
module load ffmpeg && ffprobe -v error -show_entries format=duration,size \
  -of csv=p=0 "$D/Freeplay_6mo/<file>.mov"

# 5. do the cropped folders map 1:1 to subjects?
ls "$D/MAAP_Cropped_Videos" | sed -E 's/[_ .].*//' | sort -u | wc -l

# 6. naming conventions inside the raw trees (what tasks exist per visit)
find "$D/6_Months" -maxdepth 2 -type f -iname '*.mov' | sed 's|.*/||' \
  | sed -E 's/^[A-Z]*[0-9]+_//' | sort | uniq -c | sort -rn | head -30

# 7. fNIRS layout
find "$D/42_Months/fNIRS" -maxdepth 3 -type d | head -40

# 8. task-log schemas
head -2 "$D/42_Months/Puzzles CSVs/"*.csv | head -20

# 9. duplicate / mismatched IDs inside notes filenames
find "$D" -iname '*notes*.pdf' | awk -F/ '{d=$(NF-1); f=$NF; sub(/[_ ].*/,"",f); if (f!=d) print}'

# 10. total per-subject footprint (slow — submit as a job, not on the login node)
du -sh "$D/6_Months"/* | sort -h | tail -20
```

## Open questions for the data manager

1. What do the four ID prefixes (`M`, `N`, `T`, bare numeric) encode?
2. `E1/E2/E3` — examiner, visit attempt, or session block?
3. `421/439_42_notes.pdf` and `615/625_42_notes.pdf` — misfiled notes or intentional?
4. `M25139` vs `M25139-2`, and `MAAP Cropped Videos Redo/` — which copy is canonical?
5. What are the 4 files in `final-datasets/`, and is there a codebook?
