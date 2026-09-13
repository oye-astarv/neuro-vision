# NeuroVision — Project Notes

> Living document. Updated as we progress so context loss is a non-event.
> Bootstraps a new session: read this file + the code in `src/` / notebooks.

## Goal
Decode motor intention from fMRI and map it to a 3D digital human twin.
1. Classify which body part is moving (12 classes) from whole-brain activity.
2. Explainable AI → recover the cortical somatotopic map (motor homunculus).
3. Visualize predicted brain activation + drive a ROS2/Gazebo human skeleton
   (jointed arms/legs) that moves according to the model's prediction.

## Dataset — OpenNeuro `ds004044`
- Ma et al. 2022, *Scientific Data*; DOI 10.1038/s41597-022-01644-4.
- **62 subjects** (NOT 68 — S3 actually holds 62). 12 movement conditions.
- Conditions: `Toe, Ankle, LeftLeg, RightLeg, Finger, Wrist, Forearm,
  Upperarm, Jaw, Lip, Tongue, Eye` (exact spelling matters).
- 6 blocked-design runs/subject (~7.7 min), 2 mm isotropic, whole brain+cerebellum.
- Three derivative pipelines: `ciftify`, `fmriprep`, `melodic`.

## Data we downloaded (local)
- Path: `data/raw/ds004044/derivatives/ciftify/sub-XX/results/
  ses-1_task-motor_hp200_s4_level2.feat/sub-XX_..._level2_cope_<Cond>_hp200_s4.dscalar.nii`
- **744 files = 62 subjects × 12 conditions** (~0.7 GB). Only the
  single-condition-vs-baseline COPEs. The `-Avg` / `Finger-Wrist` /
  `Upper-Lower` contrast files were deleted (39 leftovers from an interrupted
  first run).
- COPE = GLM beta = activation for that movement **relative to rest (baseline)**.
  Already ICA-denoised by authors; clean.

## Data format / key facts
- CIFTI `.dscalar.nii` → `nibabel` loads as `Cifti2Image`, shape `(1, 91282)`.
- `vec = nib.load(f).get_fdata()[0].astype("float32")` → `(91282,)`;
  each entry = one **grayordinate** (cortical surface vertex + subcortical voxel,
  standardized across subjects → directly comparable, no registration).
- Subject IDs are **non-contiguous** (some missing, e.g. 15/19/28/40/41/64);
  map real ID → 0..61 for any array indexing.

## Environment
- venv at `.venv` (Python 3.9.6 system). Jupyter kernel = `.venv`.
- ⚠️ Python 3.9 is EOL → Jupyter shows "kernel no longer supported" (non-fatal).
  Optional future-proofing: `brew install python@3.12` + rebuild venv.
- Installed: `numpy, scipy, nibabel, pandas`.
- NOT yet installed: `scikit-learn, nilearn, torch, captum, vedo`
  (install per stage as needed).
- Restart Jupyter kernel if `import numpy` errors with
  "CPU dispatcher tracer already initialized" (stale-kernel artifact).

## Chosen approach — "Route B" (CIFTI surface)
- Use the COPE dscalar maps directly as model features (no re-preprocessing).
- Plan: loader → `X(744,91282)`, `y(744,)`, `groups(744,)` →
  cross-subject **LOSO** decoding → XAI → brain viz → ROS2/Gazebo twin.
- Within-subject CV is NOT meaningful (only 1 map/condition/subject; the
  level-2 COPE already averages runs). Always split by `groups` (subject).

## Pipeline plan (stages)
1. **Loader** (user building): glob COPEs → per file (vector, label, subject)
   → stack X/y/groups. Optional 3D form `(subject, condition, 91282)`.
2. **Baseline decoder**: regularized logistic regression, LOSO.
   `coef_` shape `(12, 91282)` = first XAI / somatotopic map.
   (Alternative discussed: XGBoost — strong + feature-importance XAI, but NOT
   deep learning; only use if "deep" isn't a hard requirement.)
3. **Deep model** (later): MLP on parcellated features → surface **graph-CNN**
   on cortical mesh (the novel deep contribution; learns spatial structure the
   linear/XGBoost models ignore). Mesh available in `ciftify/.../native_surface`.
4. **XAI**: linear coef / XGBoost importance / captum gradients → map 91282
   weights back to brain surface = somatotopic map.
5. **Brain viz**: Connectome Workbench (`wb_command` + `wb_view`), separate
   binary (not pip).
6. **3D digital human twin**: ROS2 + Gazebo human skeleton; joints (arms/legs/
   etc.) animate per predicted movement. ⚠️ On macOS, native ROS2+Gazebo is
   painful → plan to run in a **Docker (Linux) container**. Not installed yet.

## Key concepts (glossary)
- **Grayordinate**: one gray-matter location (cortex vertex or subcortical
  voxel); CIFTI lists ~91,282 of them as a single vector.
- **COPE**: GLM beta = condition activation vs rest/baseline.
- **Overfitting**: memorizing training subjects instead of the movement;
  LOSO is the test for it.
- **XAI**: showing which input locations drove a prediction.
- **LOSO**: leave-one-subject-out; train on 61, test on 1; repeat 62×.

## Decisions log
- Download only the 12 condition COPEs (~0.7 GB), not the 75 GB raw.
- Route B (CIFTI) chosen over volumetric fmriprep denoised BOLD.
- Linear baseline first (proof + XAI floor); graph-CNN is the deep goal.
- 3D tensor view = `(subject, condition, 91282)` for slicing/visualization;
  flatten to 2D for the classifier.

## Status / next
- ✅ Dataset downloaded + cleaned (744 files).
- ✅ venv + core libs; read test passed (shape (1,91282)). torch now installed too.
- ✅ Loader BUILT: `dataset` dict keyed by condition -> list of 62 (91282,) vectors.
  12×62 = 744 verified. Subject order consistent across conditions (lexical glob).
  Convert to 3D tensor: `X3 = np.stack([np.stack(dataset[c]) for c in CONDS], axis=1)`
  -> (62, 12, 91282). Cast to float32 when stacking.
- ⏭ Build X3 tensor; then LOSO evaluation loop -> accuracy + first XAI map.

## Loader notes
- Chose dict-of-lists over manual (subject,condition) tensor indexing (avoids
  missing-subject index bug). Subject IDs not explicitly tracked; relies on
  consistent lexical glob order (valid since all 12 conditions have 62 each).
- Considering switch to a FLAT LIST OF TUPLES (values, label, group) for all 744
  samples: keeps dataset intact, avoids per-key positional indexing, single glob
  pass (extract subj+cond from filename), easy to derive X/y/groups + group-split.
  Source of truth = full list; train/test derived by filtering on `group`.
- `groups` for LOSO = `np.repeat(np.arange(62), 12)` under that assumption.

## Train/test split (decided)
- Manual grouped hold-out by SUBJECT, not by sample (avoids leakage).
- `test_subj_idx = np.random.choice(62, size=10, replace=False)` -> 10 subject
  POSITIONS (0..61, order of `dataset[cond]` lists). For each, take all 12
  conditions via `dataset[part][idx]` -> 120 test samples (10x12).
- Test sample = tuple `(idx, condition, values)`; idx kept as group label.
- Train = complement (52 subjects). Set random seed for reproducibility.
- Single 10-subj hold-out = dev/eval. FINAL eval = LOSO (62 folds).
- `idx` is list-position, not real sub-ID; fine for grouping.

## Processed split / X-y build (planned)
- From `d_train`/`d_test` (lists of `(subject, condition, values)` tuples):
  X = stacked value vectors (float32), y = condition index via `CONDS.index`,
  groups = subject (kept for later LOSO).
- Expected: X_train (624,91282) y_train (624,), X_test (120,91282) y_test (120,).
- Persist to `data/processed/split.npz` (np.savez) so nib reload not needed each run.
- Also saved raw tuple lists `d_train.pkl` / `d_test.pkl` (pickle) on user request.

## Baseline (linear) — why it exists
- NOT the final model (deep graph-CNN on surface is the novelty). It is the baseline.
- Proves task is learnable & pipeline/split correct (trains in seconds).
- `coef_` = instant human-readable somatotopic map (XAI floor); validates model
  looks at real motor regions, not leakage.
- Required comparison: deep model's value = how much it beats this simple method.

## Next steps (current)
1. Train regularized LogisticRegression on X_train->y_train (91282 feats >> 624
   samples -> heavy L2 reg, small C).
2. Eval on X_test: accuracy, per-class, confusion matrix.
3. Extract coef_ (12x91282) -> first importance map; save for surface viz.
4. Build deeper model (MLP -> graph-CNN on surface mesh); compare to baseline.
5. 3D digital human twin (ROS2+Gazebo, needs Docker).

## Do we even need a NN? (decision logic)
- If linear baseline already decodes well, a NN is NOT required for a working system
  (twin just consumes predictions either way).
- NN is justified only if it: (1) raises accuracy via non-linearities, (2) is the
  geometry-aware graph-CNN on the brain surface (the actual novel contribution),
  (3) enables richer XAI (attention / relevance propagation).
- Conditional: if baseline already ~85-90% cross-subject, plain MLP may add little;
  then NN's role = match/beat via structured graph-CNN + better XAI. Report both,
  show delta (or honestly note it's small). Get baseline number FIRST, then decide.

## Model candidates (selection ladder)
All use same X/y + grouped split; differ in architecture/input shape.
- T1 baselines (flat vector, 91282 feats): Logistic Regression (L2, small C; coef_
  XAI), LinearSVC (decoding standard), optional RF/XGBoost (tree baseline).
- T2 shallow deep (flat): MLP (PyTorch or sklearn) — tests non-linearity alone.
- T3 NOVEL: Graph CNN on cortical surface mesh (PyTorch Geometric: GCN/GAT/MeshCNN).
  Cortex as graph (vertices=nodes, mesh edges=connectivity, activation=features).
  Spatial weight-sharing = viable deep model on tiny data (624 samples); biologically
  motivated; richest XAI (attention, Grad-CAM, integrated gradients).
  Needs: torch-geometric (NOT installed) + grayordinate->surface node mapping
  (CIFTI already vertex-ordered for cortex; handle subcortical separately).
- Selection: compare via GroupKFold CV on TRAIN (no subject leakage). Baseline=LR is
  reference; a model wins only if it beats baseline meaningfully + serves XAI/twin.
- Path: LR -> LinearSVM -> MLP -> Graph-CNN.
- Metrics: accuracy + macro-F1 + confusion matrix (catch LeftLeg<->RightLeg confusion).

## SCOPE CHANGE (paper de-scoped)
- User dropped the paper requirement. Multiple models NOT required for the project.
- Project deliverable = classifier + XAI + twin. Ship ONE model.
- Minimal viable NeuroVision: Logistic Regression alone satisfies all 3 goals
  (classify + coef_ brain map XAI + label feeds twin).
- Graph-CNN now OPTIONAL polish (better accuracy / richer XAI / DL centerpiece),
  not a requirement. Lean plan: train LR -> confirm -> use for XAI + twin;
  add graph-CNN later only if desired.
- CONFIRMED PLAN (user): (1) Logistic Regression baseline -> eval + coef_ XAI map;
  (2) THEN build a CNN / graph-CNN on the brain surface for optional boost.
  scikit-learn NOT installed yet -> must `pip install scikit-learn` in .venv.
  Baseline cell uses Pipeline(StandardScaler + LogisticRegression(C=0.1,max_iter=3000)).

## Baseline results (LR)
- ACHIEVED 85% cross-subject accuracy on X_test (120 samples, 10 held-out subjects).
  Chance = 8.3% (1/12). Strong baseline. Confusion matrix diagonal-dominant.
- classification_report initially printed all-zeros + UndefinedMetricWarnings:
  bug was `labels=CONDS` (strings) while y/pred are ints 0-11. Fix = `target_names=CONDS`.
- Next: extract coef_ (12x91282) as first brain-region XAI map; then build CNN/graph-CNN.

## NN build plan (DECIDED: build it, 85% baseline notwithstanding)
- User chose to build the NN. Worthwhile target = graph-CNN on brain surface (geometry-aware),
  since plain MLP unlikely to beat 85%.
- Step 1 MLP warm-up (PyTorch, no new deps): validate train/eval loop, test non-linearity.
  Risk: 91282 feats / 624 samples -> overfit; may NOT beat 85%. That's the signal that
  structure (graph) is needed, not more layers.
- Step 2 graph-CNN (real contribution): needs torch-geometric (install), cortical surface
  mesh (ciftify native_surface) + vertex adjacency, map 91282 grayordinates->surface nodes
  (cortex vertices direct; subcortical handle separately/drop). GCN/GAT, node features=activation,
  12-class. Compare to 85%.
- NN XAI: attention / Grad-CAM / integrated gradients (richer than coef_).
- Twin is model-agnostic: NN just replaces LR as the predictor (brain map -> label).

## GCN LINE CLOSED (CPU cost)
- GCN test acc: raw+mean 0.175, standardized+mean 0.175, raw+max 0.158
  (loss 2.98->2.42, starts ABOVE chance 2.485 = raw-scale pathology confirmed).
  Both single-variable fixes acquitted; architecture-as-built convicted. THREAD CLOSED.
- A single retrain costs ~10 HOURS on this CPU -> further GCN iteration unviable here.
- DECISION: LR (85%) crowned final cortex branch. GCN infra (mesh, A, placement, H)
  retained for reference/viz; may reopen on GPU later.
- Pivot: step 2 = subcortical MLP on S (minutes); step 3 = final linear fusion head
  over [LR verdict + subcortical verdict] -> random-case predictor demo.

## MLP warm-up RESULTS
- MLP test acc = 0.758, train acc = 0.907. UNDERPERFORMS LR (0.85) and overfits
  (train 0.91 -> test 0.76). Confirms flat non-linearity alone does NOT beat the
  structured linear model.
- Implication: plain MLP is not the answer. To beat 85% need the graph-CNN's spatial
  weight-sharing; otherwise accept LR (85%) as the final model.
- Environment fix: scikit-learn install pulled numpy 2.0.2 which broke torch's numpy
  bridge; pinned numpy<2 (1.26.4) in .venv to make torch work.
- INSTALLED for graph-CNN: torch-geometric 2.6.1, neuromaps 0.0.7 (fetch_fslr mesh),
  nilearn 0.12.1 (viz). 

## Graph-CNN build steps (status)
- Step 1 (topology): fetch fs_LR midthickness mesh (neuromaps.fetch_fslr), build
  edge_index from faces (left+right, right offset by nL). nodes=64984, edges from faces.
- Step 2 (features): map 91282 COPE/coef_ -> cortex nodes via CIFTI brain-model axis
  (cortex vertices align to fs_LR directly). Each sample = one graph, node feature = activation.
- Step 3 (model): GCN/GAT (PyG Data/DataLoader), train on grouped split, compare to 85%.

## Glossary (key terms)
- COPE: GLM beta = activation of a movement vs rest. Your X feature.
- CIFTI / dscalar: brain format storing values at grayordinates (not voxels).
- Grayordinate: one CIFTI brain location = a cortex vertex OR a subcortical voxel.
  91282 total = feature size.
- Cortex vs subcortical: cortex = folded outer surface; subcortical = deep nuclei.
- Surface mesh: brain shape as vertices + faces. fs_LR/32k = standard surface template
  (~32492 verts per hemisphere).
- Somatotopy / homunculus: motor cortex laid out by body part (foot medial, hand lateral,
  face inferior).
- Node (GNN): a brain location (cortex vertex). Node feature: its activation value.
- Edge / adjacency / edge_index: connection between neighboring nodes (mesh neighbors).
- Graph: nodes + edges. Each sample = one graph (same topology, different node features).
- GNN / GCN / GAT: graph neural net; GCN = graph convolution, GAT = graph attention.
- Message passing: node updates by gathering neighbor info.
- Weight sharing: same weights at every node/edge -> few params (why GNN works on tiny data).
- Baseline: simple model to beat (LR = 85%).
- coef_: LR weights = first XAI brain map.
- Overfitting: train >> test (MLP did this: 0.91 -> 0.76).
- Grouped split / LOSO: split by subject (no leakage).
- Logits / cross-entropy: raw class scores / classification loss.
- Twin: ROS2+Gazebo body avatar (separate from brain mesh viz).
- wb_view / nilearn / vedo: brain viewers / visualizers.

## Brain surface viz (planned)
- Goal: 3D brain mesh that "lights up" per condition. Needs fs_LR template mesh
  (NOT in download; get from HCP/Connectome Workbench) + nilearn or vedo (not installed).
- Approach A: per-condition heatmap (coef_[c] red/blue) -> flip through 12 brains.
- Approach B (preferred): ONE brain, each cortex vertex colored by its WINNING class
  (coef_.argmax(0)) from a 12-color map (tab20), brightness = coef_.max(0).
  Shows full somatotopic layout in distinct colors at a glance. Great project figure.
- coef_ (12,91282) -> argmax over classes per grayordinate -> map cortex vertices -> render.
- Distinguish from ROS2/Gazebo body "twin" (separate piece).

## Viz saga RESOLVED
- view_surf plotly path failed on NaN backgrounds (grey renders despite hot data).
  Fix: zero-background canvases + threshold=3.0 -> full color, interactive 3D WORKS.
- Final demo = ONE self-contained block: random held-out case -> LR verdict +
  correctness -> glowing left-hemisphere map. GIF plan retired.
- Remaining: clear stale viewer outputs (iframe bloat), 12-color somatotopy figure
  (optional polish), twin wiring, README.

## Open questions / todo
- Install `scikit-learn` when starting the decoder.
- Decide final model: baseline-only vs MLP vs graph-CNN (depends on whether
  "deep learning" is a hard requirement).
- Set up ROS2/Gazebo (Docker) for the twin — biggest infra unknown.
- `wb_command` for CIFTI surface visualization.
