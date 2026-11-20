# DFDC Deepfake Detection — Reproduction and Extension

**Advanced Generative AI | MS Data Science | FAST-NUCES Islamabad**  
**Author:** Haider Rasool Qadri (24I-8023)

This repository records two connected assignments. **Assignment 2 (A2)** is a resource-constrained partial reproduction of CNN deepfake detection and FGSM experiments from Abbasi et al. (2025). **Assignment 3 (A3)** extends the same trained models to a second dataset (FaceForensics++), tests two input preprocessing methods, and evaluates a lightweight conditional preprocessing head.

> **Main finding:** The head accurately distinguished clean images from FGSM-attacked images in the tested setting, but conditional JPEG and Gaussian preprocessing did **not** consistently recover correct REAL/FAKE classifications. Detection accuracy must not be confused with deepfake classification accuracy.

## Research stages

| Stage | Implemented work | Dataset |
| --- | --- | --- |
| A1 — Proposal | Initial research plan for lightweight inference-time defenses. The eventual experiments use a reduced scope. | Planned: FF++, DFDC and other benchmarks |
| A2 — Reproduction | Fine-tune Xception, VGG16, ResNet50 and ResNet101; evaluate clean frames and FGSM; test ResNet50+101 ensemble. | Reduced DFDC subset |
| A3 — Extension | Evaluate the four **frozen** CNNs on FF++; FGSM; JPEG Q75; Gaussian 3×3, σ=0.5; logistic-regression detection head; conditional routing; video-folder cluster bootstrap. | FaceForensics++ extracted-frame subset |

## A2 reproduction: DFDC

- Data: **9,590 preprocessed face images**, split into **6,454 train / 1,538 validation / 1,598 test** images using source-video groups.
- Test data: **550 REAL / 1,048 FAKE**. This class imbalance affects raw accuracy.
- Input: **299×299 for all four models**. This and the reduced data size differ from the base-paper protocol.
- Start: ImageNet-pretrained backbones; partial fine-tuning in PyTorch/Kaggle.
- Attack: untargeted white-box FGSM at ε=0.01 (pixel values in [0, 1]).

| Model | Clean accuracy | FGSM ε=0.01 accuracy |
| --- | ---: | ---: |
| Xception | 86.48% | 44.68% |
| VGG16 | 92.30% | 8.07% |
| ResNet50 | 86.86% | 64.58% |
| ResNet101 | 91.30% | 65.71% |

These values are **not an exact reproduction** of the published paper. The ResNet50 FGSM result is especially affected by majority-class FAKE predictions; consult A2 class-wise diagnostics before interpreting accuracy as robustness.

## A3 extension: FaceForensics++

The A3 evaluation uses **1,600 frames** (800 REAL and 800 FAKE) from **200 video/method folders**, eight frames per selected folder. The fake half covers Deepfakes, Face2Face, FaceShifter, FaceSwap and NeuralTextures equally (160 frames each). Four A2-trained CNN checkpoints were reused **without additional fine-tuning**.

### Clean cross-dataset accuracy

| Model | DFDC clean | FF++ clean |
| --- | ---: | ---: |
| Xception | 86.48% | 54.06% |
| VGG16 | 92.30% | 50.19% |
| ResNet50 | 86.86% | 49.19% |
| ResNet101 | 91.30% | 51.06% |

The two dataset splits are different samples, so the change is evidence of weak **cross-dataset transfer in this experiment**, not a controlled estimate of domain shift alone.

### FGSM and preprocessing results on FF++

FGSM uses ε=0.01 and is generated separately using each CNN. JPEG compression is quality 75; Gaussian smoothing uses a 3×3 kernel and σ=0.5. The attack is **non-adaptive to the preprocessing pipeline**.

| Model | FGSM, no defense | Conditional JPEG | Conditional Gaussian |
| --- | ---: | ---: | ---: |
| Xception | 7.13% | 2.13% | 2.38% |
| VGG16 | 15.44% | 35.56% | 15.31% |
| ResNet50 | 48.13% | 34.38% | 40.38% |
| ResNet101 | 35.19% | 25.88% | 24.50% |

**Warning:** VGG16 JPEG improvement does **not** recover the REAL class: its REAL recall under attack remains **0%**. The other methods generally reduce attacked accuracy.

### Adaptive perturbation detection head

The A3 notebook trains a `StandardScaler + LogisticRegression` binary classifier on **30 image-only statistics** (channel means, standard deviations, residuals and image gradients). It learns **CLEAN vs ATTACKED** from Xception-generated FGSM ε=0.01 examples. It does not classify REAL/FAKE and does not use a paired clean reference at prediction time.

- Head training: 480 original FF++ images, paired with 480 FGSM versions → **960** binary examples.
- Head validation: 160 original images + 160 attacked → **320** binary examples.
- Head evaluation: the separate held-out 1,600 images paired with attacked counterparts; **98.97% attack-state accuracy**, **100% attack detection**, **97.94% clean detection** in this tested FGSM setting.
- Conditional routing: if the head predicts CLEAN, the original CNN prediction is used; if ATTACKED, it uses the result after **JPEG OR Gaussian**, evaluated as two separate defenses.

The conditional experiment combines **aligned, saved per-image prediction tables**, rather than measuring a new end-to-end service. The head was not tested against PGD, unseen attack strengths, or a defense-aware attacker. It would be incorrect to claim 98.97% deepfake classification accuracy or general adversarial robustness.

### Statistics

A paired, REAL/FAKE-stratified **video-folder cluster bootstrap (2,000 resamples, seed 42)** estimates 95% confidence intervals for accuracy changes. The video folders are the clusters; some method folders can still share underlying source content or identity.

## Reproducing the experiments

**Recommended platform:** Kaggle Notebook with a GPU for CNN evaluation and FGSM. Most post-processing stages read saved CSV files and do not need GPU.

1. Obtain access to the datasets separately (they are **not bundled in this public repository**):
   - A2 checkpoints / project backup: Kaggle dataset `haiderrasoolqadri/dfdc-trained-models-results` (**private; access required**).
   - A2 DFDC images: `haiderrasoolqadri/dfdc-10k-preprocessed-face-images` (**private; access required**).
   - FF++ extracted frames: Kaggle dataset `adham7elmy/faceforencispp-extracted-frames` (third-party dataset; follow its access/license terms).
   - A3 saved results: `haiderrasoolqadri/dfdc-ffpp-assignment3-results` (**private; access required**).
2. Attach the relevant datasets to Kaggle. Verify the notebook's `/kaggle/input/datasets/...` paths; Kaggle mount paths may vary.
3. Open [`extension/notebooks/dfdc-extension.ipynb`](extension/notebooks/dfdc-extension.ipynb). This is the **original recorded A3 run** with output and backup/recovery cells; it is **not** a clean one-click script. Start with the setup and data manifest cells; adjust input paths; run only the stages you need.
4. Source checkpoints are loaded through the A2 project modules (`evaluate.py`, `config.py`, `attacks/fgsm.py`). These must be available in the restored A2 project directory.
5. The notebook writes its outputs to `/kaggle/working/assignment3/` and figures to `final_figures/`.
6. For a quick audit without GPU, restore the saved A3 CSVs and run the notebook's analysis and plotting cells. To independently reproduce model evaluation, run the GPU-dependent cells and ensure the same sampling manifest, models and seed are used.

**Privacy / access:** Public GitHub viewers cannot automatically access private Kaggle datasets. Checkpoints, raw images and proprietary or third-party frames are intentionally excluded from the repository. The manifests preserve paths and selection metadata, not pixels.

## Repository layout

The **existing A2 code stays at its current paths**. Add the A3 extension rather than renaming all older modules.

```text
README.md                         <- this project overview
[existing A2 Python files/folders] <- leave in place
extension/
  README.md                       <- A3 experiment documentation
  notebooks/dfdc-extension.ipynb  <- original recorded A3 notebook
  manifests/                     <- add sampled CSV manifests
  results/                       <- add saved summary result CSVs
  figures/                       <- add saved PNG plots (if available)
reports/
  assignment3_updated.tex        <- updated A3 report source
```

See [`extension/README.md`](extension/README.md) for exact artifact filenames and limitations. The notebook and reports alone do not contain the underlying private images or checkpoints.

## Scope changes from the A1 proposal

The A1 proposal considered **PGD, multiple ε settings, ViT-small, seven transformations, perceptual quality (PSNR/SSIM), and inference latency**. These were **not completed in A3**. The implemented study instead tests one FGSM strength, two transformations and four existing CNNs. The attack-state head was added in the A3 extension. Uncompleted proposal items are **future work**, not results.

## Base paper and credits

- Abbasi et al., *Comprehensive Evaluation of Deepfake Detection Models: Accuracy, Generalization, and Resilience to Adversarial Attacks*, *Applied Sciences* 15(3), 1225 (2025). https://doi.org/10.3390/app15031225
- DFDC dataset: Dolhansky et al., *The DeepFake Detection Challenge (DFDC) Dataset* (2020).
- FaceForensics++: Rössler et al., *FaceForensics++: Learning to Detect Manipulated Facial Images* (ICCV 2019).
- Attack method: Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (ICLR 2015).

**Status:** A2 completed; A3 experiments recorded. The repository documentation does **not** claim that all original A1 research objectives were completed.
