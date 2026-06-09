
<img width="1433" height="702" alt="Screenshot 2026-06-09 at 11 04 52 PM" src="https://github.com/user-attachments/assets/b0d71597-8a72-4d5f-a98a-84f7f0b98e89" />
# EpiRNA



# EpiRNA-Scanner
**Rapid, Length-Agnostic Mapping of m6A Motifs at Single-Nucleotide Resolution**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)](https://huggingface.co/spaces/supzammy/EpiRNAh)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**EpiRNA-Scanner** is an *in-silico* triage engine designed to rapidly map N6-methyladenosine (m6A) regulatory sites across entire transcriptomes. By leveraging sequence-derived biophysical embeddings and a dilated 1D Convolutional Neural Network (CNN) backbone, it bypasses the strict length limitations of traditional fixed-window models. 

The framework features a **Global-Local Hybrid Variance Stabilizer** to prevent division-by-zero inflation in low-entropy homopolymer regions, and **Epitranscriptomic Boundary Contrast Scoring (EBCS)** to identify catalytic target coordinates with single-nucleotide spatial precision.

##  Live Demo
You can run EpiRNA-Scanner directly in your browser with zero local installation or disk storage required via our deployed web interface:
👉 **[EpiRNA-Scanner on Hugging Face](https://huggingface.co/spaces/supzammy/EpiRNAh)**

---

## ✨ Key Features
* **Length-Agnostic Architecture:** Process raw FASTA sequences ranging from 50 bp to **>100,000 bp** without manual tiling or tensor shape mismatch errors. (Benchmarked 100kb+ processing in ~4.12 seconds).
* **Single-Nucleotide Resolution:** Extracts exact topological coordinates for the target Adenine within canonical `DRACH` motifs.
* **Homopolymer Stability:** Maintains a stable 0.0 baseline contrast in dense Poly-A/GC regions, eliminating false positives caused by local variance collapse.
* **Zero-Disk NCBI Streaming:** Fetch and score transcripts directly from NCBI Entrez accession numbers without downloading genome files to your local drive.

---

##  Installation

To run EpiRNA-Scanner locally for reproducibility testing:

1. **Clone the repository or extract the Zenodo archive:**
   ```bash
   git clone [https://github.com/YourUsername/EpiRNA-Scanner.git](https://github.com/YourUsername/EpiRNA-Scanner.git)
   cd EpiRNA-Scanner


2. ** Create a virtual environment (recommended):
     ```bash
     python -m venv epirna_env
     source epirna_env/bin/activate  # On Windows use: epirna_env\Scripts\activate

    
3. ** Install dependencies:
     ```bash
     pip install -r requirements.txt

   
5. ** Reproducibility & Testing
This repository includes the validation datasets used to benchmark the tool's biological gating and spatial resolution.

I. The U135C Mutation Stress-Test
Test the system's sensitivity to canonical sequence motif disruption by comparing a wild-type transcript against a point-mutated control.
     ```bash

     python epirna_engine.py --fasta test_data/MYC_wildtype.fasta
     python epirna_engine.py --fasta test_data/MYC_mutant_U135C.fasta

II. The Homopolymer Variance Test
Verify the Global-Local Hybrid Variance Stabilizer against a simulated sequence designed to trigger division-by-zero errors in standard algorithms.

     python epirna_engine.py --fasta test_data/TEST_01_HOMOPOLYMER_STABILITY.fasta



III. Direct Sequence Input
Run the tool directly from the command line using a raw sequence string:
    ```bash

     python epirna_engine.py --sequence "UCCGGCUCCGCUUCGGCGGACUCCGGCUUCGGC"


##  Architecture Pipeline (v1.0)
* Biophysical Embedding: Input sequences are initialized into a numerical tensor based on hydrogen-bond potential, aromatic stacking energy, and solvent accessibility.

* Dilated CNN Framework: Evaluates spatial homology for degenerate DRACH constraints.

* EBCS Derivative Scoring: Spatial derivatives compute a normalized contrast boundary, anchoring the prediction to the catalytic Adenine.

* Variance Stabilization: Modulates local scoring denominators against the global transcript entropy to suppress background noise.

**Note: Future iterations will integrate thermodynamic Minimum Free Energy (MFE) calculations via algorithms such as RNAfold to structurally penalize thermodynamically inaccessible motifs.**

##  Citation
** If you use EpiRNA-Scanner in your research, please cite our corresponding manuscript and this Zenodo repository:

** Zam, et al. (2026). "EpiRNA-Scanner: Rapid, Length-Agnostic Mapping of m6A Motifs at Single-Nucleotide Resolution." [Target Journal/Conference Name].
DOI: 10.5281/zenodo.XXXXXXX

## License

This project is licensed under the MIT License - see the MIT LICENSE file for details.



