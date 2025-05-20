# 3D-Cascaded-NN-Code
This project aims to develop a hierarchical deep learning framework based on structural brain MRI, primarily for disease diagnosis classification such as Alzheimer's disease (AD). The framework incorporates three types of loss functions:

🧮 Regression Loss – for continuous cognitive score prediction

🧠 Classification Loss – for distinguishing subject classes (e.g., AD vs NC)

🔗 Fusion Loss – for integrating features across multiple tasks/models

📁 Project Structure

```

├── main_三个模型_loss1_loss2_loss3_改进.py       # Main training script (3 models with hierarchical losses)
├── 融合改进4.py                                # Fusion module (combines scores and features)
├── test_拆开三个_改进融合.py                   # Evaluation script (separate and fused model testing)
├── dataloader_1.py                             # Data loading 
├── config.py                                   # Configurations (paths, hyperparameters, etc.)
└── README.md                                   # Project documentation


````

---

📦 Requirements

- Python 3.8+
- PyTorch >= 1.10
- numpy
- nibabel
- pandas
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
````

---

📊 Data Preparation

* MRI NIfTI files (`.nii` or `.nii.gz`) should be placed in a folder like:

```
./data/24train/AD/
```

