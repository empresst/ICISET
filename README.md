
```markdown
# Enhancing Prediction Accuracy using Ensemble Deep Learning Methods on Time Series Data

**Official Code Repository** for the paper presented at **ICISET 2024**

**Authors**: Fatema Ferdous Tamanna, Tamanna Akter Sonaly, Md Abdul Masud  
**Paper**: [ICISET 2024](https://ieeexplore.ieee.org/abstract/document/10939673) 

---

## 📋 Project Overview

This repository contains the complete implementation of our proposed **ensemble deep learning model** for time series forecasting. The model combines **BiLSTM + BiGRU + TCN** with **XGBoost** as the final meta-learner.

We achieved **state-of-the-art performance** on multiple real-world datasets including:
- Household Power Consumption (2M records)
- Industrial Load Consumption datasets
- Delhi Climate Temperature Prediction

## 🚀 Key Results

| Dataset                          | Best Model     | Accuracy | RMSE    | MAE    |
|----------------------------------|----------------|----------|---------|--------|
| Delhi Climate                    | Hybrid         | **94.70%** | 2.01    | 1.52   |
| Non-Metallic Mineral Industry    | Hybrid         | **88.29%** | 1.24    | 0.82   |
| Textile Industry                 | Hybrid         | **81.59%** | 1.20    | 0.85   |
| Household Electricity            | Hybrid         | **92.23%** | **0.207** | **0.076** |
| Plastic Industry                 | Hybrid         | **74.57%** | 13.76   | 10.00  |
| Paper Industry                   | Hybrid         | **90.11%** | 5.74    | 3.97   |

**Best performing model**: Proposed Hybrid Ensemble (BiLSTM + BiGRU + TCN + XGBoost)

---

## 🛠️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/empresst/ICISET.git
cd ICISET
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Place your dataset
ALL data files inside the `data/` folder:
- `etaysob.csv` or `ab.csv` (Household Power Consumption)

### 4. Train all models
```bash
python src/train.py
```

### 5. Evaluate results
```bash
python src/evaluate.py
```

---

## 📁 Project Structure

```
Ensemble-TimeSeries-Forecasting-ICISET2024/
├── src/
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation and plotting
│   ├── config.py
│   ├── models/               # All 4 models (CNN-GRU, GRU-FCN, L2-GRU, Hybrid)
│   └── utils/                # Data loading, preprocessing, metrics, visualization
├── notebooks/                # Original Jupyter notebooks
├── data/                     # Raw datasets
├── results/                  # Saved predictions (.npy files)
└── experiments/              # Scripts for different datasets
```

---

## 📚 Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{tamanna2024enhancing,
  title={Enhancing Prediction Accuracy using Ensemble Deep Learning Methods on Time Series Data},
  author={Tamanna, Fatema Ferdous and Sonaly, Tamanna Akter and Masud, Md Abdul},
  booktitle={2024 4th International Conference on Innovations in Science, Engineering and Technology (ICISET)},
  year={2024},
  organization={IEEE}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙋 Contact

**Fatema Ferdous Tamanna**  
Email: fatimatamannaah@gmail.com  
GitHub: [@fftamanna](https://github.com/empresst)

---

⭐ **Star this repository** if you find it useful!

---

**Made with ❤️ for research and reproducible science**
```

---

**How to use:**
1. Copy all the above code.
2. Paste it into `README.md` at the root folder.
3. Replace `yourusername` with your actual GitHub username.

Would you like me to make any changes to this README (e.g., add your exact GitHub link, co-author names, or paper DOI)? Just tell me.
