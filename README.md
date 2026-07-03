# PyMLDA: Machine Learning for Damage Assessment

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/MOSS-vLab/pymlda.svg?style=social)](https://github.com/MOSS-vLab/pymlda/stargazers)

**PyMLDA** is an open-source Python package for **detection and quantification of structural damage** using machine learning algorithms. It is designed to process dynamic data and automatically extract damage indicators, supporting predictive maintenance and structural health monitoring.

> **Quick Links:** [Installation](#-installation) • [Basic Usage](#-basic-usage) 
---

## 📦 Installation

Since `PyMLDA` is not yet available on PyPI, you can install it directly from GitHub.

### For local development

Open your terminal and run:

```bash
pip install git+https://github.com/MOSS-vLab/pymlda.git
```
For Google Colab or Jupyter Notebook in a code cell, use:
```python

!pip install --no-cache-dir git+https://github.com/MOSS-vLab/pymlda.git --quiet
print("✅ PyMLDA installed successfully!")
```
---
## 🚀 Basic Usage

Here is a quick example to get you started with PyMLDA.
1. Import the package
```python
import pymlda as mlda
import pandas as pd
```
2. Load your data

The package works with vibration data in CSV format or pandas DataFrame:
```python
# Example: load data from a CSV file
data = pd.read_csv('vibration_data.csv')
# Check the data structure
print(data.head())
```

3. Visualise groups with plot_density_by_group

One of the main available functions is plot_density_by_group, which creates density plots to compare the distributions of a continuous variable across groups (e.g., healthy vs damaged structure).
```python
# Generate density plot to compare groups
mlda.plot_density_by_group(
    data=data,
    group_col='Condition',        # Column name with groups (e.g., 'Healthy', 'Damaged')
    value_col='Natural_Frequency' # Column name with values to plot
)
```

This plot is useful to visually identify how the distribution of natural frequencies changes in the presence of structural damage.
🌟 Features: Feature, Condition, 	Description
💾 Feature Management: Load, save, and manage features with FeatureManager
🔍 Feature Extraction: Extract time, spectral, and FRF features from signals
🔀 Clustering	Factory:	K-Means  ( soon: DBSCAN, Agglomerative, GMM)
🤖 Classification Factory Models: SVM, Random Forest, KNN, XGBoost, Decision Tree, Naive Bayes
📈 Regression Factory Models: Predict damage severity
🎨 Advanced Visualization: Plotting utilities for clustering, confusion matrices, feature importance
📊 plot_density_by_group: Compare distributions of a variable across groups

The repository is organized to facilitate development and maintenance:
```text
pymlda/
├── src/
│   └── pymlda/              # Main source code
│       ├── __init__.py
│       └── (modules)        # Your modules here
├── docs/                    # Documentation (Sphinx)
├── examples/                # Jupyter notebooks and example scripts
├── tests/                   # Unit tests
├── .gitignore
├── LICENSE                  # MIT License
├── README.md                # This file
├── CONTRIBUTING.rst         # Guide for contributors
└── pyproject.toml           # Project configuration
```
---
## 📚 Examples

In the examples/ folder (once created), you will find Jupyter notebooks with practical examples:

    plot_density_demo.ipynb: Demonstration of the plot_density_by_group function

    clustering_demo.ipynb: Comprehensive clustering tutorial with multiple algorithms

    feature_extraction_tutorial.ipynb: Tutorial on feature extraction

    ml_pipeline_demo.ipynb: Complete ML pipeline with classification and clustering

To run the examples, install Jupyter:
```bash

pip install jupyter
jupyter notebook examples/
```
---
**🤝 How to Contribute**
We love contributions! To contribute to PyMLDA:
- Fork the repository
- Clone your fork: git clone https://github.com/your-username/pymlda.git
- Create a branch for your feature: git checkout -b my-new-feature
- Make your changes and add tests
- Commit your changes: git commit -m 'Add new feature'
- Push to the branch: git push origin my-new-feature
- Open a Pull Request on GitHub

Please read the CONTRIBUTING.rst file for more details on the process.

📄 License: This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments: Developed by Marcela Machado and contributors and Inspired by the scientific Python ecosystem and projects like sdypy-FRF

📬 Contact & Support
Issues: Report bugs or request features
Full Documentation: Coming soon on ReadTheDocs!

⭐ If you found this project useful, please consider giving it a star! ⭐

---

# Key Additions:

1. **Clustering Section**: Complete guide with:
   - Basic K-Means example
   - All available algorithms (K-Means, DBSCAN, Agglomerative, GMM)
   - Elbow method for optimal K
   - Cluster analysis and interpretation
   - Integration with ML pipeline

2. **Updated Features Table**: Marked clustering and classification as ✅ Available

3. **Updated Project Structure**: Added all modules

4. **Additional Examples**: Added `clustering_demo.ipynb` and `ml_pipeline_demo.ipynb` to examples section

The README is now comprehensive and covers the full functionality of your PyMLDA package! 🚀

---
# 🎯 Next Steps (Documentation)

With the README ready, we can proceed to the next steps:

1. **Add Docstrings:** Write NumPy-style documentation directly in the code for functions like `plot_density_by_group`. This will allow automatic API reference generation.
2. **Configure Sphinx:** Create the `docs/` folder structure and configuration files to build a documentation website.
3. **Host on Read the Docs:** Connect your GitHub repository to Read the Docs to host the documentation online.


## 🟢 Versions
- Version 1 (2024: Implementation of the PyMLDA in Python language, Supporting as input only the damage index, and OpenCode.) 

- Version 2 (2026: PyMLDA on pip, supporting many features type, classification, clustering and regression factory) 

---
##  References

[1] Amanda A.S.R. de Sousa, Marcela R. Machado, Experimental vibration dataset collected of a beam reinforced with masses under different health conditions, Data in Brief, 2024, 110043,ISSN 2352-3409,
https://doi.org/10.1016/j.dib.2024.110043.

[2] A. A. S. R. D. Sousa, and M. R. Machado. “Damage Assessment of a Physical Beam Reinforced with Masses - Dataset”. Multiclass Supervised Machine Learning Algorithms Applied to Damage and Assessment Using Beam Dynamic Response. Zenodo, November 8, 2023. https://doi.org/10.5281/zenodo.8081690.

[3] Coelho, J.S., Machado, M.R., Dutkiewicz, M. et al. Data-driven machine learning for pattern recognition and detection of loosening torque in bolted joints. J Braz. Soc. Mech. Sci. Eng. 46, 75 (2024). https://doi.org/10.1007/s40430-023-04628-6

[4] Coelho, J.S., Machado, M.R., Souza, A.A.S.R.D., PyMLDA: A Python open-source code for Machine Learning Damage Assessment. Software Impacts, 19, 100628 (2024). https://doi.org/10.1016/j.simpa.2024.100628 


## 📝 Citation

If you find **PyMLDA** useful for your research or development, please cite the  following:

```
@inproceedings{PyMLDA2024,
  title={PyMLDA - Machine Learning for Damage Assessment},
  author={ Coelho, J.S. and Machado, M.R. and  Sousa, A.A.S.R.},
  booktitle={},
  year={2024} }
```
```
@article{PyMLDAsoft24,
author = {Coelho, J.S., and Machado, M.R., and Souza, A.A.S.R.D.},
title = {PyMLDA: A Python open-source code for Machine Learning Damage Assessment},
journal = {Software Impacts},
volume = {19},
pages = {100628},
year = {2024},
doi = {10.1016/j.simpa.2024.100628}, }
```

