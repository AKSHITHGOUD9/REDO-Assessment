# REDO Assessment: Anti-Reproductive Rights Crimes & Domestic Violence Related Calls Analysis

---

## 🚀 Overview

This project delivers a rigorous, ethical, and reproducible analysis of anti-reproductive rights crimes and domestic violence in California. It uses advanced data science techniques (EDA, clustering, classification) and is fully dockerized for **one-click execution**—no manual downloads or path changes required.

- **Detailed explanations and step-by-step analysis for each part can be found in:**

 - `part-1.ipynb` (Anti-Reproductive Rights Crimes)
 - `part-2.ipynb` (Domestic Violence Calls)

- **Interactive Visual Analysis:**
 - The project includes a Streamlit web app for interactive, visual data exploration. Using Docker, you can launch a web interface to:
 - Upload and explore datasets
 - Assess data quality and missingness
 - Perform exploratory data analysis (EDA)
 - Run clustering and classification models
 - Visualize results (PCA plots, feature importances, confusion matrices, etc.)
 - Review ethical considerations and references
 - This makes the project accessible and visually engaging for both technical and non-technical users.

> **Prerequisite:** 
> [Docker](https://www.docker.com/products/docker-desktop/) **must be installed on your system** to run this project as described. Download Docker Desktop for [Windows/Mac](https://www.docker.com/products/docker-desktop/) or [get Docker for Linux](https://docs.docker.com/engine/install/).

---

## Quick Start: One-Click via Docker

1. **Clone this repository:**
 ```sh
 git clone https://github.com/yourusername/REDO-Assessment.git
 cd REDO-Assessment
 ```
2. **Build and run the Docker container:**
 ```sh
 docker build -t redo-assessment .
 docker run -p 8502:8502 redo-assessment
 ```
3. **Open your browser and go to:**
 ```
 http://localhost:8502
 ```
 _(Copy and paste or cmd/ctrl+click the link above to open it in your browser.)_
 
4. **Open and run the notebooks:**
 - `part-1.ipynb` (Anti-Reproductive Rights Crimes)
 - `part-2.ipynb` (Domestic Violence Calls)

---

## 📁 Project Structure

```
REDO-Assessment/
├── Dockerfile
├── requirements.txt
├── README.md
├── part-1.ipynb
├── part-2.ipynb
├── app/
│ ├── main.py
│ ├── config.py
│ ├── utils.py
│ ├── models.py
│ ├── Anti-Reproductive Data.csv
│ └── Domestic violence Data.csv
```

---

## 📊 Usage & Analysis

- **Notebooks:**
 - `part-1.ipynb`: Anti-reproductive rights crimes analysis (EDA, clustering, classification, ethics)
 - `part-2.ipynb`: Domestic violence calls analysis (EDA, trends, modeling ideas)
- **Scripts:**
 - `main.py`, `utils.py`, `models.py`, `config.py`: Modular code for advanced analysis and reproducibility
- **Data:**
 - `Anti-Reproductive Data.csv`, `Domestic violence Data.csv` (included in the root folder)

---

## 📝 Notes

- **All code uses relative paths** (e.g., `Anti-Reproductive Data.csv`)
- **All data is included**—no external downloads needed
- **Tested for top-to-bottom execution in Jupyter**

---

## 📬 Contact

For questions or support, contact:

**Akshith** 
[akshithgoud4499@gmail.com](mailto:akshith)

---

> _This project was developed as part of the REDO Assessment. It demonstrates best practices in ethical, responsible, and reproducible data science._
