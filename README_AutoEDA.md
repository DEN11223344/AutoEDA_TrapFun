# 📊 Automated EDA & Preprocessing Tool  

A **Streamlit-based web app** that automates **Exploratory Data Analysis (EDA)** and preprocessing of datasets. Users can upload a CSV file, and the app will instantly generate data insights, visualizations, and preprocessing suggestions to help in building machine learning models.  

---

## 🚀 Features  

- 📂 **Upload CSV Dataset** (any structured dataset)  
- 🔍 **Data Overview** – Summary statistics, missing values, and column types  
- 📊 **Data Visualization** – Distribution plots, correlations, and relationships between features  
- 🧹 **Preprocessing Suggestions** – Label Encoding, Scaling, Missing Value handling  
- 🧠 **ML Readiness** – Suggestions on which ML algorithms might work best  
- 🎨 **Interactive Dashboard** built with Streamlit & Seaborn  

---

## 📂 Project Structure  

```
Auto-EDA-Tool/
│── app.py                 # Main Streamlit app
│── requirements.txt       # Dependencies
│── README.md              # Project documentation
│── sample_datasets/       # Example datasets (optional)
```

---

## ⚙️ Installation & Setup  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/auto-eda-tool.git
   cd auto-eda-tool
   ```

2. **Create virtual environment (recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate   # (Linux/Mac)
   venv\Scripts\activate      # (Windows)
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**  
   ```bash
   streamlit run app.py
   ```

---

## 📊 Example Workflow  

1. Upload a dataset (`.csv`)  
2. Instantly view **data summary** (rows, columns, null values)  
3. Explore **EDA visualizations** (histograms, scatter plots, correlations)  
4. Get **preprocessing suggestions** like:  
   - Missing value imputation  
   - Categorical encoding (Label/One-hot)  
   - Feature scaling  
5. Get **recommended ML models** based on dataset type  

---

## 🎨 UI Screenshots  

(Add screenshots/gifs of your dashboard here)  

---

## 🔮 Future Enhancements  

- ✅ Support for Excel files (`.xlsx`)  
- ✅ Auto ML training & evaluation  
- ✅ Export processed dataset  
- ✅ Model comparison charts  

---

## 🤝 Contributing  

Contributions are welcome!  
1. Fork the repo  
2. Create a new branch (`feature-branch`)  
3. Commit changes & push  
4. Open a Pull Request  

---

## 📜 License  

This project is licensed under the MIT License. See `LICENSE` for details.  

---

## 👨‍💻 Author  

- **Piyush Balode** – MSc Data Science Student, Fergusson College, Pune  
- 📧 [Your Email]  
- 🔗 [LinkedIn / Portfolio link]  
