# 🛡️ FakeNews Detector

A desktop application designed to help users identify the credibility of news articles. By leveraging Machine Learning and real-time web scraping, this tool provides a fast and intuitive way to cross-check information.

---

## 🚀 Overview
This application features a dual-mode detection system:
1.  **Online Detection:** Input a URL, and the app will automatically scrape the title and article body for analysis.
2.  **Offline Detection:** Manually input text (Title & Body) to check specific snippets or offline content.

**Priority Logic:** If a URL is provided, the application will prioritize scraping that link, overriding any text manually entered in the offline fields.

---

## ✨ Key Features
* **Automated Scraping:** Uses `BeautifulSoup4` to extract clean text from news websites, ignoring ads and sidebars.
* **Live Preview:** Scraped content is automatically populated into the UI fields so you can verify what is being analyzed.
* **Multi-Model Support:** Compare results across 6 different algorithms:
    * Multinomial Naive Bayes
    * Support Vector Machine (SVM)
    * Logistic Regression
    * AdaBoost
    * Random Forest
    * LightGBM
* **Clear Verdicts:** Color-coded results (**Fake News** / **Real News**) for immediate clarity.

---

## 🛠️ Technical Stack
* **Language:** Python 3.x
* **GUI Framework:** Tkinter (TTK)
* **Web Scraping:** Requests & BeautifulSoup4
* **Machine Learning:** Scikit-learn, LightGBM, and Joblib for model serialization.

---

## Setup

2.  **Install Required Libraries:**
    ```bash
    pip install -r requirements.txt    
    ```


## 🚦 How to Use

1.  **Launch the App:**
    ```bash
    python FakeNewsDetectorApp.py
    ```
2.  **Analyze a Website:** Paste a link into the **URL** field and click **Run Detection**.
3.  **Analyze Manually:** Ensure the URL field is empty, type the **Title** and **Body** in the respective boxes, select a model, and click **Run Detection**.

---
