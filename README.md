# Fake News Detector

Developed for the Data Mining and Machine Learning course, 
Master’s Degree in Artificial Intelligence and Data Engineering, 
University of Pisa, A.Y. 2025–2026.

Desktop application for fake news classification using Machine Learning and automated web scraping.
Developed for the Data Mining and Machine Learning course, A.Y. 2025-2026. 
**Authors:** [Alessio Grillo](https://github.com/agrillo6-prog), [Andrea Giacomazzi](https://github.com/jackomazi)

---

## Project Overview

The application allows users to evaluate the credibility of online news articles through two different analysis modes:

* **Online Detection** – Analyze a news article directly from its URL.
* **Offline Detection** – Analyze manually provided article title and content.

When a URL is provided, the application automatically extracts the article text and prioritizes the scraped content over manually entered text.

The system supports multiple Machine Learning models, enabling users to compare classification results across different algorithms.

---

## Features

### Automated Web Scraping

The application retrieves article content directly from news websites using `Requests` and `BeautifulSoup4`, extracting:

* Article title
* Main article body

Irrelevant elements such as advertisements, navigation menus, and sidebars are filtered out whenever possible.

### Machine Learning Classification

The following models are available:

| Model                        |
| ---------------------------- |
| Multinomial Naive Bayes      |
| Support Vector Machine (SVM) |
| Logistic Regression          |
| AdaBoost                     |
| Random Forest                |
| LightGBM                     |

### User Interface

Built with Tkinter, the graphical interface provides:

* URL-based article analysis
* Manual text input
* Scraped content preview
* Model selection
* Color-coded prediction results

---

## Technical Stack

| Component           | Technology               |
| ------------------- | ------------------------ |
| Language            | Python 3                 |
| GUI                 | Tkinter (TTK)            |
| Web Scraping        | Requests, BeautifulSoup4 |
| Machine Learning    | Scikit-learn, LightGBM   |
| Model Serialization | Joblib                   |

---

## Repository Structure

```text
.
├── FakeNewsDetectorApp.py
├── models/
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

Launch the application with:

```bash
python FakeNewsDetectorApp.py
```

---

## Usage

### Analyze an Online Article

1. Paste a news article URL into the **URL** field.
2. Select a classification model.
3. Click **Run Detection**.

The application will automatically scrape the article content and display the prediction.

### Analyze Custom Text

1. Leave the **URL** field empty.
2. Enter the article **Title** and **Body** manually.
3. Select a classification model.
4. Click **Run Detection**.

The selected model will classify the provided text as **Fake News** or **Real News**.

---

## Notes

* URL analysis always takes precedence over manually entered text.
* Different models may produce different predictions depending on the article content.
* Classification performance depends on the quality and representativeness of the training dataset.
