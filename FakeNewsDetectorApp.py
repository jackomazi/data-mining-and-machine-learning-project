import tkinter as tk
from tkinter import ttk, messagebox
import requests
from bs4 import BeautifulSoup
import joblib
import webbrowser
import os
import re
import numpy as np
import scipy.sparse as sp
from lime.lime_text import LimeTextExplainer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker
from nltk.stem import WordNetLemmatizer

class FakeNewsDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FakeNews Detector")
        self.root.geometry("700x850")
        self.root.configure(bg="#f0f2f5")

        # Initialize NLTK and SpellChecker components
        self.init_nlp_tools()

        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color Palette
        self.primary_color = "#1a73e8"
        self.bg_color = "#f0f2f5"
        self.card_color = "#ffffff"
        
        # Cache for models (LIME needs them loaded in memory)
        self.vectorizer = None
        self.scaler = None  # MinMaxScaler for stylometry
        self.model = None
        self.last_analyzed_text = ""

        self.setup_styles()
        self.create_widgets()

    def init_nlp_tools(self):
        """Downloads NLTK resources and initializes the spellchecker and lemmatizer"""
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)

            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            
            custom_stopwords = [
                'said', 'mr', 'mrs', 'just', 'like', 'new', 'year', 'time', 'also', 'would', 'one',
                'trump', 'clinton', 'hillary', 'obama', 'donald', 'barack', 'president', 
                'state', 'states', 'government', 'house', 'white', 'republican', 'democrat', 'american',
                'reuters', 'washington', 'featured', 'image', 'images', 'getty', 'pic', 
                'twitter', 'com', 'via', 'fox', 'news', 'video', 'youtube'
            ]
            self.stop_words.update(custom_stopwords)
            
            self.spell = SpellChecker()
            
            self.lemmatizer = WordNetLemmatizer()
            
        except Exception as e:
            print(f"Warning: Issue initializing NLP tools: {e}")

    def setup_styles(self):
        self.style.configure("Header.TLabel", font=('Segoe UI', 18, 'bold'), background=self.bg_color, foreground="#202124")
        self.style.configure("SubHeader.TLabel", font=('Segoe UI', 12, 'bold'), background=self.card_color, foreground=self.primary_color)
        
        self.style.configure("Action.TButton", font=('Segoe UI', 11, 'bold'), foreground="white", background=self.primary_color, padding=10)
        self.style.map("Action.TButton", background=[('active', '#1557b0')])
        
        # Secondary button for LIME
        self.style.configure("Secondary.TButton", font=('Segoe UI', 10, 'bold'), foreground="#1a73e8", background=self.card_color, padding=8)
        self.style.configure("Card.TFrame", background=self.card_color, relief="flat")

    def create_widgets(self):
        # Header
        header_label = ttk.Label(self.root, text="🛡️ FakeNews Detector", style="Header.TLabel")
        header_label.pack(pady=(20, 15))

        main_container = ttk.Frame(self.root, style="Card.TFrame", padding=30)
        main_container.pack(fill="both", expand=False, padx=40, pady=(0, 20))

        # Online section (URL input)
        ttk.Label(main_container, text="ONLINE SOURCE", style="SubHeader.TLabel").pack(anchor="w")
        ttk.Label(main_container, text="Paste a news link to automatically extract content", font=('Segoe UI', 9), background=self.card_color, foreground="grey").pack(anchor="w", pady=(0, 10))
        
        self.url_entry = ttk.Entry(main_container, font=('Segoe UI', 11))
        self.url_entry.pack(fill="x", pady=(0, 20), ipady=8) 

        ttk.Separator(main_container, orient='horizontal').pack(fill='x', pady=20)

        # Offline section (manual input)
        ttk.Label(main_container, text="OFFLINE SOURCE", style="SubHeader.TLabel").pack(anchor="w")
        
        ttk.Label(main_container, text="Article Title", font=('Segoe UI', 10), background=self.card_color).pack(anchor="w", pady=(10, 5))
        self.title_entry = ttk.Entry(main_container, font=('Segoe UI', 11))
        self.title_entry.pack(fill="x", pady=(0, 15), ipady=5)

        ttk.Label(main_container, text="Article Body", font=('Segoe UI', 10), background=self.card_color).pack(anchor="w", pady=(5, 5))
        self.body_text = tk.Text(main_container, height=6, font=('Segoe UI', 10), relief="flat", highlightthickness=1, highlightbackground="#dadce0", padx=10, pady=10)
        self.body_text.pack(fill="x", pady=(0, 20))

        # Action button
        button_container = ttk.Frame(main_container, style="Card.TFrame")
        button_container.pack(fill="x")
        
        self.detect_btn = ttk.Button(button_container, text="RUN ANALYSIS", style="Action.TButton", command=self.run_detection)
        self.detect_btn.pack(anchor="center")

        # Result display
        result_container = tk.Frame(self.root, bg=self.bg_color)
        result_container.pack(fill="both", expand=True)

        self.result_label = ttk.Label(result_container, text="Waiting for input...", font=('Segoe UI', 14, 'bold'), background=self.bg_color, foreground="#5f6368")
        self.result_label.pack(pady=(10, 10))

        # LIME explanation button (hidden until analysis is done)
        self.explain_btn = ttk.Button(result_container, text="VIEW LIME EXPLANATION", style="Secondary.TButton", command=self.show_lime_explanation, state="disabled")
        self.explain_btn.pack(pady=(0, 20))

    def extract_text_from_url(self, url):
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() 
            soup = BeautifulSoup(response.text, 'html.parser')

            title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No Title Found"
            paragraphs = soup.find_all('p')
            body_content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])

            if not body_content.strip():
                raise ValueError("Loaded page, but no significant text content was found.")
            return title, body_content
        except Exception as e:
            raise Exception(f"Failed to scrape URL: {e}")

    def load_models(self):
        """Loads vectorizer, scaler, and model into memory once"""
        if self.vectorizer is None or self.model is None or self.scaler is None:
            try:
                self.vectorizer = joblib.load(r'saved_models\tfidf_vectorizer.pkl')
                self.scaler = joblib.load(r'saved_models\stylometric_scaler.pkl') 
                self.model = joblib.load(r'saved_models\lightgbm_tuned.pkl')
            except Exception as e:
                raise Exception(f"Failed to load models: {e}")

    # Text cleaning and feature extraction methods

    def clean_text(self, text):
        """Applies NLTK and Regex cleaning steps for the TF-IDF vectorizer"""
        # Noise Removal (HTML, URLs)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Case Folding
        text = text.lower()
        
        # Tokenization 
        tokens = word_tokenize(text)
        
        # Lemmatization
        lemmas = [self.lemmatizer.lemmatize(token) for token in tokens]
    
        # Stopword elimination + removes everything that is not a letter, number or space
        cleaned_tokens = []
        for w in lemmas:
            clean_w = re.sub(r'[^\w\s]', '', w)
            if clean_w and clean_w not in self.stop_words:
                cleaned_tokens.append(clean_w)

        
        return " ".join(cleaned_tokens)

    def count_spelling_errors(self, text):
        """Calculates the ratio of spelling errors in a text"""
        if not isinstance(text, str) or len(text) == 0:
            return 0
        # Divide into words
        words = re.findall(r'\b\w+\b', text.lower())
        if len(words) == 0:
            return 0
        
        misspelled = self.spell.unknown(words)
        return len(misspelled) / len(words)

    def extract_stylometric_features(self, combined_text):
        """
        Extracts stylometric features. Expects Title and Body to be separated by '|||',
        if '|||' is missing (e.g., LIME perturbation), it treats the whole text as body
        """
        parts = combined_text.split("|||", 1)
        if len(parts) == 2:
            title, text_body = parts
        else:
            title, text_body = "", combined_text

        # length of the text
        text_length = len(text_body)
        
        # uppercase chars ratio in the title
        title_uppercase_ratio = sum(1 for c in title if c.isupper()) / len(title) if len(title) > 0 else 0
        
        # ratio of exclamation point and question mark in the title
        title_exclamation_ratio = (title.count('!') + title.count('?')) / len(title) if len(title) > 0 else 0
        
        # ratio of exclamation point and question mark in the text
        text_exclamation_ratio = (text_body.count('!') + text_body.count('?')) / len(text_body) if len(text_body) > 0 else 0
        
        # spelling errors ratio (calculated on full combined text)
        full_text = title + " " + text_body
        spelling_errors_ratio = self.count_spelling_errors(full_text)
        
        # Return as 2D array for the scaler
        return np.array([[text_length, title_uppercase_ratio, title_exclamation_ratio, text_exclamation_ratio, spelling_errors_ratio]])

    # Prediction and pipeline
    def make_prediction(self, combined_text):
        try:
            self.load_models()
            
            # Remove the separator to get standard text for cleaning
            full_text_for_cleaning = combined_text.replace("|||", " ")
            cleaned_text = self.clean_text(full_text_for_cleaning)
            
            # Transform cleaned text using TF-IDF (Sparse Matrix)
            vectorized_text = self.vectorizer.transform([cleaned_text])
            
            # Extract and scale stylometric features (Dense Matrix)
            raw_stylo_features = self.extract_stylometric_features(combined_text)
            scaled_stylo_features = self.scaler.transform(raw_stylo_features)
            
            # Combine TF-IDF and Stylometric features safely
            combined_features = sp.hstack([vectorized_text, scaled_stylo_features])
            
            # Predict
            prediction = self.model.predict(combined_features)
            
            return "FAKE NEWS" if prediction[0] == 0 else "REAL NEWS"
        except Exception as e:
            return f"Error: {e}"

    def run_detection(self):
        url_val = self.url_entry.get().strip()

        # Disable LIME button while running
        self.explain_btn.config(state="disabled")

        if url_val:
            self.result_label.config(text="Scraping content...", foreground="orange")
            self.root.update()
            try:
                scraped_title, scraped_body = self.extract_text_from_url(url_val)
                
                self.title_entry.delete(0, tk.END)
                self.title_entry.insert(0, scraped_title)
                
                self.body_text.delete("1.0", tk.END)
                self.body_text.insert("1.0", scraped_body)
                
                title_to_analyze = scraped_title
                body_to_analyze = scraped_body
            except Exception as e:
                messagebox.showerror("Scraping Error", str(e))
                self.result_label.config(text="Result: Error", foreground="red")
                return
        else:
            title_to_analyze = self.title_entry.get().strip()
            body_to_analyze = self.body_text.get("1.0", tk.END).strip()

        if not title_to_analyze and not body_to_analyze:
            messagebox.showwarning("Input Missing", "Please provide a URL or fill in the Title/Body.")
            return

        self.result_label.config(text="Analyzing...", foreground="orange")
        self.root.update()

        # We combine Title and Body using a unique separator "|||"
        # This allows extract_stylometric_features to distinguish them, 
        # while LIME can still treat it as a single string document
        combined_text = f"{title_to_analyze}|||{body_to_analyze}"
        self.last_analyzed_text = combined_text
        
        final_result = self.make_prediction(combined_text)

        if "Error" not in final_result:
            color = "#d93025" if "FAKE" in final_result else "#188038" # Red vs Green
            self.result_label.config(text=f"{final_result}", foreground=color)
            self.explain_btn.config(state="normal")
        else:
            self.result_label.config(text=final_result, foreground="red")

    # Lime explanation method
    def predict_proba_pipeline(self, texts):
        """
        Helper function for LIME. Takes raw texts, transforms them 
        (Cleaning and then TF-IDF + Scaled Stylometry), and returns probabilities
        """
        # Clean texts (remove the "|||" separator before cleaning)
        cleaned_texts = [self.clean_text(t.replace("|||", " ")) for t in texts]
        
        # Vectorize text (Sparse)
        vectorized_texts = self.vectorizer.transform(cleaned_texts)
        
        # Extract stylometric features for all texts generated by LIME
        raw_stylo_list = [self.extract_stylometric_features(t)[0] for t in texts]
        raw_stylo_array = np.array(raw_stylo_list)
        
        # Scale the batch of stylometric features
        scaled_stylo_array = self.scaler.transform(raw_stylo_array)
        
        # Combine features
        combined_features = sp.hstack([vectorized_texts, scaled_stylo_array])
        
        # Return prediction probabilities
        return self.model.predict_proba(combined_features)

    def show_lime_explanation(self):
        if not self.last_analyzed_text:
            return

        self.explain_btn.config(text="GENERATING REPORT...", state="disabled")
        self.root.update()

        try:
            explainer = LimeTextExplainer(class_names=['FAKE NEWS', 'REAL NEWS'])
            
            exp = explainer.explain_instance(
                self.last_analyzed_text, 
                self.predict_proba_pipeline, 
                num_features=15 
            )
            
            html_file_path = os.path.abspath("lime_report.html")
            exp.save_to_file(html_file_path)
            webbrowser.open(f"file://{html_file_path}")
            
        except AttributeError:
            messagebox.showerror("Model Error", "Your model does not support predict_proba().")
        except Exception as e:
            messagebox.showerror("LIME Error", f"Failed to generate explanation: {str(e)}")
        finally: 
            self.explain_btn.config(text="VIEW LIME EXPLANATION", state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeNewsDetectorApp(root)
    root.mainloop()