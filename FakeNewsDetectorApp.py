import tkinter as tk
from tkinter import ttk, messagebox
import requests
from bs4 import BeautifulSoup
import joblib
import webbrowser
import os
from lime.lime_text import LimeTextExplainer

class FakeNewsDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FakeNews Detector")
        self.root.geometry("700x850")
        self.root.configure(bg="#f0f2f5")

        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color Palette
        self.primary_color = "#1a73e8"
        self.bg_color = "#f0f2f5"
        self.card_color = "#ffffff"
        
        # Cache for model and vectorizer (LIME needs them loaded in memory)
        self.vectorizer = None
        self.model = None
        self.last_analyzed_text = ""

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        self.style.configure("Header.TLabel", font=('Segoe UI', 18, 'bold'), background=self.bg_color, foreground="#202124")
        self.style.configure("SubHeader.TLabel", font=('Segoe UI', 12, 'bold'), background=self.card_color, foreground=self.primary_color)
        
        self.style.configure("Action.TButton", font=('Segoe UI', 11, 'bold'), foreground="white", background=self.primary_color, padding=10)
        self.style.map("Action.TButton", background=[('active', '#1557b0')])
        
        # Secondary Button for LIME
        self.style.configure("Secondary.TButton", font=('Segoe UI', 10, 'bold'), foreground="#1a73e8", background=self.card_color, padding=8)

        self.style.configure("Card.TFrame", background=self.card_color, relief="flat")

    def create_widgets(self):
        # HEADER
        header_label = ttk.Label(self.root, text="🛡️ FakeNews Detector", style="Header.TLabel")
        header_label.pack(pady=(20, 15))

        main_container = ttk.Frame(self.root, style="Card.TFrame", padding=30)
        main_container.pack(fill="both", expand=False, padx=40, pady=(0, 20))

        # ONLINE SECTION (URL)
        ttk.Label(main_container, text="ONLINE SOURCE", style="SubHeader.TLabel").pack(anchor="w")
        ttk.Label(main_container, text="Paste a news link to automatically extract content", font=('Segoe UI', 9), background=self.card_color, foreground="grey").pack(anchor="w", pady=(0, 10))
        
        self.url_entry = ttk.Entry(main_container, font=('Segoe UI', 11))
        self.url_entry.pack(fill="x", pady=(0, 20), ipady=8) 

        ttk.Separator(main_container, orient='horizontal').pack(fill='x', pady=20)

        # OFFLINE SECTION (MANUAL INPUT)
        ttk.Label(main_container, text="OFFLINE SOURCE", style="SubHeader.TLabel").pack(anchor="w")
        
        ttk.Label(main_container, text="Article Title", font=('Segoe UI', 10), background=self.card_color).pack(anchor="w", pady=(10, 5))
        self.title_entry = ttk.Entry(main_container, font=('Segoe UI', 11))
        self.title_entry.pack(fill="x", pady=(0, 15), ipady=5)

        ttk.Label(main_container, text="Article Body", font=('Segoe UI', 10), background=self.card_color).pack(anchor="w", pady=(5, 5))
        self.body_text = tk.Text(main_container, height=6, font=('Segoe UI', 10), relief="flat", highlightthickness=1, highlightbackground="#dadce0", padx=10, pady=10)
        self.body_text.pack(fill="x", pady=(0, 20))

        # ACTION BUTTON
        button_container = ttk.Frame(main_container, style="Card.TFrame")
        button_container.pack(fill="x")
        
        self.detect_btn = ttk.Button(button_container, text="RUN ANALYSIS", style="Action.TButton", command=self.run_detection)
        self.detect_btn.pack(anchor="center")

        # RESULT DISPLAY
        result_container = tk.Frame(self.root, bg=self.bg_color)
        result_container.pack(fill="both", expand=True)

        self.result_label = ttk.Label(result_container, text="Waiting for input...", font=('Segoe UI', 14, 'bold'), background=self.bg_color, foreground="#5f6368")
        self.result_label.pack(pady=(10, 10))

        # LIME EXPLANATION BUTTON (Hidden until analysis is done)
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

    def load_models_if_needed(self):
        """Loads models into memory once to speed up LIME and predictions."""
        if self.vectorizer is None or self.model is None:
            try:
                self.vectorizer = joblib.load('vectorizer.pkl')
                self.model = joblib.load('random_forest_model.pkl')
            except Exception as e:
                raise Exception(f"Failed to load models: {e}")

    def make_prediction(self, text):
        try:
            self.load_models_if_needed()
            vectorized_text = self.vectorizer.transform([text])
            prediction = self.model.predict(vectorized_text)
            
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

        combined_text = f"{title_to_analyze} {body_to_analyze}"
        self.last_analyzed_text = combined_text
        
        final_result = self.make_prediction(combined_text)

        if "Error" not in final_result:
            color = "#d93025" if "FAKE" in final_result else "#188038" # Red vs Green
            self.result_label.config(text=f"{final_result}", foreground=color)
            # Enable the LIME button since analysis was successful
            self.explain_btn.config(state="normal")
        else:
            self.result_label.config(text=final_result, foreground="red")

    # LIME
    def predict_proba_pipeline(self, texts):
        """Helper function for LIME. Takes raw texts, transforms them, and returns probabilities."""
        vectorized_texts = self.vectorizer.transform(texts)
        return self.model.predict_proba(vectorized_texts)

    def show_lime_explanation(self):
        if not self.last_analyzed_text:
            return

        # Show user that the report is being generated
        self.explain_btn.config(text="GENERATING REPORT...", state="disabled")
        self.root.update()

        try:
            # Initialize LIME explainer
            explainer = LimeTextExplainer(class_names=['FAKE NEWS', 'REAL NEWS'])
            
            # Generate Explanation
            exp = explainer.explain_instance(
                self.last_analyzed_text, 
                self.predict_proba_pipeline, 
                num_features=15 # Top 15 words
            )
            
            # Save to a temporary HTML file
            html_file_path = os.path.abspath("lime_report.html")
            exp.save_to_file(html_file_path)
            
            # Open the HTML file in the default web browser
            webbrowser.open(f"file://{html_file_path}")
            
        except AttributeError:
            messagebox.showerror("Model Error", "Your model does not support predict_proba(). Ensure you are using a model that outputs probabilities (like Random Forest or Logistic Regression).")
        except Exception as e:
            messagebox.showerror("LIME Error", f"Failed to generate explanation: {str(e)}")
        finally: 
            # Reset button
            self.explain_btn.config(text="VIEW LIME EXPLANATION", state="normal")

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeNewsDetectorApp(root)
    root.mainloop()