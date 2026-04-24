import tkinter as tk
from tkinter import ttk, messagebox
import requests
from bs4 import BeautifulSoup
import joblib

class FakeNewsDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FakeNews Detector")
        self.root.geometry("600x750")
        self.root.resizable(False, False)

        style = ttk.Style()
        style.theme_use('clam')

        self.create_widgets()

    def create_widgets(self):
        # ONLINE SECTION (URL)
        ttk.Label(self.root, text="Online Detection", font=('Helvetica', 12, 'bold')).pack(pady=(20, 5), anchor='w', padx=25)
        
        url_frame = ttk.Frame(self.root)
        url_frame.pack(fill='x', padx=25)
        ttk.Label(url_frame, text="URL:").pack(side='left')
        self.url_entry = ttk.Entry(url_frame, width=50)
        self.url_entry.pack(side='left', padx=10, fill='x', expand=True)

        ttk.Separator(self.root, orient='horizontal').pack(fill='x', pady=20, padx=25)

        # OFFLINE SECTION (Manual Input)
        ttk.Label(self.root, text="Offline Detection", font=('Helvetica', 12, 'bold')).pack(pady=5, anchor='w', padx=25)
        
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill='x', padx=25, pady=5)
        ttk.Label(title_frame, text="Title:").pack(side='left')
        self.title_entry = ttk.Entry(title_frame, width=50)
        self.title_entry.pack(side='left', padx=10, fill='x', expand=True)

        body_frame = ttk.Frame(self.root)
        body_frame.pack(fill='both', expand=True, padx=25, pady=5)
        ttk.Label(body_frame, text="Article Body:").pack(anchor='w')
        self.body_text = tk.Text(body_frame, height=10, width=50, wrap='word', font=('Helvetica', 10))
        self.body_text.pack(fill='both', expand=True, pady=5)

        ttk.Separator(self.root, orient='horizontal').pack(fill='x', pady=20, padx=25)

        # MODEL SELECTION AND BUTTON
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill='x', padx=25, pady=10)

        ttk.Label(action_frame, text="Model:").pack(side='left')
        
        models = ["Multinomial Naive Bayes", "SVM", "Logistic Regression", "AdaBoost", "Random Forest", "LightGBM"]
        self.model_combobox = ttk.Combobox(action_frame, values=models, state="readonly", width=25)
        self.model_combobox.set(models[0]) 
        self.model_combobox.pack(side='left', padx=10)

        self.detect_btn = ttk.Button(action_frame, text="Run Detection", command=self.run_detection)
        self.detect_btn.pack(side='right')

        # RESULT DISPLAY
        self.result_label = ttk.Label(self.root, text="Result: Waiting for input...", font=('Helvetica', 13, 'bold'), foreground="blue")
        self.result_label.pack(pady=25)

    def extract_text_from_url(self, url):
        """
        Scrapes the website and returns a tuple: (title, body_content)
        """
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status() 

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract Title and Paragraphs
            title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "No Title Found"
            paragraphs = soup.find_all('p')
            body_content = "\n\n".join([p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 20])

            if not body_content.strip():
                raise ValueError("Loaded page, but no significant text content was found.")

            return title, body_content

        except Exception as e:
            raise Exception(f"Failed to scrape URL: {e}")

    def make_prediction(self, text, model_name):
        """
        Loads the real model and vectorizer to perform a prediction.
        """
        try:
            # Load the Vectorizer
            # This turns the scraped text into the numerical format the model understands
            vectorizer = joblib.load('vectorizer.pkl')
            
            # Map the UI name to your saved filename
            model_files = {
                "Multinomial Naive Bayes": "mnb_model.pkl",
                "SVM": "svm_model.pkl",
                "Logistic Regression": "logreg_model.pkl",
                "AdaBoost": "adaboost_model.pkl",
                "Random Forest": "random_forest_model.pkl",
                "LightGBM": "lgbm_model.pkl"
            }
            
            filename = model_files.get(model_name)
            
            # Load the specific model chosen in the GUI
            model = joblib.load(filename)
            
            # Transform the text and predict
            vectorized_text = vectorizer.transform([text])
            prediction = model.predict(vectorized_text)
            
            # Return a string based on the result
            if prediction[0] == 0:
                return "FAKE NEWS"
            else:
                return "REAL NEWS"

        except FileNotFoundError:
            return f" Error: Model file '{filename}' not found."
        except Exception as e:
            return f" Error during prediction: {e}"

    def run_detection(self):
        url_val = self.url_entry.get().strip()
        selected_model = self.model_combobox.get()

        title_to_analyze = ""
        body_to_analyze = ""
        mode = ""

        # If URL is present, it overrides and populates the boxes
        if url_val:
            self.result_label.config(text="Scraping content...", foreground="orange")
            self.root.update()
            try:
                scraped_title, scraped_body = self.extract_text_from_url(url_val)
                
                # Update the Title and Body boxes
                self.title_entry.delete(0, tk.END)
                self.title_entry.insert(0, scraped_title)
                
                self.body_text.delete("1.0", tk.END)
                self.body_text.insert("1.0", scraped_body)
                
                title_to_analyze = scraped_title
                body_to_analyze = scraped_body
                mode = "URL"
            except Exception as e:
                messagebox.showerror("Scraping Error", str(e))
                self.result_label.config(text="Result: Error", foreground="red")
                return
        else:
            # Use Manual Input
            title_to_analyze = self.title_entry.get().strip()
            body_to_analyze = self.body_text.get("1.0", tk.END).strip()
            mode = "Manual Input"

        # Validate content
        if not title_to_analyze and not body_to_analyze:
            messagebox.showwarning("Input Missing", "Please provide a URL or fill in the Title/Body.")
            return

        # Prediction
        self.result_label.config(text=f"Analyzing with {selected_model}...", foreground="orange")
        self.root.update()

        combined_text = f"{title_to_analyze} {body_to_analyze}"
        final_result = self.make_prediction(combined_text, selected_model)

        color = "red" if "FAKE" in final_result else "green"
        self.result_label.config(text=f"[{mode}] {final_result}", foreground=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeNewsDetectorApp(root)
    root.mainloop()