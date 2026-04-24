import tkinter as tk
from tkinter import ttk, messagebox
import requests
from bs4 import BeautifulSoup
import joblib

class FakeNewsDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("FakeNews Detector")
        self.root.geometry("700x800")
        self.root.configure(bg="#f0f2f5")

        # Style Configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Color Palette
        self.primary_color = "#1a73e8"
        self.bg_color = "#f0f2f5"
        self.card_color = "#ffffff"

        self.setup_styles()
        self.create_widgets()

    def setup_styles(self):
        # Header Style
        self.style.configure("Header.TLabel", font=('Segoe UI', 18, 'bold'), background=self.bg_color, foreground="#202124")
        
        # Sub-header Style
        self.style.configure("SubHeader.TLabel", font=('Segoe UI', 12, 'bold'), background=self.card_color, foreground=self.primary_color)
        
        # Action Button Style
        self.style.configure("Action.TButton", font=('Segoe UI', 11, 'bold'), foreground="white", background=self.primary_color, padding=10)
        self.style.map("Action.TButton", background=[('active', '#1557b0')])

        # Card Style (The white container)
        self.style.configure("Card.TFrame", background=self.card_color, relief="flat")

    def create_widgets(self):
        # HEADER
        header_label = ttk.Label(self.root, text="🛡️ FakeNews Detector", style="Header.TLabel")
        header_label.pack(pady=(30, 20))

        main_container = ttk.Frame(self.root, style="Card.TFrame", padding=30)
        main_container.pack(fill="both", expand=True, padx=40, pady=(0, 30))

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
        self.body_text = tk.Text(main_container, height=8, font=('Segoe UI', 10), relief="flat", highlightthickness=1, highlightbackground="#dadce0", padx=10, pady=10)
        self.body_text.pack(fill="both", expand=True, pady=(0, 20))

        button_container = ttk.Frame(main_container, style="Card.TFrame")
        button_container.pack(fill="x")
        
        self.detect_btn = ttk.Button(button_container, text="RUN ANALYSIS", style="Action.TButton", command=self.run_detection)
        self.detect_btn.pack(anchor="center")

        # RESULT DISPLAY
        self.result_label = ttk.Label(self.root, text="Waiting for input...", font=('Segoe UI', 14, 'bold'), background=self.bg_color, foreground="#5f6368")
        self.result_label.pack(anchor="center",pady=(0, 30))

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

    def make_prediction(self, text):
        """
        Loads the real model and vectorizer to perform a prediction.
        """
        try:
            # Load the Vectorizer
            # This turns the scraped text into the numerical format the model understands
            vectorizer = joblib.load('vectorizer.pkl')
            
            # Load your single best model here
            filename = "random_forest_model.pkl"
            
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
        self.result_label.config(text="Analyzing...", foreground="orange")
        self.root.update()

        combined_text = f"{title_to_analyze} {body_to_analyze}"
        final_result = self.make_prediction(combined_text)

        color = "red" if "FAKE" in final_result else "green"
        self.result_label.config(text=f"[{mode}] {final_result}", foreground=color)

if __name__ == "__main__":
    root = tk.Tk()
    app = FakeNewsDetectorApp(root)
    root.mainloop()