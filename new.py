import re
import string
from collections import Counter, defaultdict
import tkinter as tk
from tkinter import filedialog, messagebox
from nltk import pos_tag, word_tokenize
from sklearn.metrics import precision_score, recall_score, f1_score
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

# Function to read the corpus
def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()
        words = re.findall(r'\w+', text)
    return words

# Function to calculate word probabilities
def word_probabilities(word_counts):
    total_words = sum(word_counts.values())
    return {word: count / total_words for word, count in word_counts.items()}

# HMM-based Spell Checker
class HMMSpellChecker:
    def __init__(self, words):
        self.vocabs = set(words)
        self.word_counts = Counter(words)
        self.word_probs = word_probabilities(self.word_counts)

    def level_one_edits(self, word):
        letters = string.ascii_lowercase
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [left + right[1:] for left, right in splits if right]
        swaps = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right) > 1]
        replaces = [left + c + right[1:] for left, right in splits if right for c in letters]
        inserts = [left + c + right for left, right in splits for c in letters]
        return set(deletes + swaps + replaces + inserts)

    def check(self, word, context=""):
        if word in self.vocabs:
            return [word]
        
        suggestions = self.level_one_edits(word)
        valid_suggestions = [w for w in suggestions if w in self.vocabs]
        ranked_suggestions = sorted(valid_suggestions, key=lambda w: self.word_probs.get(w, 0), reverse=True)
        
        return ranked_suggestions[:10] if ranked_suggestions else [word]

# Enhanced N-Gram Model for Autocomplete and Next-Word Prediction
class NGramModel:
    def __init__(self, words, n=3):
        self.n = n
        self.ngrams = defaultdict(list)
        self.ngram_counts = Counter()
        for i in range(len(words) - n + 1):
            gram = tuple(words[i:i + n])
            self.ngrams[gram[:-1]].append(gram[-1])
            self.ngram_counts[gram[:-1]] += 1
        self.ngram_probs = self.calculate_ngram_probabilities()

    def calculate_ngram_probabilities(self):
        total_ngrams = sum(self.ngram_counts.values())
        return {gram: count / total_ngrams for gram, count in self.ngram_counts.items()}

    def autocomplete(self, prefix):
        prefix_words = prefix.split()
        if len(prefix_words) >= self.n - 1:
            context = tuple(prefix_words[-(self.n - 1):])
            suggestions = Counter(self.ngrams[context]).most_common()
            return [word for word, _ in suggestions[:10]]
        else:
            return []

# Spell Checker Application with POS Tagging
class SpellCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Spell Checker and Autocomplete with POS Tagging")
        
        # Set window size and center it
        window_width = 600
        window_height = 400
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        position_top = int(screen_height / 2 - window_height / 2)
        position_left = int(screen_width / 2 - window_width / 2)
        self.root.geometry(f'{window_width}x{window_height}+{position_left}+{position_top}')

        # Create a frame for centering all widgets in the window
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(expand=True)

        # Upload button for file
        self.upload_button = tk.Button(self.main_frame, text="Upload File", command=self.load_file, font=("Arial", 12))
        self.upload_button.pack(pady=10)

        # Input box for text
        self.input_text = tk.Entry(self.main_frame, width=80, font=("Arial", 14))
        self.input_text.pack(pady=10)
        self.input_text.bind("<KeyRelease>", self.update_suggestions)

        # Frame for suggestions
        self.suggestions_frame = tk.Frame(self.main_frame)
        self.suggestions_frame.pack(pady=5)

        # Initialize spell checker and n-gram model to None initially
        self.spell_checker = None
        self.ngram_model = None
        self.ground_truth = []  # Placeholder for ground truth

        # Tracking evaluation metrics for later display
        self.total_precision = 0
        self.total_recall = 0
        self.total_f1 = 0
        self.total_count = 0

        # Bind window close to display accuracy
        self.root.protocol("WM_DELETE_WINDOW", self.display_accuracy)

    def load_file(self):
        # Open file dialog to select a file
        file_path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if file_path:
            try:
                words = read_corpus(file_path)
                self.spell_checker = HMMSpellChecker(words)
                self.ngram_model = NGramModel(words, n=3)
                messagebox.showinfo("Success", "File loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {e}")

    def update_suggestions(self, event):
        if not self.spell_checker or not self.ngram_model:
            return  # Exit if no spell checker or model is loaded

        # Clear previous suggestions
        for widget in self.suggestions_frame.winfo_children():
            widget.destroy()

        input_text = self.input_text.get()
        if not input_text:
            return

        input_words = input_text.strip().split()
        if not input_words:
            return

        last_word = input_words[-1]
        prefix = " ".join(input_words[:-1])

        # Create a container frame for both autocomplete and spellcheck
        suggestions_container = tk.Frame(self.suggestions_frame)
        suggestions_container.pack(fill="both", expand=True)

        # Get autocomplete suggestions
        autocomplete_suggestions = self.ngram_model.autocomplete(input_text)

        # Display autocomplete suggestions
        tk.Label(suggestions_container, text="Autocomplete Suggestions:", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
        autocomplete_frame = tk.Frame(suggestions_container)
        autocomplete_frame.pack(fill="both", expand=True)

        for suggestion in autocomplete_suggestions:
            button = tk.Button(autocomplete_frame, text=suggestion, command=lambda s=suggestion: self.append_word(s), font=("Arial", 12), relief="solid")
            button.pack(side="left", padx=5)

        # Check and display spell-check suggestions after completing a word
        if len(input_words) > 1:
            last_word = input_words[-1]
            spell_suggestions = self.spell_checker.check(last_word, prefix)
            tk.Label(suggestions_container, text="Spell Check Suggestions:", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
            spell_frame = tk.Frame(suggestions_container)
            spell_frame.pack(fill="both", expand=True)

            for suggestion in spell_suggestions:
                button = tk.Button(spell_frame, text=suggestion, command=lambda s=suggestion: self.replace_word(s), font=("Arial", 12), relief="solid")
                button.pack(side="left", padx=5)

        # Evaluate and calculate performance after suggestions are displayed
        if len(input_words) > 1:
            ground_truth = self.get_ground_truth(input_words[-1])  # Assume ground_truth is available
            self.evaluate_performance(spell_suggestions, ground_truth)

        # POS tagging for the entire input sentence
        pos_tags = pos_tag(word_tokenize(input_text))
        pos_tags_str = ", ".join([f"{word}/{tag}" for word, tag in pos_tags])

        # Display POS tagging result
        tk.Label(suggestions_container, text="POS Tags:", font=("Arial", 12, "bold")).pack(anchor="w", pady=5)
        tk.Label(suggestions_container, text=pos_tags_str, font=("Arial", 10)).pack(anchor="w")

    def append_word(self, word):
        current_text = self.input_text.get().strip()
        updated_text = f"{current_text} {word}"
        self.input_text.delete(0, tk.END)
        self.input_text.insert(tk.END, updated_text)

    def replace_word(self, suggestion):
        input_text = self.input_text.get()
        input_words = input_text.strip().split()
        input_words[-1] = suggestion
        updated_text = " ".join(input_words)
        self.input_text.delete(0, tk.END)
        self.input_text.insert(tk.END, updated_text)

    def evaluate_performance(self, predictions, ground_truth):
        true_positives = len(set(predictions).intersection(ground_truth))
        false_positives = len(set(predictions) - set(ground_truth))
        false_negatives = len(set(ground_truth) - set(predictions))

        if true_positives + false_positives > 0:
            precision = true_positives / (true_positives + false_positives)
        else:
            precision = 0.0

        if true_positives + false_negatives > 0:
            recall = true_positives / (true_positives + false_negatives)
        else:
            recall = 0.0

        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0

        self.total_precision += precision
        self.total_recall += recall
        self.total_f1 += f1
        self.total_count += 1

    def display_accuracy(self):
        if self.total_count > 0:
            avg_precision = self.total_precision / self.total_count
            avg_recall = self.total_recall / self.total_count
            avg_f1 = self.total_f1 / self.total_count
            performance_metrics = f"Precision: {avg_precision:.4f}\nRecall: {avg_recall:.4f}\nF1-Score: {avg_f1:.4f}"
            messagebox.showinfo("Final Evaluation", performance_metrics)
        self.root.quit()

    def get_ground_truth(self, word):
        # Placeholder for ground truth; in real applications, replace with actual ground truth data
        return [word]  # Returning the word itself as placeholder ground truth

# Main function to run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = SpellCheckerApp(root)
    root.mainloop()