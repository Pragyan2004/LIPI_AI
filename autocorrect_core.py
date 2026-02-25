"""
AutoCorrect System Core Module
Provides word correction functionality using NLP techniques
"""

import nltk
import re
import string
import pickle
import os
import json
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autocorrect.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoCorrectSystem:
    """
    Main class for AutoCorrect functionality
    Handles word correction, vocabulary management, and probability calculations
    """
    
    def __init__(self):
        """Initialize the AutoCorrect System"""
        # Download required NLTK data first
        self.download_nltk_data()
        
        try:
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            self.word_count = {}
            self.probabilities = {}
            self.vocab = set()
            self.total_words = 0
            self.word_context = {}  # Store word contexts for advanced suggestions
            self.bigrams = {}  # Store word pairs for context-aware correction
            logger.info("AutoCorrect System initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing AutoCorrect System: {e}")
            raise
    
    def download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            logger.info("NLTK data downloaded successfully")
        except Exception as e:
            logger.error(f"Error downloading NLTK data: {e}")
            raise
    
    def load_text_file(self, file_name, encoding='utf8'):
        """
        Load and process text file
        
        Args:
            file_name (str): Path to the text file
            encoding (str): File encoding (default: utf8)
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Loading text file: {file_name}")
            
            with open(file_name, 'r', encoding=encoding) as f:
                text_data = f.read().lower()
                
                # Extract words and handle punctuation
                words = re.findall(r'\b[a-z]+\b', text_data)
                
                # Remove stopwords if needed (optional)
                # words = [w for w in words if w not in self.stop_words]
                
            self.vocab = set(words)
            self.total_words = len(words)
            self.word_count = self.count_word_frequency(words)
            self.probabilities = self.calculate_probability()
            self.build_bigram_model(words)
            
            logger.info(f"Total words loaded: {self.total_words}")
            logger.info(f"Unique words: {len(self.vocab)}")
            logger.info(f"Top 10 most common words: {Counter(self.word_count).most_common(10)}")
            
            return True
            
        except FileNotFoundError:
            logger.error(f"Error: {file_name} not found!")
            return False
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return False
    
    def count_word_frequency(self, words):
        """Count frequency of each word"""
        try:
            return dict(Counter(words))
        except Exception as e:
            logger.error(f"Error counting word frequency: {e}")
            return {}
    
    def calculate_probability(self):
        """Calculate probability of each word"""
        try:
            total = sum(self.word_count.values())
            if total == 0:
                logger.warning("No words to calculate probabilities")
                return {}
            return {word: count/total for word, count in self.word_count.items()}
        except Exception as e:
            logger.error(f"Error calculating probabilities: {e}")
            return {}
    
    def build_bigram_model(self, words):
        """Build bigram model for context-aware suggestions"""
        try:
            for i in range(len(words) - 1):
                current_word = words[i]
                next_word = words[i + 1]
                
                if current_word not in self.bigrams:
                    self.bigrams[current_word] = {}
                
                self.bigrams[current_word][next_word] = self.bigrams[current_word].get(next_word, 0) + 1
            
            # Convert counts to probabilities
            for word in self.bigrams:
                total = sum(self.bigrams[word].values())
                for next_word in self.bigrams[word]:
                    self.bigrams[word][next_word] /= total
            
            logger.info("Bigram model built successfully")
        except Exception as e:
            logger.error(f"Error building bigram model: {e}")
    
    def lemmatize_word(self, word):
        """Lemmatize a given word"""
        try:
            return self.lemmatizer.lemmatize(word.lower())
        except Exception as e:
            logger.error(f"Error lemmatizing word {word}: {e}")
            return word.lower()
    
    def delete_letter(self, word):
        """Generate strings by deleting one character"""
        try:
            return [word[:i] + word[i+1:] for i in range(len(word))]
        except Exception as e:
            logger.error(f"Error in delete_letter for {word}: {e}")
            return []
    
    def swap_letters(self, word):
        """Generate strings by swapping adjacent characters"""
        try:
            return [word[:i] + word[i+1] + word[i] + word[i+2:] 
                    for i in range(len(word)-1)]
        except Exception as e:
            logger.error(f"Error in swap_letters for {word}: {e}")
            return []
    
    def replace_letter(self, word):
        """Generate strings by replacing one character"""
        try:
            letters = string.ascii_lowercase
            return [word[:i] + l + word[i+1:] 
                    for i in range(len(word)) for l in letters]
        except Exception as e:
            logger.error(f"Error in replace_letter for {word}: {e}")
            return []
    
    def insert_letter(self, word):
        """Generate strings by inserting one character"""
        try:
            letters = string.ascii_lowercase
            return [word[:i] + l + word[i:] 
                    for i in range(len(word)+1) for l in letters]
        except Exception as e:
            logger.error(f"Error in insert_letter for {word}: {e}")
            return []
    
    def generate_candidates_level1(self, word):
        """Generate all possible level 1 candidates"""
        try:
            candidates = set()
            candidates.update(self.delete_letter(word))
            candidates.update(self.swap_letters(word))
            candidates.update(self.replace_letter(word))
            candidates.update(self.insert_letter(word))
            return candidates
        except Exception as e:
            logger.error(f"Error generating level 1 candidates for {word}: {e}")
            return set()
    
    def generate_candidates_level2(self, word):
        """Generate level 2 candidates"""
        try:
            level1 = self.generate_candidates_level1(word)
            level2 = set()
            for w in level1:
                if len(w) > 0:  # Avoid empty strings
                    level2.update(self.generate_candidates_level1(w))
            return level2
        except Exception as e:
            logger.error(f"Error generating level 2 candidates for {word}: {e}")
            return set()
    
    def get_contextual_suggestions(self, word, previous_word=None, max_suggestions=5):
        """Get suggestions based on context"""
        try:
            if previous_word and previous_word in self.bigrams:
                context_words = self.bigrams[previous_word]
                suggestions = []
                
                # Find words in context that are similar to the input
                candidates = self.generate_candidates_level1(word)
                valid_candidates = candidates.intersection(set(context_words.keys()))
                
                for candidate in valid_candidates:
                    if candidate in context_words:
                        suggestions.append((candidate, context_words[candidate]))
                
                suggestions.sort(key=lambda x: x[1], reverse=True)
                return suggestions[:max_suggestions]
            
            return []
        except Exception as e:
            logger.error(f"Error getting contextual suggestions: {e}")
            return []
    
    def get_best_correction(self, word, max_suggestions=5, use_lemmatization=True, previous_word=None):
        """
        Get best corrections for a word
        
        Args:
            word (str): Word to correct
            max_suggestions (int): Maximum number of suggestions
            use_lemmatization (bool): Whether to use lemmatization
            previous_word (str): Previous word for context
        
        Returns:
            list: List of (suggestion, probability) tuples
        """
        try:
            word = word.lower().strip()
            
            if use_lemmatization:
                word = self.lemmatize_word(word)
            
            # Check if word is in vocabulary
            if word in self.vocab:
                return [(word, self.probabilities.get(word, 0))]
            
            # Try contextual suggestions first if previous word provided
            if previous_word:
                contextual = self.get_contextual_suggestions(word, previous_word, max_suggestions)
                if contextual:
                    return contextual
            
            # Generate level 1 candidates
            candidates = self.generate_candidates_level1(word)
            valid_candidates = candidates.intersection(self.vocab)
            
            # If no level 1 candidates, try level 2
            if not valid_candidates:
                logger.debug(f"No level 1 candidates for '{word}', trying level 2")
                candidates = self.generate_candidates_level2(word)
                valid_candidates = candidates.intersection(self.vocab)
            
            # Sort by probability
            suggestions = [(w, self.probabilities.get(w, 0)) 
                          for w in valid_candidates]
            suggestions.sort(key=lambda x: x[1], reverse=True)
            
            # If no suggestions found, return word with warning
            if not suggestions:
                logger.warning(f"No suggestions found for word: {word}")
                suggestions = [(word, 0.0)]
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error getting best correction for {word}: {e}")
            return [(word, 0.0)]
    
    def add_word_to_vocab(self, word, frequency=1):
        """
        Add a new word to vocabulary
        
        Args:
            word (str): Word to add
            frequency (int): Initial frequency count
        
        Returns:
            bool: True if added, False if already exists
        """
        try:
            word = word.lower().strip()
            
            if not word:
                logger.warning("Attempted to add empty word")
                return False
            
            if word not in self.vocab:
                self.vocab.add(word)
                self.word_count[word] = self.word_count.get(word, 0) + frequency
                self.total_words += frequency
                self.probabilities = self.calculate_probability()
                logger.info(f"Word '{word}' added to vocabulary")
                return True
            
            logger.info(f"Word '{word}' already exists in vocabulary")
            return False
            
        except Exception as e:
            logger.error(f"Error adding word {word}: {e}")
            return False
    
    def remove_word_from_vocab(self, word):
        """
        Remove a word from vocabulary
        
        Args:
            word (str): Word to remove
        
        Returns:
            bool: True if removed, False if not found
        """
        try:
            word = word.lower().strip()
            
            if word in self.vocab:
                self.vocab.remove(word)
                if word in self.word_count:
                    self.total_words -= self.word_count[word]
                    del self.word_count[word]
                self.probabilities = self.calculate_probability()
                logger.info(f"Word '{word}' removed from vocabulary")
                return True
            
            logger.info(f"Word '{word}' not found in vocabulary")
            return False
            
        except Exception as e:
            logger.error(f"Error removing word {word}: {e}")
            return False
    
    def get_word_stats(self, word):
        """
        Get statistics for a word
        
        Args:
            word (str): Word to get stats for
        
        Returns:
            dict: Word statistics
        """
        try:
            word = word.lower().strip()
            
            stats = {
                'word': word,
                'in_vocab': word in self.vocab,
                'frequency': self.word_count.get(word, 0),
                'probability': self.probabilities.get(word, 0),
                'lemmatized': self.lemmatize_word(word),
                'length': len(word)
            }
            
            # Add similar words
            if word in self.vocab:
                similar_words = self.find_similar_words(word, 5)
                stats['similar_words'] = similar_words
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting word stats for {word}: {e}")
            return {'word': word, 'error': str(e)}
    
    def find_similar_words(self, word, max_results=5):
        """Find similar words in vocabulary"""
        try:
            candidates = self.generate_candidates_level1(word)
            valid_candidates = candidates.intersection(self.vocab)
            
            similar = [(w, self.probabilities.get(w, 0)) for w in valid_candidates]
            similar.sort(key=lambda x: x[1], reverse=True)
            
            return [w[0] for w in similar[:max_results]]
            
        except Exception as e:
            logger.error(f"Error finding similar words for {word}: {e}")
            return []
    
    def batch_correction(self, words, previous_words=None):
        """
        Correct multiple words in batch
        
        Args:
            words (list): List of words to correct
            previous_words (list): List of previous words for context
        
        Returns:
            list: List of corrections
        """
        try:
            corrections = []
            
            for i, word in enumerate(words):
                prev_word = previous_words[i] if previous_words and i < len(previous_words) else None
                suggestions = self.get_best_correction(word, previous_word=prev_word)
                corrections.append({
                    'original': word,
                    'suggestions': suggestions,
                    'best': suggestions[0][0] if suggestions else word
                })
            
            return corrections
            
        except Exception as e:
            logger.error(f"Error in batch correction: {e}")
            return []
    
    def save_model(self, filename='autocorrect_model.pkl'):
        """
        Save the model to a pickle file
        
        Args:
            filename (str): Output filename
        """
        try:
            model_data = {
                'word_count': self.word_count,
                'probabilities': self.probabilities,
                'vocab': self.vocab,
                'total_words': self.total_words,
                'bigrams': self.bigrams,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'version': '1.0.0'
                }
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filename='autocorrect_model.pkl'):
        """
        Load the model from a pickle file
        
        Args:
            filename (str): Input filename
        
        Returns:
            bool: True if loaded successfully
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    model_data = pickle.load(f)
                
                self.word_count = model_data['word_count']
                self.probabilities = model_data['probabilities']
                self.vocab = model_data['vocab']
                self.total_words = model_data['total_words']
                self.bigrams = model_data.get('bigrams', {})
                
                logger.info(f"Model loaded from {filename}")
                if 'metadata' in model_data:
                    logger.info(f"Model metadata: {model_data['metadata']}")
                
                return True
            else:
                logger.warning(f"Model file {filename} not found")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def export_vocabulary(self, filename='vocabulary.txt'):
        """
        Export vocabulary to text file
        
        Args:
            filename (str): Output filename
        """
        try:
            with open(filename, 'w', encoding='utf8') as f:
                for word in sorted(self.vocab):
                    f.write(f"{word}\n")
            
            logger.info(f"Vocabulary exported to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting vocabulary: {e}")
            return False
    
    def get_model_info(self):
        """Get model information"""
        return {
            'vocab_size': len(self.vocab),
            'total_words': self.total_words,
            'unique_words': len(self.word_count),
            'bigram_size': len(self.bigrams),
            'top_words': Counter(self.word_count).most_common(10)
        }

def create_sample_data():
    """Create sample data file if not exists"""
    if not os.path.exists('final.txt'):
        sample_text = """The quick brown fox jumps over the lazy dog. 
        Python programming language is amazing for natural language processing. 
        Machine learning and artificial intelligence are transforming the world. 
        The cat sat on the mat. She sells sea shells on the sea shore. 
        How much wood would a woodchuck chuck if a woodchuck could chuck wood?
        Peter Piper picked a peck of pickled peppers.
        The rain in Spain stays mainly in the plain.
        To be or not to be that is the question.
        All that glitters is not gold.
        A journey of a thousand miles begins with a single step."""
        
        with open('final.txt', 'w', encoding='utf8') as f:
            f.write(sample_text)
        
        print("Sample data created in 'final.txt'")
        return True
    return False

# Main execution
if __name__ == "__main__":
    print("=" * 50)
    print("AutoCorrect System - Training Module")
    print("=" * 50)
    
    # Create sample data if needed
    create_sample_data()
    
    # Initialize the system
    print("\nInitializing AutoCorrect System...")
    autocorrect = AutoCorrectSystem()
    
    # Download NLTK data
    print("Downloading NLTK data...")
    autocorrect.download_nltk_data()
    
    # Load training data
    print("\nLoading training data...")
    if autocorrect.load_text_file("final.txt"):
        print("\n" + "=" * 50)
        print("Training Statistics:")
        print("=" * 50)
        print(f"Total words: {autocorrect.total_words:,}")
        print(f"Unique words: {len(autocorrect.vocab):,}")
        print(f"Bigram pairs: {len(autocorrect.bigrams):,}")
        
        # Save the model
        print("\nSaving model...")
        if autocorrect.save_model('autocorrect_model.pkl'):
            print("✓ Model saved successfully")
        
        # Export vocabulary
        print("Exporting vocabulary...")
        if autocorrect.export_vocabulary('vocabulary.txt'):
            print("✓ Vocabulary exported successfully")
        
        # Test the system
        print("\n" + "=" * 50)
        print("Testing AutoCorrect System:")
        print("=" * 50)
        
        test_words = [
            ("worls", None),        # Common misspelling
            ("programing", None),   # Missing letter
            ("pythoon", None),      # Extra letter
            ("computr", None),      # Missing letter
            ("recieve", None),      # Common error
            ("definately", None),   # Common error
            ("accomodate", None),   # Common error
            ("seperate", None)      # Common error
        ]
        
        for word, context in test_words:
            suggestions = autocorrect.get_best_correction(word, max_suggestions=3)
            print(f"\nWord: '{word}'")
            print(f"Suggestions: {', '.join([f'{s[0]} ({s[1]:.4f})' for s in suggestions])}")
        
        # Test contextual correction
        print("\n" + "=" * 50)
        print("Contextual Correction Test:")
        print("=" * 50)
        
        contextual_tests = [
            ("th", "the"),          # With context
            ("quik", "the"),        # With context
            ("brwn", "quick")       # With context
        ]
        
        for word, context in contextual_tests:
            suggestions = autocorrect.get_best_correction(word, previous_word=context)
            print(f"\nContext: '... {context} {word}'")
            print(f"Suggestions: {', '.join([f'{s[0]} ({s[1]:.4f})' for s in suggestions])}")
        
        # Model info
        print("\n" + "=" * 50)
        print("Model Information:")
        print("=" * 50)
        model_info = autocorrect.get_model_info()
        for key, value in model_info.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        print("\n✓ Training complete! Ready to use.")
        
    else:
        print("\n✗ Error: Could not load training data.")
        print("Please ensure 'final.txt' exists in the current directory.")