from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from autocorrect_core import AutoCorrectSystem
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'lipi-ai-neural-key-2025'

# Load the autocorrect model
autocorrect = AutoCorrectSystem()
if os.path.exists('autocorrect_model.pkl'):
    autocorrect.load_model('autocorrect_model.pkl')
else:
    print("Warning: Model file not found. Please train the model first.")

# File to store correction history
HISTORY_FILE = 'correction_history.json'

def load_history():
    """Load correction history from file"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history):
    """Save correction history to file"""
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        # Handle contact form submission
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        return jsonify({'status': 'success', 'message': 'Thank you for contacting us!'})
    return render_template('contact.html')

@app.route('/history')
def history():
    history_data = load_history()
    return render_template('history.html', history=history_data)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/documentation')
def documentation():
    return render_template('documentation.html')

@app.route('/benchmarks')
def benchmarks():
    return render_template('benchmarks.html')

@app.route('/privacy')
def privacy():
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    return render_template('terms.html')

@app.route('/autocorrect', methods=['GET', 'POST'])
def autocorrect_page():
    if request.method == 'POST':
        word = request.form.get('word', '').strip()
        max_suggestions = int(request.form.get('max_suggestions', 5))
        
        if word:
            suggestions = autocorrect.get_best_correction(word, max_suggestions)
            
            # Save to history
            history = load_history()
            history.append({
                'word': word,
                'suggestions': [s[0] for s in suggestions],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'language': detect_language(word)
            })
            save_history(history[-100:])  # Keep last 100 entries
            
            return render_template('autocorrect.html', 
                                 word=word, 
                                 suggestions=suggestions,
                                 max_suggestions=max_suggestions)
    
    return render_template('autocorrect.html')

def detect_language(word):
    """Simple language detection based on Unicode blocks"""
    if any('\u0900' <= char <= '\u097F' for char in word):
        return 'हिन्दी'
    elif any('\u0B80' <= char <= '\u0BFF' for char in word):
        return 'தமிழ்'
    elif any('\u0C00' <= char <= '\u0C7F' for char in word):
        return 'తెలుగు'
    elif any('\u0980' <= char <= '\u09FF' for char in word):
        return 'বাংলা'
    else:
        return 'English'

@app.route('/api/correct', methods=['POST'])
def api_correct():
    data = request.get_json()
    word = data.get('word', '').strip()
    max_suggestions = data.get('max_suggestions', 5)
    
    if not word:
        return jsonify({'error': 'No word provided'}), 400
    
    suggestions = autocorrect.get_best_correction(word, max_suggestions)
    return jsonify({
        'word': word,
        'suggestions': [{'word': s[0], 'probability': s[1]} for s in suggestions],
        'language': detect_language(word)
    })

@app.route('/api/add-word', methods=['POST'])
def api_add_word():
    data = request.get_json()
    word = data.get('word', '').strip()
    
    if not word:
        return jsonify({'error': 'No word provided'}), 400
    
    success = autocorrect.add_word_to_vocab(word)
    if success:
        autocorrect.save_model('autocorrect_model.pkl')
        return jsonify({'message': f'Word "{word}" added successfully', 'success': True})
    else:
        return jsonify({'message': f'Word "{word}" already exists', 'success': False})

@app.route('/api/word-stats', methods=['GET'])
def api_word_stats():
    word = request.args.get('word', '').strip()
    if not word:
        return jsonify({'error': 'No word provided'}), 400
    stats = autocorrect.get_word_stats(word)
    stats['language'] = detect_language(word)
    return jsonify(stats)

@app.route('/api/model-info', methods=['GET'])
def api_model_info():
    return jsonify({
        'vocab_size': len(autocorrect.vocab),
        'total_words': autocorrect.total_words,
        'unique_words': len(autocorrect.word_count),
        'languages_supported': 22,
        'made_in_india': True,
        'version': '2.0.26'
    })

@app.route('/api/clear-history', methods=['POST'])
def clear_history():
    try:
        save_history([])
        return jsonify({'message': 'History cleared successfully', 'success': True})
    except Exception as e:
        return jsonify({'message': f'Error: {str(e)}', 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)