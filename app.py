from flask import Flask, render_template, request, jsonify
from backend.scorer import search, get_progress, get_results, get_sources, reset_progress
import threading
import logging
from dotenv import load_dotenv
import os

# Load .env file
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

@app.route('/')
def home():
    reset_progress()
    return render_template('index.html')

@app.route('/progress', methods=['POST'])
def progress():
    query = request.form.get('query', '').strip()
    logging.debug(f"Starting search for query: {query}")
    # Start search in a separate thread
    threading.Thread(target=search, args=(query,10), daemon=True).start()
    return render_template('progress.html', query=query)

@app.route('/progress_status', methods=['GET'])
def progress_status():
    progress = get_progress()
    logging.debug(f"Progress status requested: {progress}")
    return jsonify(progress)

@app.route('/results', methods=['GET'])
def results_route():
    try:
        results = get_results()
        sources = get_sources()
        logging.debug(f"Results requested: results={results}, sources={sources}")
        return jsonify({"results": results, "sources": sources})
    except Exception as e:
        logging.error(f"Error in results_route: {e}")
        return jsonify({"results": ["Error retrieving results."], "sources": ["No sources available."]}), 500

@app.route('/search', methods=['POST'])
def search_route():
    query = request.form.get('query', '').strip()
    results = get_results()
    logging.debug(f"Rendering results for query: {query}")
    return render_template('results.html', query=query, results=results)

if __name__ == '__main__':
    app.run(debug=True)