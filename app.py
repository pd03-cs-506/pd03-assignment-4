from flask import Flask, render_template, request, jsonify
from helpers import *

# vectorizer, svd_model, svd_matrix = apply_lsa_news()

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    results = None
    show_image = False

    if request.method == "POST":
        query = request.form.get('query')
        documents, similarities, indices = get_top_documents(query)
        results = list(zip(documents, similarities, indices))
        get_graph(similarities, indices)
        show_image = True 

    return render_template('index.html', 
                           results=results,
                           show_image=show_image)

if __name__ == '__main__':
    app.run(debug=True)