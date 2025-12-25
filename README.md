<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Customer Support Ticket Classifier</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            background-color: #f8f9fa;
            color: #212529;
            line-height: 1.6;
            margin: 0;
            padding: 0;
        }

        .container {
            max-width: 1000px;
            margin: 40px auto;
            background: #ffffff;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 0 10px rgba(0,0,0,0.08);
        }

        h1, h2, h3 {
            color: #0d6efd;
        }

        h1 {
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #6c757d;
            margin-bottom: 30px;
        }

        ul {
            margin-left: 20px;
        }

        li {
            margin-bottom: 10px;
        }

        code {
            background: #f1f3f5;
            padding: 4px 6px;
            border-radius: 4px;
            font-size: 0.95em;
        }

        .badge {
            display: inline-block;
            padding: 6px 10px;
            background: #e9ecef;
            border-radius: 6px;
            margin: 4px 4px 4px 0;
            font-size: 0.9em;
        }

        .highlight {
            background: #e7f5ff;
            border-left: 4px solid #0d6efd;
            padding: 15px;
            margin: 20px 0;
        }

        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 20px 0;
        }

        .success {
            background: #e6fcf5;
            border-left: 4px solid #12b886;
            padding: 15px;
            margin: 20px 0;
        }

        footer {
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
            margin-top: 40px;
        }
    </style>
</head>

<body>
<div class="container">

    <h1>Customer Support Ticket Classifier</h1>
    <p class="subtitle">
        End-to-End NLP-Based Classification System with Ensemble Learning & Confidence-Aware Decision Logic
    </p>

    <h2>📌 Project Overview</h2>
    <p>
        This project is an end-to-end machine learning system designed to automatically classify
        customer support tickets into predefined categories such as
        <strong>Billing, Refund, Technical Issue, Account, Delivery/Service,</strong> and
        <strong>General Inquiry</strong>.
    </p>
    <p>
        The system combines classical machine learning models and deep learning models,
        integrates unsupervised clustering for insight discovery,
        and deploys a confidence-aware ensemble strategy to ensure reliable predictions.
    </p>

    <h2>🎯 Business Problem</h2>
    <ul>
        <li>Manual ticket classification is slow, inconsistent, and does not scale.</li>
        <li>Incorrect routing increases resolution time and customer dissatisfaction.</li>
        <li>Support teams need a reliable, automated categorization system.</li>
    </ul>

    <h2>🧠 Solution Architecture</h2>
    <ul>
        <li>Text preprocessing with cleaning, normalization, and TF-IDF vectorization</li>
        <li>Dimensionality reduction using PCA</li>
        <li>Multiple supervised classification models</li>
        <li>Unsupervised clustering for ticket pattern analysis</li>
        <li>Confidence-based ensemble voting for final decision</li>
    </ul>

    <h2>⚙️ Models Implemented</h2>
    <div class="badge">Logistic Regression (TF-IDF + PCA)</div>
    <div class="badge">Support Vector Machine (TF-IDF + PCA)</div>
    <div class="badge">Artificial Neural Network (Dense)</div>
    <div class="badge">1D CNN (Experimental)</div>
    <div class="badge">K-Means Clustering</div>

    <h2>📊 Model Performance Summary</h2>
    <ul>
        <li><strong>SVM & ANN:</strong> ~99% accuracy with high confidence and consistent predictions</li>
        <li><strong>Logistic Regression:</strong> Strong baseline performance (~98–99%)</li>
        <li><strong>CNN:</strong> Lower confidence due to TF-IDF features and limited dataset</li>
    </ul>

    <div class="highlight">
        <strong>Key Design Decision:</strong><br>
        Only models exceeding a confidence threshold participate in the final prediction.
        Experimental models (CNN) are evaluated but excluded from voting when confidence is low.
    </div>

    <h2>🧪 Why CNN Was Included</h2>
    <div class="warning">
        CNN was intentionally included as an experimental deep learning baseline.
        While CNNs excel with word embeddings and large datasets,
        they underperformed with TF-IDF features and limited data.
        <br><br>
        Instead of removing it, a confidence-aware exclusion mechanism was implemented,
        reflecting real-world ML system design.
    </div>

    <h2>🗳️ Final Decision Logic</h2>
    <ul>
        <li>Models with confidence ≥ threshold participate in voting</li>
        <li>Final prediction is chosen via majority vote</li>
        <li>CNN predictions are displayed transparently but excluded when unreliable</li>
    </ul>

    <div class="success">
        <strong>Final Output:</strong><br>
        Reliable category prediction + model-wise confidence + cluster assignment
    </div>

    <h2>🖥️ Web Application</h2>
    <ul>
        <li>Flask-based REST API</li>
        <li>Interactive HTML/CSS/JavaScript frontend</li>
        <li>Visual confidence bars for each model</li>
        <li>Clear explanation of decision logic</li>
    </ul>

    <h2>📁 Tech Stack</h2>
    <ul>
        <li><strong>Languages:</strong> Python, HTML, CSS, JavaScript</li>
        <li><strong>ML:</strong> Scikit-learn, TensorFlow / Keras</li>
        <li><strong>NLP:</strong> TF-IDF, PCA</li>
        <li><strong>Backend:</strong> Flask</li>
        <li><strong>Deployment:</strong> Local / Docker-ready</li>
    </ul>

    <h2>🚀 Key Takeaways</h2>
    <ul>
        <li>Demonstrates real-world ML decision-making</li>
        <li>Balances accuracy with reliability</li>
        <li>Shows ability to justify model selection</li>
        <li>Highlights production-oriented thinking</li>
    </ul>

    <footer>
        Built as part of an AI/ML internship & portfolio project
    </footer>

</div>
</body>
</html>
