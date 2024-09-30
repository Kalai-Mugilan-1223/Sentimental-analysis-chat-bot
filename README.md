Emotion-Based Chatbot using Flask, NLTK, and Machine Learning
Overview
This project is an interactive chatbot built using Flask that can classify the user's emotional sentiment based on their text input and respond accordingly. The chatbot leverages NLTK's SentimentIntensityAnalyzer, Spacy, and a machine learning pipeline built with RandomForestClassifier for emotion classification. It also incorporates TF-IDF vectorization to preprocess text.

Features
Sentiment Analysis using NLTK's VADER lexicon.
Text preprocessing and lemmatization using Spacy.
Emotion classification with a RandomForest Classifier model.
Dynamic chatbot responses based on the detected emotion.
A simple user interface using HTML/CSS and Bootstrap.
Displays a confusion matrix to visualize the model's performance.
Project Structure
plaintext
Copy code
project_root/
│
├── static/
│   ├── css/
│   ├── images/
│   └── style.css
│
├── templates/
│   └── index.html
│
├── responses.json       # Predefined responses for different emotions
├── app.py               # Main Flask application
├── mydataset.csv        # Dataset containing descriptions and emotions
├── requirements.txt     # Python dependencies
└── README.md            # This README file
How to Run
Clone the repository.
bash
Copy code
git clone <repository_url>
cd project_root
Install the required dependencies.
bash
Copy code
pip install -r requirements.txt
Download NLTK's VADER lexicon and Spacy's language model:
bash
Copy code
python -m nltk.downloader vader_lexicon
python -m spacy download en_core_web_sm
Run the Flask application.
bash
Copy code
python app.py
Open your browser and visit http://localhost:5001 to interact with the chatbot.
Dataset
The chatbot is trained using a CSV file (mydataset.csv) containing user descriptions and corresponding emotions. The dataset is preprocessed using Spacy to remove stop words and punctuation, followed by text lemmatization.

Response Logic
The chatbot uses a combination of Sentiment Analysis and Emotion Classification to determine the appropriate response. The following emotions are detected based on the text's sentiment score:

Positive Emotions: Happy, Grateful, Excited
Neutral Emotions: Hopeful, Relieved
Negative Emotions: Angry, Sad, Lonely, etc.
Dependencies
Flask
NLTK
Spacy
scikit-learn
Pandas
Matplotlib
You can install these dependencies using:

bash
Copy code
pip install flask nltk spacy scikit-learn pandas matplotlib
Future Improvements
Integrate more complex models like BERT for better emotion classification.
Enhance the user interface for a more engaging user experience.
Add more responses for each emotion category.