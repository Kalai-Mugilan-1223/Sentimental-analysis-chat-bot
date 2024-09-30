Emotion-Based Chatbot using Flask, NLTK, and Machine Learning

Overview

	This project is an interactive chatbot built using Flask that can classify the user's emotional sentiment based on their text input and respond accordingly. The chatbot leverages NLTK's 		SentimentIntensityAnalyzer, Spacy, and a machine learning pipeline built with RandomForestClassifier for emotion classification. It also incorporates TF-IDF vectorization to preprocess text.

Features

	-Sentiment Analysis using NLTK's VADER lexicon.
	-Text preprocessing and lemmatization using Spacy.
	-Emotion classification with a RandomForest Classifier model.
	-Dynamic chatbot responses based on the detected emotion.
	-A simple user interface using HTML/CSS and Bootstrap.
	-Displays a confusion matrix to visualize the model's performance.


Project Structure

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
	├── responses.json       
	├── app.py               
	├── mydataset.csv        
	├── requirements.txt     
	└── README.md            

How to Run

	1.Clone the repository
	    git clone (https://github.com/Kalai-Mugilan-1223/Sentimental-analysis-chat-bot)
	    cd project_root
	
	2.Install the required dependencies.
	      pip install -r requirements.txt
	3.Download NLTK's VADER lexicon and Spacy's language model:
	      python -m nltk.downloader vader_lexicon
	      python -m spacy download en_core_web_sm
	4.Run the Flask application.
	      python app.py
	      Open your browser and visit http://localhost:5001 to interact with the chatbot.
	5.Dataset
	      The chatbot is trained using a CSV file (mydataset.csv) containing user descriptions and corresponding emotions. The dataset is preprocessed using Spacy to remove stop words and punctuation, 
	      followed by text lemmatization.
	
	Response Logic
	   The chatbot uses a combination of Sentiment Analysis and Emotion Classification to determine the appropriate response. The following emotions are detected based on the text's sentiment score:
	
	  -Positive Emotions: Happy, Grateful, Excited
	  -Neutral Emotions: Hopeful, Relieved
	  -Negative Emotions: Angry, Sad, Lonely, etc.
		
	Dependencies
	  -Flask
	  -NLTK
	  -Spacy
	  -scikit-learn
	  -Pandas
	  -Matplotlib
	
	 
	You can install these dependencies using:
	  -pip install flask nltk spacy scikit-learn pandas matplotlib
	
	
	
	

