import sys
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from textblob import TextBlob
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics.pairwise import cosine_similarity
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QLabel, QListWidget
from PyQt5.QtCore import QThread, pyqtSignal

nltk.download('punkt')
nltk.download('stopwords')

class TrainingThread(QThread):
    output_signal = pyqtSignal(str)
    similar_wines_signal = pyqtSignal(list)

    def run(self):
        # Dataset
        file_path = "winemag-data-130k-v2.csv"
        wine_reviews = pd.read_csv(file_path, nrows=10000)

        # Data Exploration and Cleaning
        self.output_signal.emit("Dataset Information:")
        self.output_signal.emit(str(wine_reviews.info()))
        self.output_signal.emit("\nFirst few rows:")
        self.output_signal.emit(str(wine_reviews.head()))

        wine_reviews.drop_duplicates(inplace=True)

        categorical_columns = ['country', 'province', 'region_1', 'region_2', 'taster_name', 'taster_twitter_handle', 'variety', 'winery']
        for column in categorical_columns:
            wine_reviews[column].fillna('Unknown', inplace=True)

        wine_reviews['price'].fillna(wine_reviews['price'].median(), inplace=True)

        descriptions = wine_reviews['description']

        # Text Preprocessing
        def preprocess_text(text):
            text = text.lower()
            text = re.sub(r'\W+', ' ', text)
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stop_words]
            ps = PorterStemmer()
            stemmed_tokens = [ps.stem(t) for t in tokens]
            return ' '.join(stemmed_tokens)

        preprocessed_descriptions = descriptions.apply(preprocess_text)

        # Feature Extraction
        tfidf = TfidfVectorizer()
        tfidf_features = tfidf.fit_transform(preprocessed_descriptions)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(preprocessed_descriptions)
        sequence_features = tokenizer.texts_to_sequences(preprocessed_descriptions)
        max_length = max([len(seq) for seq in sequence_features])
        padded_sequence_features = pad_sequences(sequence_features, maxlen=max_length)

        # Sentiment Analysis
        sentiments = preprocessed_descriptions.apply(lambda text: TextBlob(text).sentiment)
        wine_reviews['sentiment_polarity'] = sentiments.apply(lambda x: x.polarity)
        wine_reviews['sentiment_subjectivity'] = sentiments.apply(lambda x: x.subjectivity)

        self.output_signal.emit("\nSentiment Analysis:")
        self.output_signal.emit(str(wine_reviews[['description', 'sentiment_polarity', 'sentiment_subjectivity']].head(10)))

        wine_reviews['sentiment_label'] = (wine_reviews['sentiment_polarity'] > 0).astype(int)

        # Model Development and Training
        X_train_tfidf, X_test_tfidf, y_train, y_test = train_test_split(tfidf_features, wine_reviews['sentiment_label'], test_size=0.2, random_state=42)
        X_train_seq, X_test_seq, _, _ = train_test_split(padded_sequence_features, wine_reviews['sentiment_label'], test_size=0.2, random_state=42)

        logistic_model = LogisticRegression()
        logistic_model.fit(X_train_tfidf, y_train)

        svm_model = SVC()
        svm_model.fit(X_train_tfidf, y_train)

        cnn_model = Sequential([
            Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=max_length),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(units=1, activation='sigmoid')
        ])
        cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        cnn_model.fit(X_train_seq, y_train, epochs=5, batch_size=32, validation_data=(X_test_seq, y_test))

        # Model Evaluation
        models = [
            (logistic_model, "Logistic Regression", X_test_tfidf),
            (svm_model, "Support Vector Machine", X_test_tfidf),
            (cnn_model, "Convolutional Neural Network", X_test_seq)
        ]

        for model, name, X_test in models:
            y_pred = model.predict(X_test)
            y_pred_binary = (y_pred > 0.5).astype(int) if name == "Convolutional Neural Network" else y_pred
            self.output_signal.emit(f"\n{name} Model Evaluation:")
            self.output_signal.emit("Accuracy: " + str(accuracy_score(y_test, y_pred_binary)))
            self.output_signal.emit("Precision: " + str(precision_score(y_test, y_pred_binary)))
            self.output_signal.emit("Recall: " + str(recall_score(y_test, y_pred_binary)))
            self.output_signal.emit("F1-score: " + str(f1_score(y_test, y_pred_binary)))
            self.output_signal.emit("Classification Report:")
            self.output_signal.emit(str(classification_report(y_test, y_pred_binary)))
            
            if name != "Convolutional Neural Network":
                cv_scores = cross_val_score(model, tfidf_features, wine_reviews['sentiment_label'], cv=5)
                self.output_signal.emit("Cross-validation Scores: " + str(cv_scores))
                self.output_signal.emit("Mean CV Score: " + str(cv_scores.mean()))

        # Interpretation and Insights
        feature_names = tfidf.get_feature_names_out()
        logistic_coef = logistic_model.coef_[0]
        top_positive_words = [feature_names[i] for i in logistic_coef.argsort()[-10:][::-1]]
        top_negative_words = [feature_names[i] for i in logistic_coef.argsort()[:10]]

        # Calculate the cosine similarity matrix from the TF-IDF features
        cosine_sim = cosine_similarity(tfidf_features, tfidf_features)

        # Function to get the most similar wines
        def get_similar_wines(index, cosine_sim=cosine_sim):
            sim_scores = list(enumerate(cosine_sim[index]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:11]
            wine_indices = [i[0] for i in sim_scores]
            return wine_reviews['title'].iloc[wine_indices]

        # Test the system by passing a wine index, for example index 1 for the second wine
        similar_wines = get_similar_wines(1)
        self.similar_wines_signal.emit(similar_wines.tolist())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wine Sentiment Analysis and Recommendation")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.output_text = QTextEdit(self)
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.similar_wines_label = QLabel("Wines similar to the one at index 1:", self)
        layout.addWidget(self.similar_wines_label)

        self.similar_wines_list = QListWidget(self)
        layout.addWidget(self.similar_wines_list)

        self.training_thread = TrainingThread()
        self.training_thread.output_signal.connect(self.update_output)
        self.training_thread.similar_wines_signal.connect(self.update_similar_wines)
        self.training_thread.start()

    def update_output(self, text):
        self.output_text.append(text)

    def update_similar_wines(self, wines):
        self.similar_wines_list.clear()
        self.similar_wines_list.addItems(wines)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())