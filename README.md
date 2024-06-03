<!DOCTYPE html>
<html lang="en">
<body>

<header>
    <h1>Sentiment Analysis on Restaurant Reviews</h1>
</header>
<div class="container">
    <h2>Introduction</h2>
    <p>This project involves building a machine learning model to classify restaurant reviews as positive or negative based on their text content. The dataset used is the 'Restaurant_Reviews.tsv' file containing reviews and corresponding sentiments.</p>
    <h2>Data Preprocessing</h2>
    <p>Data preprocessing involves cleaning the text data to remove noise and make it suitable for modeling. The steps include:</p>
    <ul>
        <li>Removing special characters from the reviews</li>
        <li>Converting the text to lowercase</li>
        <li>Tokenizing the text into words</li>
        <li>Removing stop words</li>
        <li>Stemming the words to their root forms</li>
    </ul>
    <p>The cleaned text data is then used to create a corpus for further processing.</p>
    <h2>Feature Extraction</h2>
    <p>We use the Bag of Words model with <code>CountVectorizer</code> to convert the text data into numerical data. The maximum number of features is set to 1500.</p>
    <h2>Model Building</h2>
    <p>The dataset is split into training and testing sets with an 80-20 split. A Naive Bayes classifier is used to train the model on the training set.</p>
    <h2>Model Evaluation</h2>
    <p>The model's performance is evaluated using accuracy, precision, and recall metrics. A confusion matrix is also plotted to visualize the results.</p>
    <h2>Hyperparameter Tuning</h2>
    <p>We perform hyperparameter tuning on the Naive Bayes classifier to find the best alpha value that gives the highest accuracy.</p>
    <h2>Prediction</h2>
    <p>A function is defined to predict the sentiment of a given review. The review is cleaned and transformed using the same preprocessing steps, and the trained model is used to make predictions.</p>
    <h2>Results</h2>
    <p>The model performs well on the test set with the following scores:</p>
    <ul>
        <li>Accuracy: 78.5%</li>
        <li>Precision: 0.76</li>
        <li>Recall: 0.79</li>
    </ul>
    <p>The best accuracy is achieved with an alpha value of 0.2.</p>
    <h2>Conclusion</h2>
    <p>This project successfully demonstrates the process of building a sentiment analysis model using machine learning techniques. The model can accurately classify restaurant reviews as positive or negative, and the performance is further improved through hyperparameter tuning.</p>
    <h2>How to Run the Project</h2>
    <p>To run this project, follow these steps:</p>
    <ol>
        <li>Clone the repository to your local machine.</li>
        <li>Ensure you have the necessary libraries installed: <code>numpy</code>, <code>pandas</code>, <code>nltk</code>, <code>sklearn</code>, <code>matplotlib</code>, <code>seaborn</code>.</li>
        <li>Download the 'Restaurant_Reviews.tsv' dataset and place it in the project directory.</li>
        <li>Run the Python script to preprocess the data, build and evaluate the model, and make predictions.</li>
    </ol>
</div>

</body>
</html>
