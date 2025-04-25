import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


img_size = (180, 180)
batch_size = 32

train = tf.keras.utils.image_dataset_from_directory(r"archive (2) (1)",image_size=img_size, batch_size=batch_size)
test = tf.keras.utils.image_dataset_from_directory(r"archive (2) (1)",image_size=img_size,batch_size=batch_size)
class_names = train.class_names
model = tf.keras.models.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 2, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 2, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(class_names))
])

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(train, epochs=3)

model.summary()

# Evaluation
y_true = []
y_pred = []

for image, label in test:
    pre = model.predict(image)
    y_true.extend(label.numpy())
    y_pred.extend(np.argmax(pre, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=class_names, yticklabels=class_names)
plt.show()

# Save using new format
model.save('image_model.keras')

# Prediction loop
while True:
    img = input("Enter image path: ").strip()
    img = tf.keras.preprocessing.image.load_img(img, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    pred = model.predict(img_array)
    print("Predicted Class:", class_names[np.argmax(tf.nn.softmax(pred[0]))])



#     Dataset Inference
# The dataset is located in the modellab/train directory.

# It follows a folder-based structure where each subfolder corresponds to a class label (rock, paper, scissors).

# The images are resized to 180x180 pixels and loaded in batches of 32.

# Both training and test datasets are loaded from the same directory, which is not ideal for evaluation.

# Model Architecture
# The model is a sequential CNN (Convolutional Neural Network) built using TensorFlow Keras.

# The first layer rescales the pixel values to the range [0, 1].

# It uses two convolutional layers (with 32 and 64 filters) followed by max pooling layers.

# A flatten layer converts the feature maps into a 1D vector.

# The final dense layer outputs logits corresponding to the number of classes (3 in this case).

# Compilation and Training
# The model is compiled using the Adam optimizer.

# The loss function used is SparseCategoricalCrossentropy, suitable for integer-labeled classes.

# Accuracy is used as the evaluation metric during training.

# The model is trained for 3 epochs, which may not be sufficient for convergence.

# Evaluation
# Predictions are made on the same dataset used for training, which may result in overestimated performance.

# The confusion matrix shows the number of correct and incorrect predictions per class.

# The classification report provides precision, recall, and F1-score for each class.

# The model’s summary is printed to show the number of parameters and layer-wise details.





import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import nltk
import string

# Download NLTK stopwords if needed
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Load and prepare dataset
df = pd.read_csv(r"D:\data scince model lab\archive (3) (1)\spam.csv", encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = text.split()
    filtered = [word for word in words if word not in stop_words]
    return ' '.join(filtered)

df['clean_message'] = df['message'].apply(clean_text)

# Word analysis
for label in ['spam', 'ham']:
    print(f"\n🔍 Analyzing messages labeled: {label.upper()}")
    messages = df[df['label'] == label]['clean_message'].dropna()
    all_words = ' '.join(messages).split()
    word_freq = Counter(all_words)
    top_words = word_freq.most_common(10)

    for word, count in top_words:
        print(f"{word}: {count}")

    # Bar chart
    labels_, values = zip(*top_words)
    plt.figure(figsize=(10, 6))
    plt.barh(labels_, values, color='salmon' if label == 'spam' else 'skyblue')
    plt.xlabel("Frequency")
    plt.title(f"Top Words in {label.upper()} Messages")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Wordcloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {label.upper()} Messages")
    plt.tight_layout()
    plt.show()

# Vectorization
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['clean_message'].fillna(''))
y = df['label'].map({'ham': 0, 'spam': 1})

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))
print("\n✅ Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n🎯 Accuracy Score:", accuracy_score(y_test, y_pred))


# ----------------------------
# 🔮 Predict a sample message
# ----------------------------
def predict_message(sample_text):
    cleaned = clean_text(sample_text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    label = 'SPAM' if prediction == 1 else 'HAM'
    print(f"\n📨 Sample Message: {sample_text}")
    print(f"📢 Prediction: {label}")

# Example usage
predict_message("Congratulations! You have won a $1000 Walmart gift card. Click here to claim.")
predict_message("Hey, can we meet tomorrow at the usual place?")




The model achieves an accuracy of 97.58%, meaning it correctly classifies most messages as either spam or ham.

It detects spam messages with 91% precision and recall, indicating reliable spam detection, though slightly less accurate than for ham.

The model is highly effective in identifying ham messages, with 99% precision and recall, showcasing strong performance for non-spam content.

The macro average F1-score of 0.95 suggests the model is performing well across both spam and ham classes, despite the class imbalance.

The most frequent words in spam and ham messages reflect the typical content of each, helping to explain the model's learning process.

The model accurately predicts real-world examples, correctly classifying both a spam message ("gift card") and a ham message ("meet tomorrow").




















import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load the dataset
df = pd.read_csv(r"D:\\data scince model lab\\DTST_FOR_IR.csv")

# Columns to analyze
text_columns = ['Cause_of_accident', 'Type_of_collision', 'Pedestrian_movement']

# Clean and normalize phrases
def clean_phrase(text):
    text = str(text).strip()
    if not text or text.isspace():
        return None
    # Keep letters, numbers, and spaces
    return ''.join(c for c in text if c.isalnum() or c.isspace()).strip()

# Analyze each column
for col in text_columns:
    print(f"\n🔍 Analyzing column: {col}")

    phrases = df[col].dropna().apply(clean_phrase)
    phrases = phrases[phrases.notnull()]

    # Count phrases
    phrase_counts = phrases.value_counts()
    top_phrases = phrase_counts.head(10)

    # Print top phrases and their counts
    print(top_phrases)

    # Plot bar chart (horizontal) using matplotlib
    labels, values = top_phrases.index.tolist(), top_phrases.values.tolist()
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values, color='mediumseagreen')
    plt.xlabel("Frequency")
    plt.ylabel("Phrases")
    plt.title(f"Top Phrases in '{col}'")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    # Word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(phrase_counts.to_dict())
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud of '{col}'")
    plt.tight_layout()
    plt.show()
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

img_size = (180, 180)
batch_size = 32
train_ds = tf.keras.utils.image_dataset_from_directory("modellab/train", image_size=img_size, batch_size=batch_size)
test_ds = tf.keras.utils.image_dataset_from_directory("modellab/train", image_size=img_size, batch_size=batch_size)
class_names = train_ds.class_names
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(len(class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


model.fit(train_ds, epochs=3)

y_true = []
y_pred = []

for images, labels in test_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

model.summary()

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(18, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
model.save('models/simple_model.h5')

# Predict (on one image)
def predict_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # create batch axis

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    print(f"Predicted: {predicted_class} ({confidence:.2f}%)")