import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
print("helloworld");
df = pd.read_csv("flipkart_product.csv", encoding="latin1")
df.head()

df = df[['Review', 'Rate']]
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')

df = df.dropna()
def rating_to_sentiment(r):
    if r <= 2:
        return "negative"
    elif r == 3:
        return "neutral"
    else:
        return "positive"

df['sentiment'] = df['Rate'].apply(rating_to_sentiment)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

def clean(text):
    text = str(text).lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = ' '.join([w for w in text.split() if w not in stop_words])
    return text

df['clean_text'] = df['Review'].apply(clean)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['clean_text']).toarray()
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# ===== Logistic Regression (ALREADY trained) =====
y_pred_lr = model.predict(X_test)

accuracy_lr = accuracy_score(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr, average='weighted')




# ===== Random Forest (using sparse TF-IDF directly) =====
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')




from sklearn.metrics import classification_report, confusion_matrix

cm = confusion_matrix(y_test, y_pred_lr)

print("Confusion Matrix:\n", cm)

report = classification_report(y_test, y_pred_lr, output_dict=True)

report_df = pd.DataFrame(report).transpose()
report_df = report_df.loc[['negative', 'neutral', 'positive']]


total = np.sum(cm)

class_accuracy = {}

for i, label in enumerate(['negative', 'neutral', 'positive']):
    TP = cm[i, i]
    FN = sum(cm[i, :]) - TP
    FP = sum(cm[:, i]) - TP
    TN = total - (TP + FN + FP)
    acc = (TP + TN) / total
    class_accuracy[label] = acc

report_df['accuracy'] = pd.Series(class_accuracy)

print("\nFinal Table:\n")
print(report_df[['accuracy','precision','recall','f1-score','support']])










from sklearn.metrics import classification_report, confusion_matrix


cm_rf = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:\n", cm_rf)

report_rf = classification_report(y_test, y_pred_rf, output_dict=True)


report_rf_df = pd.DataFrame(report_rf).transpose()


report_rf_df = report_rf_df.loc[['negative', 'neutral', 'positive']]


total_rf = np.sum(cm_rf)

class_accuracy_rf = {}

labels = ['negative', 'neutral', 'positive']

for i, label in enumerate(labels):
    TP = cm_rf[i, i]
    FN = sum(cm_rf[i, :]) - TP
    FP = sum(cm_rf[:, i]) - TP
    TN = total_rf - (TP + FN + FP)
    acc = (TP + TN) / total_rf
    class_accuracy_rf[label] = acc

report_rf_df['accuracy'] = pd.Series(class_accuracy_rf)


final_rf_table = report_rf_df[['accuracy','precision','recall','f1-score','support']]

print("\nRandom Forest Final Table:\n")
print(final_rf_table)




