"""SafeChat training: cyberbullying type classifier.
Pipeline: clean -> train/test split -> TF-IDF (fit on train only) -> train LR + NB -> evaluate -> save best.
"""
import re, joblib, json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

def clean(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'@\w+|#\w+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

print('Loading data...')
df = pd.read_csv('cyberbullying_data.csv')
df = df.dropna(subset=['tweet_text', 'cyberbullying_type'])
df['clean'] = df['tweet_text'].apply(clean)
df = df[df['clean'].str.len() > 2].reset_index(drop=True)
print('Rows after cleaning:', len(df))
print(df['cyberbullying_type'].value_counts())

X = df['clean'].astype(str).tolist()
y = df['cyberbullying_type'].astype(str).tolist()

# Split BEFORE vectorizing (no leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f'Train: {len(X_train)}, Test: {len(X_test)}')

vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=3)
Xtr = vec.fit_transform(X_train)   # fit on TRAIN only
Xte = vec.transform(X_test)

models = {
    'logistic_regression': LogisticRegression(max_iter=1000, n_jobs=-1),
    'naive_bayes': MultinomialNB(),
}
results = {}
best_name, best_f1, best_model = None, -1, None
for name, m in models.items():
    m.fit(Xtr, y_train)
    p = m.predict(Xte)
    acc = accuracy_score(y_test, p)
    pr, rc, f1, _ = precision_recall_fscore_support(y_test, p, average='weighted', zero_division=0)
    results[name] = dict(accuracy=round(acc, 4), precision=round(pr, 4), recall=round(rc, 4), f1=round(f1, 4))
    print(f'{name}: acc={acc:.4f} prec={pr:.4f} rec={rc:.4f} f1={f1:.4f}')
    if f1 > best_f1:
        best_name, best_f1, best_model = name, f1, m

print('\nBest:', best_name)
print(classification_report(y_test, best_model.predict(Xte), zero_division=0))
print('Confusion matrix labels:', sorted(set(y_test)))

joblib.dump(best_model, 'model.pkl')
joblib.dump(vec, 'tfidf.pkl')
with open('metrics.json', 'w') as f:
    json.dump({'best_model': best_name, 'results': results,
               'labels': sorted(df['cyberbullying_type'].unique().tolist())}, f, indent=2)
print('Saved model.pkl, tfidf.pkl, metrics.json')
