import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import  accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re

from sylbreak_by_regular_expression import break_syllables, create_break_pattern


def detect_language(text_value):
    burmese_pattern = re.compile(r'[\u1000-\u109F\uAA60-\uAA7F]+')
    is_burmese = bool(burmese_pattern.search(text_value))
    return 'mm' if is_burmese else 'en'


def clean_data(df_value):
    df_value.fillna('', inplace=True)
    # df_value['name'] = df_value['name'].replace('ဦး', '', regex=True).replace('ဒေါ်', '', regex=True).replace('ေဒၚ',
    #                                                                                                           regex=True)
    # df_value['name'] = df_value['name'].str.replace('^U+', '', regex=True)

    # Convert gender labels to numerical values (e.g., Male: 0, Female: 1)
    df_value['gender'] = df_value['gender'].map({'M': 0, 'F': 1})
    return df_value


def extract_syllables(name):
    break_pattern = create_break_pattern()
    segmented_name = break_syllables(name, break_pattern, ' ')
    return segmented_name.split()


def get_data_by_lang(all_df, lang):
    all_df['language'] = all_df['name'].apply(detect_language)

    # Splitting into Myanmar and English DataFrames
    myanmar_data = all_df[all_df['language'] == 'mm']
    english_data = all_df[all_df['language'] == 'en']

    myanmar_data = myanmar_data.drop(columns=['language'])
    english_data = english_data.drop(columns=['language'])
    if (lang == 'mm'):
        return myanmar_data
    else:
        return english_data

def detect_language(text_value):
    burmese_pattern = re.compile(r'[\u1000-\u109F\uAA60-\uAA7F]+')
    is_burmese = bool(burmese_pattern.search(text_value))
    return 'my' if is_burmese else 'en'


# Calculating Accuracy For Burglish Name(Eng)
def predict_gender(cleaned_data_frame, input_value):
    lang_value = detect_language(input_value)

    df_by_lang = get_data_by_lang(cleaned_data_frame, lang_value)
    lang_data = df_by_lang

    vectorizer = CountVectorizer(lowercase=False)

    if (lang_value == 'mm'):
        lang_data['name'] = lang_data['name'].apply(extract_syllables)
        lang_data['name'] = lang_data['name'].apply(lambda x: ' '.join(x))
        vectorizer = CountVectorizer(analyzer=extract_syllables)

    train_data, test_data = train_test_split(lang_data, test_size=0.2, random_state=42)
    X_train = vectorizer.fit_transform(train_data['name'])
    y_train = train_data['gender']
    X_test = vectorizer.transform(test_data['name'])
    y_test = test_data['gender']

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    transformed_name_input = vectorizer.transform([input_value.lower()])

    # Make a prediction
    predicted_gender_prob = classifier.predict_proba(transformed_name_input)

    # Print the prediction and probability
    print(
        f'\n\n\n{input_value}: {"Male" if predicted_gender_prob[0][0] >= 0.5 else "Female"} \n(Probability: {predicted_gender_prob[0][0]:.2f} for Male, {predicted_gender_prob[0][1]:.2f} for Female)\n')


# Load the dataset
dataset_path = 'name_gender-data.csv'
df = pd.read_csv(dataset_path)

df = clean_data(df)

while True:
    input_name = input("Enter a name (or 'exit' to stop): ")

    if input_name.lower() == 'exit':
        break

    predict_gender(df, input_name)
