import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.exceptions import NotFittedError

# Definieer categorieën
categories = ["sport", "study", "funny"]

# Paden voor model en trainingsdata
MODEL_PATH = "text_classifier.pkl"
TRAINING_DATA_PATH = "training_data.pkl"

# Initialiseer of laad het model als het bestaat
def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        try:
            # Controleer of het model getraind is
            model.predict(["test"])
        except NotFittedError:
            initialize_model(model)
    else:
        # Initialiseer en train model met initiële data
        model = make_pipeline(TfidfVectorizer(), MultinomialNB())
        initialize_model(model)
    return model

def initialize_model(model):
    """Initialiseer en train het model met enkele voorbeelddata als het geen trainingsdata heeft."""
    initial_texts = ["This is a sports article.", "This is a study guide.", "This is a funny joke."]
    initial_labels = ["sport", "study", "funny"]
    model.fit(initial_texts, initial_labels)
    save_model(model)
    print("Model geïnitieerd met initiële data.")

# Sla model op schijf op
def save_model(model):
    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

# Laad of initialiseer opslag voor trainingsdata voor incrementeel leren
def load_training_data():
    if os.path.exists(TRAINING_DATA_PATH):
        with open(TRAINING_DATA_PATH, 'rb') as file:
            return pickle.load(file)
    else:
        return {"texts": [], "labels": []}

# Sla opslag voor trainingsdata op voor incrementeel leren
def save_training_data(data):
    with open(TRAINING_DATA_PATH, 'wb') as file:
        pickle.dump(data, file)

# Train model met alle verzamelde data
def retrain_model(model, data):
    model.fit(data["texts"], data["labels"])
    save_model(model)

# Voorspel categorie van een gegeven tekst
def predict_category(model, text):
    return model.predict([text])[0]

# Voeg nieuwe feedbackdata toe en train model opnieuw
def feedback_learning(model, text, correct_label, data):
    data["texts"].append(text)
    data["labels"].append(correct_label)
    save_training_data(data)
    retrain_model(model, data)  # Train opnieuw met verzamelde data

# Hoofdscript
if __name__ == "__main__":
    # Laad of creëer model en trainingsdata
    model = load_model()
    training_data = load_training_data()

    while True:
        print("\n--- Tekstclassificatiesysteem ---")
        file_path = input("Voer het pad naar het tekstbestand in (of 'exit' om te stoppen): ")

        if file_path.lower() == 'exit':
            print("Programma wordt afgesloten. Tot ziens!")
            break

        # Controleer of bestand bestaat
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                text = file.read()
            print("\n--- Tekstinhoud ---")
            print(text[:500] + "\n" + ("..." if len(text) > 500 else ""))

            # Voorspel categorie
            predicted_category = predict_category(model, text)
            print(f"\nVoorspelde categorie: {predicted_category}")

            # Vraag feedback van gebruiker
            feedback = input(f"Is dit correct? (ja/nee) Voorspeld '{predicted_category}': ").strip().lower()

            if feedback == 'nee':
                # Vraag om de juiste categorie
                correct_category = input(f"Voer de juiste categorie in ({', '.join(categories)}): ").strip().lower()
                if correct_category in categories:
                    # Pas feedbackleren toe
                    feedback_learning(model, text, correct_category, training_data)
                    print("Bedankt! Het model is bijgewerkt met de nieuwe informatie.")
                else:
                    print(f"Ongeldige categorie. Moet een van de volgende zijn: {categories}")
            else:
                print("Geweldig! De voorspelling was correct.")
        else:
            print("Bestand niet gevonden. Controleer het pad en probeer het opnieuw.")