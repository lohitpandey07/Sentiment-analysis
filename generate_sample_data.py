"""
Creates data/sample_reviews.csv: a SYNTHETIC set of short texts (positive, negative and neutral) built from
templates (including negations like "not good"). It lets you run the project
immediately. For a model that works on real-world text, train on a real dataset
instead (see README.md), for example the IMDB 50K movie reviews.

Run:  python generate_sample_data.py
"""
import os
import random

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ITEMS = ["movie", "phone", "restaurant", "book", "laptop", "hotel", "game", "service",
         "headphones", "course", "app", "meal", "show", "camera", "store"]

POSITIVE = [
    "I loved this {i}", "Absolutely fantastic {i}, highly recommend it",
    "The {i} was amazing and exceeded my expectations", "Great {i}, totally worth the money",
    "What a wonderful {i}", "I am very happy with this {i}",
    "Excellent quality, the {i} works perfectly", "Best {i} I have ever tried",
    "Really enjoyed the {i}, five stars", "The {i} is brilliant and easy to use",
    "Superb {i}, I will definitely buy it again", "The {i} was not bad at all, I actually liked it",
    "I am not disappointed, the {i} is great", "This {i} is so good, I am delighted",
    "Lovely {i} and friendly staff", "The {i} is awesome, thank you",
    "I love this {i}", "I really like this {i}", "I enjoy using this {i}", "I love it, the {i} is great",
]
NEGATIVE = [
    "I hated this {i}", "Terrible {i}, a complete waste of money",
    "The {i} was awful and stopped working in a day", "Very disappointed with this {i}",
    "Worst {i} I have ever tried", "The {i} is poor quality and does not work",
    "Horrible experience, the {i} was a disaster", "I regret buying this {i}, never again",
    "The {i} was boring and far too slow", "Awful {i}, I do not recommend it",
    "The {i} is not good at all", "I did not like the {i}",
    "Not happy with the {i}, it is bad", "This {i} is so bad, I am angry",
    "Rude staff and a dirty {i}", "The {i} is useless, so disappointing",
    "I hate this {i}", "I really dislike this {i}", "I hate it, the {i} is bad", "I cannot stand this {i}",
]
NEUTRAL = [
    "The {i} arrived on Tuesday", "The {i} comes with a charger and a manual",
    "I bought the {i} yesterday", "The {i} is open from nine to five",
    "The {i} is located near the train station", "The {i} is made of plastic",
    "I received the {i} this morning", "The {i} has two buttons and a small screen",
    "It is a {i} that comes in three colors", "The {i} was delivered in a brown box",
    "The {i} was okay, nothing special", "An average {i}, neither good nor bad",
    "The {i} is fine, it does the job", "The {i} was released last year",
    "I ordered the {i} on Monday", "The {i} is available in two sizes",
]
EXTRA_POS = ["", "", " Totally worth it.", " Five stars.", " Would use again!", " So happy."]
EXTRA_NEG = ["", "", " Total waste.", " One star.", " Never again!", " So upset."]
EXTRA_NEU = ["", "", "", " That is all.", " More details later."]


def make(n_per_class=400, seed=7):
    rng = random.Random(seed)
    rows = []
    for _ in range(n_per_class):
        rows.append((rng.choice(POSITIVE).format(i=rng.choice(ITEMS)) + rng.choice(EXTRA_POS), "positive"))
        rows.append((rng.choice(NEGATIVE).format(i=rng.choice(ITEMS)) + rng.choice(EXTRA_NEG), "negative"))
        rows.append((rng.choice(NEUTRAL).format(i=rng.choice(ITEMS)) + rng.choice(EXTRA_NEU), "neutral"))
    df = pd.DataFrame(rows, columns=["text", "label"]).drop_duplicates()
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


if __name__ == "__main__":
    out = make()
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    out.to_csv(os.path.join(BASE_DIR, "data", "sample_reviews.csv"), index=False)
    print("Saved data/sample_reviews.csv", out.shape)
