"""
Train the sentiment model and save it to model/sentiment_model.sav

Default (synthetic demo data):
    python train_model.py

Your own data (a CSV with a text column and a label column):
    python train_model.py --data data/IMDB.csv --text-col review --label-col sentiment

Labels can be positive / negative / neutral (or 1 / 0 for positive / negative).
"""
import argparse
import os

import sentiment_core as sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join(sc.BASE_DIR, "data", "sample_reviews.csv"))
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"{args.data} not found. Run `python generate_sample_data.py` for demo data, "
            "or pass your own file with --data."
        )

    df = sc.load_dataset(args.data, args.text_col, args.label_col)
    print(f"Loaded {len(df)} rows: {df['label'].map(dict(enumerate(sc.CLASS_NAMES))).value_counts().to_dict()}")

    model, metrics = sc.train(df)
    name = os.path.basename(args.data)
    metrics["dataset"] = name
    metrics["synthetic"] = name == "sample_reviews.csv"
    sc.save(model, metrics)

    print(f"Test accuracy: {metrics['accuracy']:.3f}   F1: {metrics['f1']:.3f}")
    print(f"Saved {sc.MODEL_PATH}")
    if metrics["synthetic"]:
        print("Note: this is template-based demo data, so the accuracy is not meaningful.")


if __name__ == "__main__":
    main()
