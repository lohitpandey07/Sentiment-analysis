"""
Example texts for the app's option buttons. Every click on an option shows the
NEXT example from that option's list, so you get a different one each time.

Each example is (text, expected_label) where 2 = positive, 1 = neutral and 0 = negative.
The expected label is used by the tests to make sure no example is misclassified.
"""
import random

EXAMPLES = {
    "😊 Happy review": [
        ("I absolutely loved this phone! The battery lasts all day and the camera is amazing.", 2),
        ("Excellent quality, the headphones work perfectly. Best purchase I have made this year.", 2),
        ("What a wonderful hotel and the staff were so friendly. I will stay here again.", 2),
        ("The meal was brilliant and the service was superb. Highly recommend this restaurant.", 2),
        ("I am delighted with this laptop. It is fast, easy to use and totally worth the money.", 2),
    ],
    "😞 Angry review": [
        ("Terrible service. The food was cold, the staff were rude and I will never come back.", 0),
        ("Worst laptop I have ever bought. It stopped working in a day and support was useless.", 0),
        ("I hated this movie. It was boring, far too slow and a complete waste of time.", 0),
        ("Very disappointed with the hotel. The room was dirty and the staff were rude.", 0),
        ("Awful app, it crashes all the time. I regret buying it and do not recommend it.", 0),
    ],
    "😐 Neutral": [
        ("The package arrived on Tuesday.", 1),
        ("The phone comes with a charger and a manual.", 1),
        ("I bought the laptop yesterday.", 1),
        ("The restaurant is open from nine to five.", 1),
        ("The hotel is located near the train station.", 1),
    ],
}


def next_example(category, positions):
    """
    Returns (text, number, total) for the next example of `category`.
    `positions` is a dict remembering where each category is (it is updated here).
    The first click starts at a random example, then every click moves on to the
    next one, so the same example never appears twice in a row.
    """
    pool = EXAMPLES[category]
    last = positions.get(category)
    index = random.randrange(len(pool)) if last is None else (last + 1) % len(pool)
    positions[category] = index
    return pool[index][0], index + 1, len(pool)
