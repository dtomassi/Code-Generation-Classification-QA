import pickle
import json
import random

def main():
    # Change conala_all_fp to where it is on your machine.
    conala_all_fp = 'conala-all.json'
    with open(conala_all_fp) as f:
        data = json.load(f)

    intents = set()
    for answer in data:
        intents.add(answer['intent'])

    random.seed(42)
    NUM_NEG_SAMPLES_PER_Q = 2
    NUM_SAMPLES = 10
    negative_samples = []
    for intent in intents:
        generated_samples = []
        # Sample NUM_SAMPLES such that NUM_SAMPLES > NUM_NEG_SAMPLES_PER_Q.
        # If the intents match then we can just get another answer.
        sampled_answers = random.sample(data, NUM_SAMPLES)
        for answer in sampled_answers:
            if len(generated_samples) == NUM_NEG_SAMPLES_PER_Q:
                break
            if answer['intent'] == intent:
                continue
            generated_samples.append({
                'intent': intent,
                'snippet': answer['snippet']
                })
        negative_samples += generated_samples

    with open('negative-samples.json', 'w+') as out_file:
        json.dump(negative_samples, out_file, indent=4)


main()