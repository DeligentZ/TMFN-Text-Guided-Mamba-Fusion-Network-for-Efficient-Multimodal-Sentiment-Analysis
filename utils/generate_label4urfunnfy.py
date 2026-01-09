
import pickle
import csv
import os
from collections import Counter


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        try:
            data = pickle.load(f)
        except:
            f.seek(0)
            data = pickle.load(f, encoding='latin1')
    return data


def process_text(context_sentences, punchline_sentence):
    context_text = ''
    if context_sentences:
        context_texts = []
        for sent in context_sentences:
            if isinstance(sent, list):
                context_texts.append(' '.join([str(w) for w in sent]))
            else:
                context_texts.append(str(sent))
        context_text = ' '.join(context_texts)

    punchline_text = ''
    if punchline_sentence:
        if isinstance(punchline_sentence, list):
            punchline_text = ' '.join([str(w) for w in punchline_sentence])
        else:
            punchline_text = str(punchline_sentence)

    full_text = (context_text + ' ' + punchline_text).strip()

    full_text = full_text.replace('\n', ' ').replace('\r', ' ')
    full_text = ' '.join(full_text.split())

    return full_text


def generate_label_csv(pickle_dir, output_csv_path, video_dir=None):

    print("=" * 50)
    print("UR-Funny Label CSV ")
    print("=" * 50)

    print("\n[1/5] loading pickle file...")

    data_folds_path = os.path.join(pickle_dir, 'data_folds.pkl')
    language_sdk_path = os.path.join(pickle_dir, 'language_sdk.pkl')
    humor_label_path = os.path.join(pickle_dir, 'humor_label_sdk.pkl')

    for fpath in [data_folds_path, language_sdk_path, humor_label_path]:
        if not os.path.exists(fpath):
            print(f"waring: the file is not exist - {fpath}")
            return

    data_folds = load_pickle(data_folds_path)
    language_sdk = load_pickle(language_sdk_path)
    humor_label_sdk = load_pickle(humor_label_path)

    print(f"  - data_folds keys: {list(data_folds.keys())}")
    print(f"  - language_sdk: {len(language_sdk)} ")
    print(f"  - humor_label_sdk: {len(humor_label_sdk)} ")

    print("\n[2/5] Parse data partitioning ...")

    train_ids = data_folds.get('train', [])
    dev_ids = data_folds.get('dev', [])
    test_ids = data_folds.get('test', [])

    print(f"  - Train: {len(train_ids)} ")
    print(f"  - Dev:   {len(dev_ids)} ")
    print(f"  - Test:  {len(test_ids)} ")

    id_to_mode = {}
    for sample_id in train_ids:
        id_to_mode[sample_id] = 'train'
    for sample_id in dev_ids:
        id_to_mode[sample_id] = 'valid'
    for sample_id in test_ids:
        id_to_mode[sample_id] = 'test'

    print("\n[3/5] Building data records...")

    records = []
    missing_label = 0
    missing_text = 0

    all_ids = train_ids + dev_ids + test_ids

    for sample_id in all_ids:
        if sample_id not in humor_label_sdk:
            missing_label += 1
            continue
        label = humor_label_sdk[sample_id]

        if sample_id not in language_sdk:
            missing_text += 1
            continue

        lang_data = language_sdk[sample_id]

        context_sentences = lang_data.get('context_sentences', [])
        punchline_sentence = lang_data.get('punchline_sentence', [])

        text = process_text(context_sentences, punchline_sentence)

        if not text:
            missing_text += 1
            continue

        records.append({
            'sample_id': sample_id,
            'text': text,
            'label': int(label),
            'mode': id_to_mode[sample_id]
        })

    print(f"  - Successful build: {len(records)}")
    if missing_label > 0:
        print(f"  - Missing label: {missing_label}")
    if missing_text > 0:
        print(f"  - Missing text: {missing_text}")

    if video_dir and os.path.exists(video_dir):
        print("\n[4/5] Validating video files...")
        video_files = set(os.listdir(video_dir))

        valid_records = []
        missing_video = 0

        for record in records:
            video_name = f"{record['sample_id']}.mp4"
            if video_name in video_files:
                valid_records.append(record)
            else:
                missing_video += 1

        print(f"  - There's a video for that: {len(valid_records)}")
        print(f"  - Missing video: {missing_video}")

        records = valid_records
    else:
        print("\n[4/5] Skip video verification (unavailable video_dir)")

    print("\n[5/5] Save the CSV file...")

    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'text', 'label', 'mode'])
        writer.writeheader()
        writer.writerows(records)

    print(f"  - save to: {output_csv_path}")

    print("\n" + "=" * 50)
    print("statistical information")
    print("=" * 50)
    print(f"Total number of samples: {len(records)}")

    mode_counts = Counter(r['mode'] for r in records)
    print(f"\nSplit by mode:")
    print(f"  - train: {mode_counts.get('train', 0)}")
    print(f"  - valid: {mode_counts.get('valid', 0)}")
    print(f"  - test:  {mode_counts.get('test', 0)}")

    label_counts = Counter(r['label'] for r in records)
    print(f"\nSplit by label:")
    print(f"  - funny (1):   {label_counts.get(1, 0)}")
    print(f"  - unfunny (0): {label_counts.get(0, 0)}")

    text_lengths = [len(r['text']) for r in records]
    print(f"\nText Length Statistics (characters):")
    print(f"  - shortest: {min(text_lengths)}")
    print(f"  - longest: {max(text_lengths)}")
    print(f"  - average: {sum(text_lengths) / len(text_lengths):.1f}")

    sorted_lengths = sorted(text_lengths)
    mid = len(sorted_lengths) // 2
    median = sorted_lengths[mid] if len(sorted_lengths) % 2 == 1 else (sorted_lengths[mid-1] + sorted_lengths[mid]) / 2
    print(f"  - median: {median:.1f}")
    print("=" * 50)

    print("\nSample data (first 3):")
    for i, record in enumerate(records[:3]):
        print(f"\n[{i+1}] ID: {record['sample_id']}")
        print(f"    Label: {record['label']} | Mode: {record['mode']}")
        text_preview = record['text'][:100] + "..." if len(record['text']) > 100 else record['text']
        print(f"    Text: {text_preview}")


def inspect_pickle_structure(pickle_dir):

    print("=" * 50)
    print("Pickle File structure checking")
    print("=" * 50)

    language_sdk_path = os.path.join(pickle_dir, 'language_sdk.pkl')
    if os.path.exists(language_sdk_path):
        language_sdk = load_pickle(language_sdk_path)
        print(f"\nlanguage_sdk.pkl:")
        print(f"  - type: {type(language_sdk)}")
        print(f"  - number of sample: {len(language_sdk)}")


        first_key = list(language_sdk.keys())[0]
        first_value = language_sdk[first_key]
        print(f"  - example ID: {first_key}")
        print(f"  - filed: {list(first_value.keys()) if isinstance(first_value, dict) else type(first_value)}")

        if isinstance(first_value, dict):
            for k, v in first_value.items():
                if isinstance(v, list):
                    sample_str = str(v[:2]) if len(v) > 0 else str(v)
                    if len(sample_str) > 80:
                        sample_str = sample_str[:80] + "..."
                    print(f"    - {k}: list, len={len(v)}, example={sample_str}")
                else:
                    print(f"    - {k}: {type(v).__name__}")

    humor_label_path = os.path.join(pickle_dir, 'humor_label_sdk.pkl')
    if os.path.exists(humor_label_path):
        humor_label_sdk = load_pickle(humor_label_path)
        print(f"\nhumor_label_sdk.pkl:")
        print(f"  - type: {type(humor_label_sdk)}")
        print(f"  - number of sample: {len(humor_label_sdk)}")


        label_counts = Counter(humor_label_sdk.values())
        print(f"  - distribute of label: {dict(label_counts)}")

    data_folds_path = os.path.join(pickle_dir, 'data_folds.pkl')
    if os.path.exists(data_folds_path):
        data_folds = load_pickle(data_folds_path)
        print(f"\ndata_folds.pkl:")
        print(f"  - type: {type(data_folds)}")
        for k, v in data_folds.items():
            print(f"  - {k}: {len(v)} ")


if __name__ == "__main__":
    PICKLE_DIR = ""
    OUTPUT_CSV = ""
    VIDEO_DIR = ""
    if not PICKLE_DIR or not OUTPUT_CSV:
        print("Pls set path")
        print("Example:")
        print('  PICKLE_DIR = ""')
        print('  OUTPUT_CSV = ""')
        print('  VIDEO_DIR = "" ')
        print("\n" + "=" * 50)
        print("=" * 50)
    else:
        generate_label_csv(
            pickle_dir=PICKLE_DIR,
            output_csv_path=OUTPUT_CSV,
            video_dir=VIDEO_DIR if VIDEO_DIR else None
        )
