import re
import argparse

import guidance
import pandas as pd


ap = argparse.ArgumentParser(description='Apply LLM for labeling text')
ap.add_argument('input_csv',
                help='Must contain a "message" column with the text to analyze')
ap.add_argument('prompt_file',
                help='Path to a text file with the Guidance template to use')
ap.add_argument('output_csv', help='Path to output CSV file')
ap.add_argument('--iterations', type=int, default=9,
                help='Number of rating rounds to do per text; must be odd (default 9)')
ap.add_argument('--resume-from', help='Path to incomplete output file to resume from')
args = ap.parse_args()
assert args.iterations % 2 == 1, 'Number of rating rounds must be odd'


def preprocess_text(text: str) -> str:
    if type(text) != str:  # nan for missing value in Pandas, for example
        return ''
    # Our text is anonymized with placeholders, which may influence an LLM. Here we
    # replace them with somewhat plausible fake values.
    placeholder_replacements = {
        'email_placeholder': ['azhdeef123@notanemail.com',
                              'alecaa0282@notanemail.com',
                              'jupjuriou@notanemail.com',
                              'lulkinto0001@notanemail.com'],
        'number_placeholder': list(range(10, 1000, 37)),
        'name_placeholder': ['Jasper', 'Donnel', 'Lila', 'Fernandes', 'Bennett',
                             'Kuroshima', 'Seraphina', 'Langley', 'Orion', 'MacDougal'],
        'url_placeholder': ['https://google.com', 'https://wikipedia.org',
                            'https://youtube.com', 'https://facebook.com'],
    }
    text = text.replace('[790361b7e5]', 'don')  # Over-aggressive anonymization: "don't"
    text = re.sub(r'\[[0-9a-f]{10}\]', '', text)  # Different method; just drop them
    for k, vals in placeholder_replacements.items():
        replaced = 0
        while k in text:
            text = text.replace(k, str(vals[replaced]), 1)
            replaced = (replaced + 1) % len(vals)
    return re.sub(r'\s+', ' ', text).strip()


with open(args.prompt_file, 'r', encoding='utf8') as infile:
    prompt = infile.read()

guidance.llm = guidance.llms.OpenAI(
    "text-davinci-003",
    caching=False,
    api_key="NA",  # Not checked by local server
    api_base="http://localhost:8000/v1",
)
program = guidance(prompt)

df = pd.read_csv(args.input_csv)
if args.resume_from:
    df_prev = pd.read_csv(args.resume_from)
    assert len(df) == len(df_prev), 'File to resume from does not match input file'
    df = df_prev
    print('Resuming previous run')
print(len(df), 'rows')



cols = ['help_request_rating' + str(i + 1) for i in range(args.iterations)]
for new_idx, row in df.iterrows():
    if cols[0] in df.columns and not pd.isna(df.at[new_idx, cols[0]]):
        print('Skipping original row_idx', row.row_idx, '(already done)')
        continue
    text = preprocess_text(row.message)
    print('Processing', new_idx + 1, '/', len(df), text)
    rating_items = []
    while len(rating_items) < args.iterations:
        score_program = program(forum_text=text)
        try:  # Parses as number if formatting expectations were matched
            rating_items.append([int(score_program['rating']),
                                 score_program['rating_text']])
        except ValueError:
            print('Rating attempt failed, retrying!')
    rating_items = sorted(rating_items, key=lambda item: item[0])
    median_rating_item = rating_items[len(rating_items) // 2]
    text_program = score_program(rating_to_explain=median_rating_item[1])
    print([item[0] for item in rating_items])
    df.loc[new_idx, cols] = [item[0] for item in rating_items]
    df.at[new_idx, 'median_rating_explanation'] = text_program['explanation'].strip()
    df.to_csv(args.output_csv, index=False)  # Save results along the way
