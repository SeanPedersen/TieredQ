"""Primary matrix and retained GloVe-100K scaling evidence."""
import csv
import json

NAMES=dict(sift='SIFT1M',glove1m='GloVe-200 1M',openai='OpenAI-1536 100K',
           gemini='Gemini-768 99K',openai1m='OpenAI-1536 1M',msmarco1m='MS MARCO-1M real')
DIMS=dict(sift=128,glove1m=200,openai=1536,gemini=768,openai1m=1536,msmarco1m=1024)


def memory_folders(w,bits):
    return [w/f'match-b{bits}',w/f'glove1m/match-b{bits}']


def equal_folders(w,bits):
    return [w/f'equal-recall/b{bits}',w/f'glove1m/equal-recall/b{bits}']


def primary_cells(w,bits):
    return [row for folder in memory_folders(w,bits) for row in csv.DictReader((folder/'cells.csv').open())
            if row['dataset'] in NAMES]


def primary_matches(w,bits,codec):
    rows=[row for folder in memory_folders(w,bits)
          for row in json.loads((folder/f'matches-{codec}.json').read_text()) if row['dataset'] in NAMES]
    assert len(rows)==18,(bits,codec,len(rows))
    return rows


def primary_equal(w,bits):
    return [json.loads(path.read_text()) for folder in equal_folders(w,bits) for path in folder.glob('*.json')
            if json.loads(path.read_text())['dataset'] in NAMES]
