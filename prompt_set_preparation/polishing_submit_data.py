import re
import string
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from multiprocessing import Pool, cpu_count
tqdm.pandas()



communication_words = ['outline',
                       'senence',
                       'quote',
                       'webinar',
                       'recital',
                       'biography',
                       'chronicle',
                       'narrative',
                       'newsletter',
                       'sms',
                       'memoir',
                       'evaluation',
                       'study',
                       'prose',
                       'leaflet',
                       'talk',
                       'bulletin',
                       'announcement',
                       'fiction',
                       'message',
                       'teleconference',
                       'summary',
                       'faq',
                       'article',
                       'brochure',
                       'memorandum',
                       'editorial',
                       'project',
                       'thread',
                       'screenplay',
                       'lyrics',
                       'lecture',
                       'panel',
                       'recipe',
                       'address',
                       'slide',
                       'dossier',
                       'logbook',
                       'infographic',
                       'instruction',
                       'opinion',
                       'abstract',
                       'manual',
                       'livestream',
                       'proclamation',
                       'forum',
                       'periodical',
                       'email',
                       'presentation',
                       'introduction',
                       'scene',
                       'dispatch',
                       'entry',
                       'script',
                       'audiobook',
                       'media',
                       'dialogue',
                       'almanac',
                       'podcast',
                       'document',
                       'conference',
                       'course',
                       'portfolio',
                       'passage',
                       'monograph',
                       'workshop',
                       'haiku',
                       'examination',
                       'autobiography',
                       'discussion',
                       'report',
                       'plot',
                       'workbook',
                       'page',
                       'tutorial',
                       'compendium',
                       'catalogue',
                       'video',
                       'dissertation',
                       'version',
                       'transcript',
                       'text',
                       'comment',
                       'profile',
                       'debate',
                       'reading',
                       'book',
                       'critique',
                       'register',
                       'journal',
                       'notice',
                       'vlog',
                       'letter',
                       'memo',
                       'plan',
                       'excerpt',
                       'essay',
                       'novel',
                       'ebook',
                       'database',
                       'assessment',
                       'reflection',
                       'story',
                       'statement',
                       'poem',
                       'pamphlet',
                       'blueprint',
                       'minutes',
                       'chapbook',
                       'handbook',
                       'explanation',
                       'argument',
                       'manifesto',
                       'framework',
                       'treatise',
                       'web',
                       'briefing',
                       'novella',
                       'roadmap',
                       'draft',
                       'commentary',
                       'exploration',
                       'diary',
                       'paragraph',
                       'speech',
                       'roundtable',
                       'inquiry',
                       'sentence',
                       'complaint',
                       'blog',
                       'testimony',
                       'event',
                       'anthology',
                       'piece',
                       'review',
                       'homily',
                       'research',
                       'tale',
                       'news',
                       'circular',
                       'magazine',
                       'content',
                       'symposium',
                       'archive',
                       'description',
                       'monologue',
                       'tweet',
                       'paper',
                       'information',
                       'digital',
                       'fact',
                       'online',
                       'feature',
                       'travelogue',
                       'post',
                       'ledger',
                       'narration',
                       'interview',
                       'sermon',
                       'webcast',
                       'textbook',
                       'example',
                       'synopsis',
                       'story,',
                       'seminar',
                       'anecdote',
                       'social',
                       'deck',
                       'guidebook',
                       'gazette',
                       'agenda',
                       'release',
                       'prompt']

# Prepare the regex pattern
pattern = r'\b(' + '|'.join(re.escape(word) for word in communication_words) + r')\b'
regex = re.compile(pattern, re.IGNORECASE)


def replace_text_word(example, first_n):
    try:
        splits = example.split(' ', maxsplit=first_n)
        prefix, suffix = ' '.join(splits[:first_n]), splits[first_n]

        if 'text' not in prefix:

            prefix = regex.sub('text', prefix)
            new_words = []

            for w in prefix.split(' '):
                if w not in new_words:
                    new_words.append(w)

            prefix = ' '.join(new_words)

    except IndexError as err:
        return example

    return prefix + ' ' + suffix


def prompt_polishing(prompt):
    prompt = replace_text_word(replace_text_word(replace_text_word(prompt, 3), 5), 6)

    referencing_words = ["following", "given", "next", "original", "subsequent", "preceding", "consecutive", "provided"]
    referencing_words = ['the ' + w for w in referencing_words] + ['a ' + w for w in referencing_words]

    for refer_word in referencing_words:
        refer_regex = re.compile(refer_word + "\s*\w*\s*text", re.IGNORECASE)
        prompt = refer_regex.sub('this text', prompt)

    for target_word in ['into', 'in', 'as']:
        about_regex = re.compile(rf"\babout\s+.*?{target_word}\b", re.IGNORECASE)
        prompt = about_regex.sub('in', prompt)
    return prompt






sub_prompts = pd.read_parquet('data_train.parquet')['label']

sub_prompts = sub_prompts[sub_prompts.str.len() < 250]
sub_prompts = sub_prompts.str.strip(string.punctuation + '0123456789 ')
sub_prompts = sub_prompts.drop_duplicates()

sub_prompts = sub_prompts.apply(lambda x: x.split('.')[0])
sub_prompts = sub_prompts.apply(lambda s: s.split(':')[-1])
sub_prompts = sub_prompts.str.strip(string.punctuation + '0123456789 ')


prompts = pd.Series(np.array(sub_prompts.tolist() + sub_prompts.apply(prompt_polishing).tolist()))


# Regular expression to match and remove the endings
# This pattern allows for varying amounts of whitespace
pattern = r"\s*,\s*and\s+\w+\s+it\s*$"

# Using re.sub() to remove the matched pattern in each sentence
prompts = prompts.progress_apply(lambda s: re.sub(pattern, "", s))

# sub_prompts = sub_prompts.progress_apply(prompt_polishing)
prompts = prompts.drop_duplicates()
prompts = np.array(prompts.tolist())




prompt_embedder = SentenceTransformer('sentence-transformers/sentence-t5-base', device='cuda')
prompt_embs = prompt_embedder.encode(prompts, batch_size=32, show_progress_bar=True)



def top_k_idx(ranks, topk = 3):
    top_part = np.argpartition(-ranks, topk, axis=1)[:, :topk]
    top_values = np.take_along_axis(ranks, top_part, axis=1)
    top_idx = np.take_along_axis(top_part, np.argsort(-top_values, axis=1), axis=1)
    return top_idx





top_k = 2048
chunk_size = 512

chunks = [(i*chunk_size, prompt_embs[i*chunk_size:(i+1)*chunk_size]) for i in range(len(prompt_embs)//chunk_size+2) if  i*chunk_size < len(prompt_embs)]


def process_chunk(data):
    offset, chunk = data
    distance = chunk @ prompt_embs.T

    for i in range(len(chunk)):
        distance[i, offset + i] = 0

    top_indices = top_k_idx(distance, top_k)
    top_values = np.take_along_axis(distance, top_indices, axis=1)
    return top_indices, top_values

indices, values = [], []

with Pool(36) as pool:
    for output in tqdm(pool.imap(process_chunk, chunks), total=len(chunks)):
        indices.append(output[0])
        values.append(output[1])


indices = np.vstack(indices)
values = np.vstack(values)

row_ind = np.tile(np.arange(len(indices))[:, None], (1, indices.shape[-1]))
cube_values = values.flatten() **3

distance_mat = csr_matrix((cube_values, (row_ind.flatten(), indices.flatten())), shape=(len(prompt_embs), len(prompt_embs)))





n_components, labels = connected_components(distance_mat >= 0.985, connection='weak', directed=False)
print(len(labels),'->', len(set(labels)))

deduplication_mask = np.zeros(len(prompts), dtype=bool)

for lbl in tqdm(set(labels)):
    selection = labels == lbl
    ic_score = np.asarray(distance_mat[selection][:, selection].mean(axis=1)).flatten()

    mask = selection[selection] * False
    mask[np.argmax(ic_score)] = True

    deduplication_mask[selection] = mask

dedup_prompts = prompts[deduplication_mask]
dedup_distance_mat = distance_mat[deduplication_mask][:, deduplication_mask]

pd.DataFrame(dedup_prompts).to_csv('final_prompts.csv')
