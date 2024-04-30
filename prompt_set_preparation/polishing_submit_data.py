"""
This script is designed to refine and deduplicate text prompts related to various forms of communication
like emails, articles, and reports. The process involves several key steps:

1. Preprocessing: Loads and cleans data from a dataset. It filters prompts based on specific length and character
   criteria and removes duplicates. Cleaning includes stripping certain punctuation and numbers.

2. Text Transformation: Modifies prompts by replacing specific common words with "text" to standardize terminology
   across the dataset. Additional cleaning is applied to enhance consistency.

3. Embedding Generation: Converts the cleaned and standardized text prompts into numerical vectors (embeddings) using
   a machine learning model. These embeddings represent the semantic meanings of each prompt.

4. Similarity Analysis: Calculates similarity between embeddings to identify closely related prompts. Employs matrix
   operations to process data in chunks for efficiency.

5. Clustering and Deduplication: Uses connectivity-based clustering to group similar prompts and selects the most
   representative prompt from each cluster to reduce redundancy, ensuring a diverse and unique set of prompts.

6. Output: Saves the deduplicated list of prompts to a CSV file for further usage in training machine learning models
   or providing a curated set of prompts for content generation tools.

The script aims to automate the preparation of high-quality, non-redundant text data, which is essential for improving
the performance of natural language processing (NLP) applications.
"""

# Import necessary libraries for regular expression, data manipulation, and linear algebra
import re
import string
import pandas as pd
import numpy as np

# Import progress bar for visual feedback in loops
from tqdm.auto import tqdm

# Import sparse matrix utilities for efficient memory use in matrix operations
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Import dataset loading utilities for handling various data formats
from datasets import load_dataset, Dataset

# Import SentenceTransformer for sentence embeddings using Transformers
from sentence_transformers import SentenceTransformer

# Import multiprocessing tools to utilize multiple CPU cores
from multiprocessing import Pool, cpu_count

# Enable progress_apply in pandas via tqdm for visual feedback on progress
tqdm.pandas()

# Define a list of words commonly used in communication-related content
communication_words = [ ... ]  # truncated for brevity

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
    """
    Function to replace the first 'n' words in a text with the word 'text' if they match any word from communication_words.
    """
    try:
        # Split the text by spaces, only splitting the first_n times
        splits = example.split(' ', maxsplit=first_n)
        # Reconstruct the prefix and suffix from splits
        prefix, suffix = ' '.join(splits[:first_n]), splits[first_n]

        # Replace matched words in prefix with 'text'
        if 'text' not in prefix:
            prefix = regex.sub('text', prefix)
            new_words = []
            # Remove duplicates in prefix
            for w in prefix.split(' '):
                if w not in new_words:
                    new_words.append(w)
            prefix = ' '.join(new_words)

    except IndexError as err:
        return example

    return prefix + ' ' + suffix

def prompt_polishing(prompt):
    """
    Function to refine prompts by applying transformations to make the text more consistent and clearer.
    """
    # Apply the replace_text_word function with increasing window sizes
    prompt = replace_text_word(replace_text_word(replace_text_word(prompt, 3), 5), 6)
    # Define and prepare to substitute phrases that overly specify 'text'
    referencing_words = ["following", "given", "next", "original", "subsequent", "preceding", "consecutive", "provided"]
    referencing_words = ['the ' + w for w in referencing_words] + ['a ' + w for w in referencing_words]

    for refer_word in referencing_words:
        refer_regex = re.compile(refer_word + "\s*\w*\s*text", re.IGNORECASE)
        prompt = refer_regex.sub('this text', prompt)

    for target_word in ['into', 'in', 'as']:
        about_regex = re.compile(rf"\babout\s+.*?{target_word}\b", re.IGNORECASE)
        prompt = about_regex.sub('in', prompt)
    return prompt

# Load dataset, preprocess and deduplicate prompts
sub_prompts = pd.read_parquet('../data/data_train.parquet')['label']

sub_prompts = sub_prompts[sub_prompts.str.len() < 250]
sub_prompts = sub_prompts.str.strip(string.punctuation + '0123456789 ')
sub_prompts = sub_prompts.drop_duplicates()

sub_prompts = sub_prompts.apply(lambda x: x.split('.')[0])
sub_prompts = sub_prompts.apply(lambda s: s.split(':')[-1])
sub_prompts = sub_prompts.str.strip(string.punctuation + '0123456789 ')


prompts = pd.Series(np.array(sub_prompts.tolist() + sub_prompts.apply(prompt_polishing).tolist()))

# Further clean and prepare prompts
pattern = r"\s*,\s*and\s+\w+\s+it\s*$"
prompts = prompts.progress_apply(lambda s: re.sub(pattern, "", s)).drop_duplicates()
prompts = np.array(prompts.tolist())

# Initialize the SentenceTransformer model and generate embeddings
prompt_embedder = SentenceTransformer('sentence-transformers/sentence-t5-base', device='cuda')
prompt_embs = prompt_embedder.encode(prompts, batch_size=32, show_progress_bar=True)

# Utility function to find the top k similar items
def top_k_idx(ranks, topk=3):
    top_part = np.argpartition(-ranks, topk, axis=1)[:, :topk]
    top_values = np.take_along_axis(ranks, top_part, axis=1)
    top_idx = np.take_along_axis(top_part, np.argsort(-top_values, axis=1), axis=1)
    return top_idx

# Compute similarity and process in chunks for efficiency
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

# Assemble the full matrix of distances
indices = np.vstack(indices)
values = np.vstack(values)
row_ind = np.tile(np.arange(len(indices))[:, None], (1, indices.shape[-1]))
cube_values = values.flatten() **3
distance_mat = csr_matrix((cube_values, (row_ind.flatten(), indices.flatten())), shape=(len(prompt_embs), len(prompt_embs)))

# Identify connected components for clustering
n_components, labels = connected_components(distance_mat >= 0.985, connection='weak', directed=False)
print(len(labels),'->', len(set(labels)))

# Select unique prompts based on clustering results
deduplication_mask = np.zeros(len(prompts), dtype=bool)
for lbl in tqdm(set(labels)):
    selection = labels == lbl
    ic_score = np.asarray(distance_mat[selection][:, selection].mean(axis=1)).flatten()
    mask = selection[selection] * False
    mask[np.argmax(ic_score)] = True
    deduplication_mask[selection] = mask

dedup_prompts = prompts[deduplication_mask]
dedup_distance_mat = distance_mat[deduplication_mask][:, deduplication_mask]

# Output the final prompts to a CSV file
pd.DataFrame(dedup_prompts).to_csv('../output/final_prompts.csv')

