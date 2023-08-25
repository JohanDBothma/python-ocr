import stow
from tqdm import tqdm

from mltu.dataProvider import DataProvider
from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding

# from model import train_model
from config import ModelConfigs

"""
    For this example, we have two folders in the dataset under IAM_Sentences, ascii and sentences
        -> the ascii folder has a senteces.txt which is has a line of each folder, what the status of the reading is, and what the image should be
        -> the sentences folder is where all the images are, and the idea is to train the model that the image what is pointed in the senteces.txt, should have the text
    An example of a sentence from the ascii would be:
        -> a01-000x-s00-01 0 ok 181 23 382 924 1595 148 any|more|Labour|life|Peers|is|to|be|made|at|a
    From this, we will read the line and split all the parts up to train the model what we expect it to do. Breaking this down further:
        -> a01-000x-s00-01 : This is a type of recursive path to the image, we break this up in our loop and combine the different variables to navigate to the file. It can be broken up as follows
            -> The first 3 are the folder1, a01/
            -> The first 8 is the folder2, a01-000x/
            -> The entire part is the file, a01-000x-s00-01
            -> This results in, a01/a01-00x/a01-000x-s00-01
        -> 0
        -> ok: Indicates that this training set is correct and should be used
        -> 181 23 382 924 1595 148
        -> any|more|Labour|life|Peers|is|to|be|made|at|a: The actual text that the image is
"""

# sentences_txt_path = stow.join('Datasets', 'IAM_Sentences', 'ascii', 'sentences.txt')
# sentences_folder_path = stow.join('Datasets', 'IAM_Sentences', 'sentences')

# Read the data from the training sets
sentences_txt_path = stow.join('Training',  'ascii', 'sentences.txt')
sentences_folder_path = stow.join('Training', 'sentences')

""" 
    Set variables
    dataset -> append valid lines 
    vocab ->
    max_len ->
"""
dataset, vocab, max_len = [], set(), 0
words = open(sentences_txt_path, "r").readlines()
for line in tqdm(words):
    # if the line starts with a hash, skip this line
    if line.startswith("#"):
        continue

    # check if the third index of tha word is an error, if it is, skip this line
    line_split = line.split(" ")
    if line_split[2] == "err":
        continue

    # Get the first 3 characters for the folder
    folder1 = line_split[0][:3]

    # Get the first 8 characters for the sub folder
    folder2 = line_split[0][:8]

    # Get the image name and append .png to it
    fileName = line_split[0] + ".png"

    # Get the last index of the line, and remove the newline character
    text = line_split[-1].rstrip('\n')

    # Replace '|' with ' ' in text
    text = text.replace('|', ' ')

    # Get the full path of the file, if it doesn't exist, move onto the next
    path = stow.join(sentences_folder_path, folder1, folder2, fileName)
    if not stow.exists(path):
        continue

    dataset.append([path, text])
    vocab.update(list(text))
    max_len = max(max_len, len(text))

print(f'dataset: {dataset}')
print(f'vocab: {vocab}')
print(f'max_len: {max_len}')
# Create a ModelConfigs object to store model configurations
# configs = ModelConfigs()

# # Save vocab and maximum text length to configs
# configs.vocab = "".join(vocab)
# configs.max_text_length = max_len
# configs.save()

# # Create a data provider for the dataset
# data_provider = DataProvider(
#     dataset=dataset,
#     skip_validation=True,
#     batch_size=configs.batch_size,
#     data_preprocessors=[ImageReader()],
#     transformers=[
#         ImageResizer(configs.width, configs.height, keep_aspect_ratio=True),
#         LabelIndexer(configs.vocab),
#         LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab)),
#     ],
# )

# # Split the dataset into training and validation sets
# train_data_provider, val_data_provider = data_provider.split(split = 0.9)