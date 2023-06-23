sentences_txt_path = stow.join('Datasets', 'IAM_Sentences', 'ascii', 'sentences.txt')
sentences_folder_path = stow.join('Datasets', 'IAM_Sentences', 'sentences')

dataset, vocab, max_len = [], set(), 0
words = open(sentences_txt_path, "r").readlines()
for line in tqdm(words):
    if line.startswith("#"):
        continue

    line_split = line.split(" ")
    if line_split[2] == "err":
        continue

    folder1 = line_split[0][:3]
    folder2 = line_split[0][:8]
    file_name = line_split[0] + ".png"
    label = line_split[-1].rstrip('\n')

    # recplace '|' with ' ' in label
    label = label.replace('|', ' ')

    rel_path = stow.join(sentences_folder_path, folder1, folder2, file_name)
    if not stow.exists(rel_path):
        continue

    dataset.append([rel_path, label])
    vocab.update(list(label))
    max_len = max(max_len, len(label))