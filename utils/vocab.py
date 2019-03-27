import numpy as np
import ipdb
import string
UNK = "<unk>"
PAD = "<pad>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0
PAD_ID = 1
SOS_ID = 2
EOS_ID = 3


def create_vocab_from_docs(docs, max_vocab_size, out_vocab_f_path,
                           out_count_f_path=None,
                           prepend_special_tokens=(UNK, SOS, EOS, PAD)):
    assert docs[0].token_type == "word_type"
    with VocabCreator(
            max_vocab_size, out_vocab_f_path, out_count_f_path,
            prepend_special_tokens) as vocab_creator:
        for doc in docs:
            for token in doc:
                if token not in doc.applied_flag_tokens:
                    vocab_creator.update_vocab_from_word(token)
    return Vocab(out_vocab_f_path, out_count_f_path)
    

class VocabCreator:
    def __init__(self, max_vocab_size, out_vocab_f_path, out_count_f_path=None, 
                 prepend_special_tokens=(UNK, SOS, EOS, PAD)):
        self._opened_vocab_file = open(out_vocab_f_path, "w")
        self._vocab = {}
        self._max_vocab_size = max_vocab_size
        if out_count_f_path:
            self._opened_count_file = open(out_count_f_path, "w")
        else:
            self._opened_count_file = None
        self._special_tokens = prepend_special_tokens

    def __enter__(self):
        return self

    def update_vocab_from_word(self, word):
        if not (word in self._special_tokens):
            self._vocab[word] = self._vocab[word]+1 if word in self._vocab else 1

    def update_vocab_from_sentence(self, sentence):
        for word in sentence.split():
            self.update_vocab_from_word(word)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for special_token in self._special_tokens:
            self._opened_vocab_file.write(special_token + "\n")
            self._opened_count_file.write("0\n")

        actual_vocab_len = self._max_vocab_size-len(self._special_tokens)
        for word, count in sorted(self._vocab.items(),
                                  key=lambda x: x[1],
                                  reverse=True)[:actual_vocab_len]:
            self._opened_vocab_file.write(word + "\n")
            if self._opened_count_file is not None:
                self._opened_count_file.write(str(count) + "\n")

        self._opened_vocab_file.close()
        if self._opened_count_file:
            self._opened_count_file.close()


class Vocab:
    def __init__(self, vocab_f_path, count_f_path=None):
        with open(vocab_f_path) as f:
            words = f.read().strip().split()
            ids = range(len(words))
            self._vocab_size = len(words)
            self._id2word_table = words
            self._word2id_table = dict(zip(words, ids))
            self._vocab_counts = None
            if count_f_path is not None:
                with open(count_f_path) as count_f:
                    counts = count_f.read().strip().split()
                    self._vocab_counts = [int(count) for count in counts]

    def __getitem__(self, word_token):
        return self._word2id_table.get(word_token, UNK_ID)

    def id2word(self, id_token):
        if id_token >= self.vocab_size:
            return UNK
        return self._id2word_table[id_token]

    def word2id(self, word_token):
        return self._word2id_table.get(word_token, UNK_ID)

    def check_word_exist(self, word_token):
        return word_token in self._word2id_table

    def check_id_exist(self, id_token):
        return (id_token >= 0) and (id_token < len(self._id2word_table))

    @property
    def vocab_size(self):
        return self._vocab_size

    def __len__(self):
        return self._vocab_size

    @property
    def vocab_counts_list(self):
        return self._vocab_counts

    @property
    def pad_token(self):
        return PAD

    @property
    def pad_id(self):
        return PAD_ID

    @property
    def unk_id(self):
        return UNK_ID


class LazyVocab(Vocab):
    def __init__(self, vocab_name, max_size, allow_unk, *preallocated):
        assert max_size >= 2
        if max_size < len(preallocated) + 1:
            raise ValueError("not enough size for pre allocation")
        self._max_size = max_size
        self._unk_id = 0
        self._vocab = {UNK: UNK_ID, PAD: PAD_ID}
        self._curr_id = 2
        for token in preallocated:
            self._vocab[token] = self._curr_id
            self._curr_id += 1
        self._rev_vocab = [UNK, PAD]
        self._allow_unk = allow_unk
        self._vocab_name = vocab_name

    def __getitem__(self, token):
        token_id = self._vocab.get(token)
        # pdb.set_trace()
        if token_id is None:
            if self._curr_id < self._max_size:
                token_id = self._curr_id
                self._vocab[token] = token_id
                self._rev_vocab.append(token)
                self._curr_id += 1
                return token_id
            else:
                if self._allow_unk:
                    return self._unk_id
                else:
                    print("Stored vocab")
                    print(self._vocab)
                    raise ValueError("Vocab %s exploded %s could not be stored"%(self._vocab_name, token))
        else:
            return token_id

    def __len__(self):
        return self._curr_id

    def __str__(self):
        v_str = ""
        for token, token_id in sorted(self._vocab.items(), key=lambda x:x[1]):
            v_str += (token + ": " + str(token_id) + "\n")
        return v_str

    @property
    def max_size(self):
        return self._max_size

    def mask_unk(self, token):
        if token not in self._vocab:
            return UNK
        else:
            return token

    def rev_lookups(self, token_ids):
        rev_dict = {w_id: w for w, w_id in self._vocab.items()}
        return [rev_dict[token_id] for token_id in token_ids]

    @property
    def labels(self):
        return self._rev_vocab


class LazyCharVocab(LazyVocab):
    def __init__(self, extra_size=0):
        preallocated = [letter for letter in string.ascii_letters] + ['0', '1', '2','3','4','5','6','7','8','9']
        max_size = len(preallocated) + extra_size +2
        super(LazyCharVocab, self).__init__(max_size, *preallocated)

    # def __getitem__(self, str_token):
        """
        Only takes in sentence string, returns list
        """
        # return [super(LazyCharVocab, self).__getitem__(letter) for letter in str_token]
        """
        if len(token) > 1 and token!=PAD and token!=UNK:
            ipdb.set_trace()
            return [super(LazyCharVocab, self).__getitem__(letter) for letter in token]
        else:
            return super(LazyCharVocab, self).__getitem__(token)
        """
    def mask_unk(self, token):
        letters = []
        for letter in token:
            if letter not in self._vocab:
                letters.append(UNK)
            else:
                letters.append(letter)
        return letters


def create_lazy_char_vocab(extra_size=0):
    preallocated = [letter for letter in string.ascii_letters] + ['0', '1', '2','3','4','5','6','7','8','9']
    max_size = len(preallocated) + extra_size +2
    return LazyVocab(max_size, *preallocated)


class Embed(Vocab):
    def __init__(self, vocab_f_path, embed_f_path, count_f_path=None):
        super(Embed, self).__init__(vocab_f_path, count_f_path)
        self._id2embed_table = np.load(embed_f_path)
        assert self._id2embed_table.shape[0] == self.vocab_size

    def id2embed_lookup(self, id_token):
        if id_token >= self.vocab_size:
            return self._id2embed_table[UNK]
        return self._id2embed_table[id_token]

    def word2embed_lookup(self, word_token):
        id_token = self._word2id_table.get(word_token, UNK_ID)
        return self.id2embed_lookup[id_token]

    @property
    def embed_size(self):
        return self._id2embed_table.shape[1]

