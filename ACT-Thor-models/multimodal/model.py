
"""
- Setup a siamese-framework to recognize multiple images.
- Have a cross entropy alternative
-
"""

import torch
import torch.nn as nn
import torchvision
import torchtext


class LanguageModel(nn.Module):
    def __init__(self, model='gru', vec_size=300, hidden_size=256, out_size=None, freeze_emb=True, device='cuda'):
        super(LanguageModel, self).__init__()

        self.device = device
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.BEGIN_TOKEN = "<bos>"
        self.END_TOKEN = "<eos>"
        self.PAD_TOKEN = "<pad>"
        self.tok_fn = torchtext.data.get_tokenizer("basic_english")

        self.emb = self._init_vocab(vec_size, freeze_emb).to(self.device)

        if model == "gru":
            self.model = torch.nn.GRU(input_size=vec_size, hidden_size=hidden_size, batch_first=True).to(self.device)
        else:
            raise ValueError(f"unsupported model type {model}")

        self.out_fc = None
        if self.out_size is not None:
            self.out_fc = torch.nn.Linear(self.hidden_size, self.out_size).to(self.device)

    def _init_vocab(self, vec_size, freeze_emb):
        assert vec_size in [50, 100, 200, 300]
        tmpvocab = torchtext.vocab.GloVe('6B', dim=vec_size, )

        # TODO: handle OOV words with a more complex class / use directly GLoVE as embedder, which has its handler but does not support batching (?)
        # TODO: add special tokens (<bos>, <eos>)

        # initializing special tokens by random sampling with mean and variance over all vectors
        mu, sig = tmpvocab.vectors.mean(dim=0), tmpvocab.vectors.std(dim=0)

        # BOS
        tmpvocab.stoi[self.BEGIN_TOKEN] = tmpvocab.vectors.shape[0]
        tmpvocab.vectors = torch.cat([tmpvocab.vectors, (mu + sig * torch.randn_like(tmpvocab.vectors[0])).unsqueeze(0)])

        # EOS
        tmpvocab.stoi[self.END_TOKEN] = tmpvocab.vectors.shape[0]
        tmpvocab.vectors = torch.cat([tmpvocab.vectors, (mu + sig * torch.randn_like(tmpvocab.vectors[0])).unsqueeze(0)])

        # PAD
        tmpvocab.stoi[self.PAD_TOKEN] = tmpvocab.vectors.shape[0]
        tmpvocab.vectors = torch.cat([tmpvocab.vectors, (mu + sig * torch.randn_like(tmpvocab.vectors[0])).unsqueeze(0)])

        # before
        tmpvocab.stoi["<before>"] = tmpvocab.vectors.shape[0]
        tmpvocab.vectors = torch.cat([tmpvocab.vectors, (mu + sig * torch.randn_like(tmpvocab.vectors[0])).unsqueeze(0)])

        # after
        tmpvocab.stoi["<after>"] = tmpvocab.vectors.shape[0]
        tmpvocab.vectors = torch.cat([tmpvocab.vectors, (mu + sig * torch.randn_like(tmpvocab.vectors[0])).unsqueeze(0)])

        self.stoi = tmpvocab.stoi
        self.emb = nn.Embedding.from_pretrained(tmpvocab.vectors, freeze=freeze_emb)
        return self.emb

    def forward(self, batch, hidden=None):
        """
        Calls model forward on a batch; 'hidden' parameter is needed in case of non-null initial hidden state,
        as happens in non-Transformer-based classification modules.
        """

        """
        Alternatively, to use only GLoVE object:

            torch.stack([self.glove.get_vecs_by_token(self.tok_fn(sent)) for sent in batch], dim=0)

        - this would imply having automatic handling of OOV tokens
        - could have slower performance
        """
        sentences = [self.tok_fn(sent) for sent in batch]
        lengths = [len(sent) for sent in sentences]
        indicized_sentences = [torch.LongTensor([self.stoi[tok] for tok in sent]) for sent in sentences]
        indicized_sentences = nn.utils.rnn.pad_sequence(indicized_sentences, batch_first=True, padding_value=self.stoi[self.PAD_TOKEN]).to(self.device)
        text_emb = self.emb(indicized_sentences)
        text_emb = nn.utils.rnn.pack_padded_sequence(text_emb, lengths, enforce_sorted=False)  # actually needed?

        out = self.model(text_emb, hidden)  # should be correct also when hidden is None (as in source code https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html)

        ###################
        # here eventually output transformation when model is an LSTM
        ###################

        # takes last hidden state
        out = out[-1]
        if isinstance(out, torch.nn.utils.rnn.PackedSequence):
            out = nn.utils.rnn.unpack_sequence(out)

        # performs projection if needed
        # TODO: decide if include sequence outputs here
        if self.out_fc is not None:
            out = self.out_fc(out)

        return out


class VisionModel(nn.Module):
    # TODO
    pass


class Speaker(nn.Module):
    def __init__(self):
        super(Speaker, self).__init__()

    def forward(self, x):
        pass

    def beam_inference(self, x):
        pass


class Hearer(nn.Module):

    def __init__(self, distance_fun='cosine_sim'):
        super(Hearer, self).__init__()
        self.lm = LanguageModel()
        self.vm = VisionModel()

        # Define cross modal encoder utilities
        self.__cm_transformer = None

        def cross_modal(ev, et, mode='before'):
            # TODO: define changes according to 'before' vs 'after'

            return -1

        self.cross_modal_encoder = cross_modal

        # Define similarity function for embedding ranking
        if distance_fun == 'cosine_sim':
            self._dist = lambda x, y: torch.cosine_similarity(x, y, dim=-1)
        else:
            raise ValueError('unsupported similarity function')

    def forward(self, sample: dict):
        # sample --> dict {'action': ..., 'before': ...,  'positive': ..., 'negatives': [..., ...]}

        # embeddings
        ev_before = self.vm(sample['before'])
        ev_positive = self.vm(sample['positive'])
        ev_negative = [self.vm(neg) for neg in sample['negatives']]
        et = self.lm(sample['action'])

        # cross-modality
        cross_emb_before = self.cross_modal_encoder(ev_before, et, mode='before')
        cross_emb_positive = self.cross_modal_encoder(ev_positive, et, mode='after')
        cross_emb_negative = [self.cross_modal_encoder(neg, et, mode='after') for neg in ev_negative]

        # distances & ranking
        p_dist = self._dist(cross_emb_before, cross_emb_positive)
        n_dist = [self._dist(cross_emb_before, neg) for neg in cross_emb_negative]

        return torch.argmax(torch.stack([p_dist] + n_dist, dim=-2), dim=-1)

    def process_batch(self, batch, loss_fn):
        pass

    def run_inference(self, sample):
        pass


class ClassificationHearer(nn.Module):
    """
    Hearer that performs action prediction as a classification problem over a fixed set of images.
    """
    # TODO
    pass


class NoTransformerHearer(nn.Module):
    def __init__(self, distance_fun='cosine_sim', use_special_tokens=False, device='cuda'):
        """
        Hearer model that does not include a cross-modal transformer encoder. First, it extracts image feature vector
        with a pretrained visual model, then uses it as hidden input to the recurrent model processing action sentence.
        Args:
            :param distance_fun: method used to compute distance between embeddings
            :param use_special_tokens: whether to add before/after tokens to action sentences
        """
        super(NoTransformerHearer, self).__init__()
        self.use_special_tokens = use_special_tokens
        self.device = device
        self.lm = LanguageModel(device=device)  # language model
        self.vm = VisionModel(out_size=self.lm.hidden_size, device=device)  # vision model

        # Define similarity function for embedding ranking
        if distance_fun == 'cosine_sim':
            self._dist = lambda x, y: torch.cosine_similarity(x, y, dim=-1)
        else:
            raise ValueError('unsupported similarity function')

    def forward(self, sample):
        batch_size = len(sample['action'])

        before_s = sample['action'][:]
        after_s = sample['action'][:]
        if self.use_special_tokens:
            for i in range(batch_size):
                # TODO: decide how to integrate before/after special tokens
                before_s[i] = before_s[i] + " <before>"
                after_s[i] = after_s[i] + " <after>"

        # embeddings (add fake sequence dimension (0) to visual vectors)
        e_before = self.lm(before_s, hidden=self.vm(sample['before']).unsqueeze(0))
        e_positive = self.lm(after_s, hidden=self.vm(sample['positive']).unsqueeze(0))
        e_negative = [self.lm(after_s, hidden=self.vm(neg).unsqueeze(0)) for neg in sample['negatives']]

        # distances
        p_dist = self._dist(e_before, e_positive).view(batch_size, -1)
        n_dist = [self._dist(e_before, neg).view(batch_size, -1) for neg in e_negative]

        # concatenation: tensors are
        return torch.cat([p_dist] + n_dist, dim=1)

    def process_batch(self, batch):
        pass


class DialogModel(nn.Module):
    def __init__(self):
        super(DialogModel, self).__init__()

        # Initialize speaker and hearer
        self.speaker = None
        self.hearer = None

    def forward(self, sample):
        pass

    def process_batch(self, batch, loss_fn):
        pass

    def run_train(self):
        pass

    def train_speaker(self):
        pass

    def train_hearer(self):
        pass

    def run_inference(self, sample):
        pass

    def run_evaluation(self, valid_dl):
        pass


if __name__ == "__main__":
    neg_nr = 4
    use_special_tokens = True

    # Testing - SINGLE
    s = ["the quick brown fox"]
    b = torch.randn(1, 3, 224, 224)
    a = torch.randn_like(b)
    negs = [torch.randn_like(a) for i in range(neg_nr)]
    sample = {'action': s, 'positive': a, 'before': b, 'negatives': negs}
    net = NoTransformerHearer(use_special_tokens=use_special_tokens)
    y = net(sample)
    print(y.shape)

    # Test - BATCH
    s = ["the quick brown fox", "it jumps over the lazy dog"]
    batch_size = len(s)
    b = torch.randn(batch_size, 3, 224, 224)
    a = torch.randn_like(b)
    negs = [torch.randn_like(a) for i in range(neg_nr)]
    sample = {'action': s, 'positive': a, 'before': b, 'negatives': negs}
    yb = net(sample)
    print(yb.shape)

