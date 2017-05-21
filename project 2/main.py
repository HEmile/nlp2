from helper import *
from earley import *
from libitg import *
from collections import defaultdict
from feature_helper import gradient, skip_bigrams
import pickle
import os


LIMIT_TRANS_LENGTH = 3

PARTITION = 2

DATA_SET_INDEX = 4 #Divide dataset in 9 partitions

SENTENCE_LENGTH = 10


def main(parse=False, featurise=True):
    chinese, english = read_data('data/training.zh-en')
    skip_dict = skip_bigrams(chinese)
    mn, mx = DATA_SET_INDEX * (len(chinese) // PARTITION), (DATA_SET_INDEX + 1) * (len(chinese) // PARTITION)
    chinese, english = chinese[mn: mx], english[mn: mx]
    lexicon, weights, ch_vocab, en_vocab, null_alligned = read_lexicon_ibm('lexicon')

    w = defaultdict(lambda: 0.0000001) #Initialize the weight dictionary with 1s
    delta = 0.000000001

    if not os.path.exists('parses'):
        os.makedirs('parses')

    if not os.path.exists('features'):
        os.makedirs('features')

    print('Parsing sentences', mn, 'to', mx)
    likelihood = []
    count = 0
    for i in range(len(chinese)):
        index = mn + i
        chi_src = chinese[i]
        en_src = english[i]
        chi_spl = chi_src.split()
        en_spl = en_src.split()

        if len(chi_spl) > SENTENCE_LENGTH or len(en_spl) > SENTENCE_LENGTH:
            continue

        path = "parses/" + str(index) + '.pkl'

        if parse:
            def map_unk(splt, vocab):
                for i in range(len(splt)):
                    if splt[i] not in vocab:
                        splt[i] = '-UNK-'
                return ' '.join(splt)
            chi_src = map_unk(chi_spl, ch_vocab)
            en_src = map_unk(en_spl, en_vocab)

            lexicon['-EPS-'] = set(null_alligned)
            for c in chi_spl:  # Belangrijk voor report: Deze toevoegen zorgt ervoor dat heel veel parset
                lexicon['-EPS-'] = lexicon['-EPS-'].union([lexicon[c][0]])

            src_fsa = make_fsa(chi_src)
            tgt_fsa = make_fsa(en_src)

            src_cfg = make_source_side_finite_itg(lexicon)

            forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)"))

            dx = make_target_side_finite_itg(forest, lexicon)

            dxy = earley(dx, tgt_fsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x, y)'))

            if len(dxy) == 0:
                continue

            with open(path, 'wb') as f:
                pickle.dump((dx, dxy), f)
        else:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    dx, dxy = pickle.load(f)
            else:
                continue

        count += 1
        dw, likel = gradient(dx, dxy, src_fsa, w, weights, skip_dict, index, featurise)
        likelihood.append(likel)
        if dw:
            for k, dwk in dw.items():
                w[k] += delta * dwk
        if count % 50 == 0:
            print(index)
            print(sum(likelihood) / count)
            likelihood = []


if __name__ == '__main__':
    main()

