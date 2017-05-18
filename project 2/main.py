from helper import *
from earley import *
from libitg import *
from collections import defaultdict
from feature_helper import gradient, skip_bigrams
import pickle
import os


LIMIT_TRANS_LENGTH = 3

PARTITION = 1

DATA_SET_INDEX = 0 #Divide dataset in 9 partitions

def main():
    chinese, english = read_data('data/training.zh-en')
    skip_dict = skip_bigrams(chinese)
    mn, mx = DATA_SET_INDEX * (len(chinese) // PARTITION), (DATA_SET_INDEX + 1) * (len(chinese) // PARTITION)
    chinese, english = chinese[mn: mx], english[mn: mx]
    lexicon, weights = read_lexicon_ibm('lexicon')
    src_cfg = make_source_side_itg(lexicon)
    limitfsa = InsertionConstraint(LIMIT_TRANS_LENGTH)

    w = defaultdict(lambda: 1) #Initialize the weight dictionary with 1s
    delta = 0.001

    if not os.path.exists('parses'):
        os.makedirs('parses')

    print('Parsing sentences', mn, 'to', mx)
     
    for i in range(len(chinese)):
        index = mn + i
        chi_src = chinese[i]
        en_src = english[i]
        if len(chi_src) > 20 or len(en_src) > 20:
            continue
        print(index)
        src_fsa = make_fsa(chi_src)
        tgt_fsa = make_fsa(en_src)

        forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)"))
        dx = earley(forest, limitfsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('Di(x)'), eps_symbol=None)
        print(dx)

        dix = make_target_side_itg(dx, lexicon)

        dxy = earley(dix, tgt_fsa, start_symbol=Nonterminal("Di(x)"), sprime_symbol=Nonterminal('D(x, y)'))
        if len(dxy) == 0:
            print('Skipping ungenerated y')
            continue

        dw = gradient(dix, dxy, src_fsa, w, weights)

        for k, dwk in dw.items():
            w[k] += delta * dwk


if __name__ == '__main__':
    main()

