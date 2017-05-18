from helper import *
from libitg import *
from collections import defaultdict
from feature_helper import gradient
import pickle
import os


LIMIT_TRANS_LENGTH = 15

PARTITION = 1

DATA_SET_INDEX = 0 #Divide dataset in 9 partitions

def main():
    chinese, english = read_data('data/training.zh-en')
    mn, mx = DATA_SET_INDEX * (len(chinese) // PARTITION), (DATA_SET_INDEX + 1) * (len(chinese) // PARTITION)
    chinese, english = chinese[mn: mx], english[mn: mx]
    lexicon, weights = read_lexicon_ibm('lexicon') #Waarom . bij beide elke entry
    src_cfg = make_source_side_itg(lexicon)
    limitfsa = LengthConstraint(LIMIT_TRANS_LENGTH)

    w = defaultdict(lambda: 1) #Initialize the weight dictionary with 1s
    delta = 0.001

    if not os.path.exists('parses'):
        os.makedirs('parses')

    print('Parsing sentences', mn, 'to', mx)
     
    for i in range(len(chinese)):
        index = mn + i
        chi_src = chinese[i]
        en_src = english[i]
        if len(chi_src) > 15 or len(en_src) > 15:
            continue
        print(index)
        src_fsa = make_fsa(chi_src)
        tgt_fsa = make_fsa(en_src)

        forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)"))

        proj_forest = make_target_side_itg(forest, lexicon)

        dxy = earley(proj_forest, tgt_fsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x, y)'))
        if len(dxy) == 0:
            print('Skipping ungenerated y')
            continue
        dxn = earley(proj_forest, limitfsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('Dn(x)'))
        dw = gradient(dxn, dxy, src_fsa, w)

        for k, dwk in dw.items():
            w[k] += delta * dwk



if __name__ == '__main__':
    main()

