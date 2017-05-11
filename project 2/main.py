from cfg import *
from helper import *
from fsa import *
from earley import *
from collections import defaultdict
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

    if not os.path.exists('parses'):
        os.makedirs('parses')

    print('Parsing sentences', mn, 'to', mx)
     
    for i in range(len(chinese)):
        index = mn + i
        chi_src = chinese[i]
        en_src = english[i]
        if (len(chi_src) < 10 and len(en_src) < 10) \
                or len(chi_src) > 15 or len(en_src) > 15:
            continue
        print(index)
        src_fsa = make_fsa(chi_src)

        forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)")) 

        proj_forest = make_target_side_itg(forest, lexicon)

        ref_forest = earley(proj_forest, limitfsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x,y)'))

        with open("parses/" + str(index) + '.pkl', 'wb') as f:
            pickle.dump(ref_forest, f)

        #print('Final forest: \n', ref_forest)
        # print(len(chi_src), len(eng_tgt))
        # if len(ref_forest) > 0:
        #     print("Possible Derivations:", inside_value(ref_forest))


if __name__ == '__main__':
    main()

