from cfg import *
from helper import *
from fsa import *
from earley import *
from collections import defaultdict
import pickle


LIMIT_TRANS_LENGTH = 30

DATA_SET_INDEX = 2 #Divide dataset in 9 partitions

def main():
    chinese, english = read_data('data/training.zh-en')
    mn, mx = DATA_SET_INDEX * (len(chinese) // 9), (DATA_SET_INDEX + 1) * (len(chinese) // 9)
    chinese, english = chinese[mn: mx], english[mn: mx]
    lexicon = read_lexicon_ibm('lexicon') #Waarom . bij beide elke entry
    src_cfg = make_source_side_itg(lexicon)
    limitfsa = LimitFSA(LIMIT_TRANS_LENGTH)

    forests = []
    print('Parsing sentences', mn, 'to', mx)
     
    for i in range(len(chinese)):
        print(i)
        chi_src = chinese[i]
        src_fsa = make_fsa(chi_src)

        forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)")) 

        proj_forest = make_target_side_itg(forest, lexicon)

        ref_forest = earley(proj_forest, limitfsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x,y)'))
        forests.append(ref_forest)

    with open("parse" + str(DATA_SET_INDEX) + '.pkl', 'wb') as f:
        pickle.dump(forests, f)
        #print('Final forest: \n', ref_forest)
        # print(len(chi_src), len(eng_tgt))
        # if len(ref_forest) > 0:
        #     print("Possible Derivations:", inside_value(ref_forest))


    
    
if __name__ == '__main__':
    main()

