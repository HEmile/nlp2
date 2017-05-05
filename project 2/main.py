from cfg import *
from helper import *
from fsa import *
from earley import *
from collections import defaultdict


LIMIT_TRANS_LENGTH = 20

def main():
    chinese, english = read_data('data/training.zh-en')
    lexicon = read_lexicon_ibm('lexicon') #Waarom . bij beide elke entry
    src_cfg = make_source_side_itg(lexicon)
     
    for i in range(len(chinese)):                         
        chi_src = chinese[i]
        eng_tgt = english[i]
        src_fsa = make_fsa(chi_src)
        #print('FSA-Source: \n', src_fsa)
        
        forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)")) 
        #print('Forest: \n', forest)
        
        proj_forest = make_target_side_itg(forest, lexicon)
        tgt_fsa = make_fsa(eng_tgt)
        #print('FSA-target: \n', tgt_fsa)
        
        ref_forest = earley(proj_forest, LimitFSA(LIMIT_TRANS_LENGTH), start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x,y)'))
        #print('Final forest: \n', ref_forest)
        print(len(chi_src), len(eng_tgt))
        if len(ref_forest) > 0:
            print("Possible Derivations:", inside_value(ref_forest))

    
    
if __name__ == '__main__':
    main()

