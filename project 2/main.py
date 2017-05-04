from cfg import *
from helper import *
from fsa import *
from earley import *
from collections import defaultdict

'''
S = Nonterminal('S')
X = Nonterminal('X')
a = Terminal('a')
r1 = Rule(S, [X])
r2 = Rule(X, [X, X])
r3 = Rule(X, [a])
print('Symbols')
for sym in [S, X, a]:
    print(sym)
print('Rules')
for r in [r1, r2, r3]:
    print(r)

lexicon = defaultdict(set)
lexicon['le'].update(['the', '-EPS-'])  # we will assume that `le` can be deleted
lexicon['-EPS-'].update(['a', 'the'])  # we will assume that `the` and `a` can be inserted
lexicon['e'].add('and')
lexicon['chien'].add('dog')
lexicon['noir'].update(['black', 'noir'])
lexicon['blanc'].add('white')
lexicon['petit'].update(['small', 'little'])
lexicon['petite'].update(['small', 'little'])
src_cfg = make_source_side_itg(lexicon)
print(src_cfg)
print(make_fsa('le chien noir'))

src_str = 'petit chien'
src_fsa = make_fsa(src_str)
print(src_fsa)
print(src_fsa)
print(src_cfg)

# here I am going to use [S'] as the new start symbol
forest = earley(src_cfg, src_fsa, start_symbol=Nonterminal('S'), sprime_symbol=Nonterminal("D(x)"))
print(forest)
len(forest)

# This is D(x)
projected_forest = make_target_side_itg(forest, lexicon)
len(projected_forest)
tgt_str = 'little dog'
tgt_fsa = make_fsa(tgt_str)
print(tgt_fsa)

# This is D(x, y)
ref_forest = earley(projected_forest, tgt_fsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x,y)'))
print(ref_forest)
len(ref_forest)
'''

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
        
        ref_forest = earley(proj_forest, tgt_fsa, start_symbol=Nonterminal("D(x)"), sprime_symbol=Nonterminal('D(x,y)'))
        #print('Final forest: \n', ref_forest)
        print(len(ref_forest))
    
    
if __name__ == '__main__':
    main()