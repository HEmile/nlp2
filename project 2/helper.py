from cfg import *
from fsa import *
from collections import defaultdict
from operator import itemgetter

def read_data(path):
    chinese = []
    english = []
    with open(path, encoding = 'utf-8') as f:
        for l in f:
            l = l.strip('\n')
            sent = l.split(' ||| ')
            chinese.append(sent[0])
            english.append(sent[1])
    return chinese, english

def read_lexicon_ibm(path, cut_vocab = 5):
    lexicon = defaultdict(set)
    weights = defaultdict(float)
    with open(path, encoding='utf-8') as istream:
        for n, line in enumerate(istream):
            line = line.strip()
            if not line:
                continue
            words = line.split()
            x, y, ibm1, ibm2 = words
            try:
                lexicon[x].add((y, float(ibm1)))
            except ValueError:
                pass
    for x in lexicon.keys():
        lexicon[x] = sorted(lexicon[x], reverse=True, key=itemgetter(1))
        tot = sum([z[1] for z in lexicon[x]])
        lexicon[x] = lexicon[x][0:min(cut_vocab, len(lexicon[x]))]
        for y, ibm1 in lexicon[x]:
            weights[x, y] = ibm1 / tot
        lexicon[x] = [y[0] for y in lexicon[x]]
    return lexicon, weights

def read_lexicon(path):
    """
    Read translation dictionary from a file (one word pair per line) and return a dictionary
    mapping x \in \Sigma to a set of y \in \Delta
    """
    lexicon = defaultdict(set)
    with open(path) as istream:
        for n, line in enumerate(istream):
            line = line.strip()
            if not line:
                continue
            words = line.split()
            if len(words) != 2:
                raise ValueError('I expected a word pair in line %d, got %s' % (n, line))
            x, y = words
            lexicon[x].add(y)
    return lexicon

def make_source_side_itg(lexicon, s_str='S', x_str='X') -> CFG:
    """Constructs the source side of an ITG from a dictionary"""
    S = Nonterminal(s_str)
    X = Nonterminal(x_str)
    def iter_rules():
        yield Rule(S, [X])  # Start: S -> X
        yield Rule(X, [X, X])  # Segment: X -> X X
        for x in lexicon.keys():
            yield Rule(X, [Terminal(x)])  # X - > x
    return CFG(iter_rules())

def make_target_side_itg(source_forest: CFG, lexicon: dict) -> CFG:
    """Constructs the target side of an ITG from a source forest and a dictionary"""
    def iter_rules():
        for lhs, rules in source_forest.items():
            for r in rules:
                if r.arity == 1:  # unary rules
                    if r.rhs[0].is_terminal():  # terminal rules
                        x_str = r.rhs[0].root().obj()  # this is the underlying string of a Terminal
                        targets = lexicon.get(x_str, set())
                        if not targets:
                            pass  # TODO: do something with unknown words?
                        else:
                            for y_str in targets:
                                yield Rule(r.lhs, [r.rhs[0].translate(y_str)])  # translation
                    else:
                        yield r  # nonterminal rules
                elif r.arity == 2:
                    yield r  # monotone
                    if r.rhs[0] != r.rhs[1]:  # avoiding some spurious derivations by blocking invertion of identical spans
                        yield Rule(r.lhs, [r.rhs[1], r.rhs[0]])  # inverted
                else:
                    raise ValueError('ITG rules are unary or binary, got %r' % r)
    return CFG(iter_rules())