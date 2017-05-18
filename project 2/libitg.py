"""
Here we implement ITG-related algorithms to fit a LV-CRF.
:author: Wilker Aziz
"""

from formal import *
from alg import * 
from earley import *
from time import time
import numpy as np
import sys


# # ITG
# 
# We do not really need a special class for ITGs, they are just a generalisation of CFGs for multiple streams.
# What we can do is to treat the source side and the target side of the ITG as CFGs.
# 
# We will represent a lexicon
# 
# * a collection of translation pairs \\((x, y) \in \Sigma \times \Delta\\) where \\(\Sigma\\) is the source vocabulary and \\(\Delta\\) is the target vocabulary
# * these vocabularies are extended with an empty string, i.e., \\(\epsilon\\)
# * we will assume the lexicon expliclty states which words can be inserted/deleted 
# 
# We build the source side by inspecting a lexicon
# 
# * terminal rules: \\(X \rightarrow x\\) where \\(x \in \Sigma\\)
# * binary rules: \\(X \rightarrow X ~ X\\)
# * start rule: \\(S \rightarrow X\\)
# 
# Then, when the time comes, we will project this source grammar using the lexicon
# 
# * terminal rules of the form \\(X_{i,j} \rightarrow x\\) will become \\(X_{i,j} \rightarrow y\\) for every possible translation pair \\((x, y)\\) in the lexicon
# * binary rules of the form \\(X_{i,k} \rightarrow X_{i,j} ~ X_{j,k}\\) will be copied and also inverted as in \\(X_{i,k} \rightarrow X_{j,k} ~ X_{i,j}\\)
# * the start rule will be copied
# 

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

        
def make_fsa(string: str) -> FSA:
    """Converts a sentence (string) to an FSA (labels are python str objects)"""
    fsa = FSA()
    fsa.add_state(initial=True)
    for i, word in enumerate(string.split()):
        fsa.add_state()  # create a destination state 
        fsa.add_arc(i, i + 1, word)  # label the arc with the current word
    fsa.make_final(fsa.nb_states() - 1)
    return fsa


# # Target side of the ITG
# 
# Now we can project the forest onto the target vocabulary by using ITG rules.

def make_target_side_itg(source_forest: CFG, lexicon: dict) -> CFG:
    """Constructs the target side of an ITG from a source forest and a dictionary"""    
    def iter_rules():
        for lhs, rules in source_forest.items():            
            for r in rules:
                if r.arity == 1:  # unary rules
                    if r.rhs[0].is_terminal():  # terminal rules
                        x_str = r.rhs[0].root().obj()  # this is the underlying string of a Terminal
                        targets = lexicon.get(x_str, set())
                        yield Rule(r.lhs, [r.rhs[0].translate('-EPS-')])
                        if not targets:
                            yield Rule(r.lhs, [r.rhs[0].translate('-UNK-')])
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


# # Legth constraint
# 
# To constrain the space of derivations by length we can parse a special FSA using the forest that represents \\(D(x)\\), i.e. `tgt_forest` in the code above.
# 
# For maximum lenght \\(n\\), this special FSA must accept the language \\(\Sigma^0 \cup \Sigma^1 \cup \cdots \cup \Sigma^n\\). You can implement this FSA designing a special FSA class which never rejects a terminal (for example by defining a *wildcard* symbol).
# 


class LengthConstraint(FSA):
    """
    This implement an automaton that accepts strings containing up to n (non-empty) symbols.
    """
    
    def __init__(self, n: int, strict=False, wildcard_str='-WILDCARD-'):
        """
        :param n: length constraint
        :param strict: if True, accepts the language \Sigma^n, if False, accepts union of \Sigma^i for i from 0 to n
        """
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        super(LengthConstraint, self).__init__()
        assert n > 0, 'We better use n > 0.'
        self.add_state(initial=True, final=not strict)  # we start by adding an initial state
        for i in range(n):
            self.add_state(final=not strict)  # then we add a state for each unit of length
            self.add_arc(i, i + 1, wildcard_str)  # and an arc labelled with a WILDCARD
        # we always make the last state final
        self.make_final(n)
        self._wildcard_str = wildcard_str
                
    def destinations(self, origin: int, label: str) -> set:
        """Return the destination from a certain `origin` state with a certain `label` (-1 means no destination available)"""
        if origin < self.nb_states():
            return super(LengthConstraint, self).destinations(origin, self._wildcard_str)
        else:
            return set()


class InsertionConstraint(FSA):
    """
    This implements an automaton that accepts up to n insertions.
    For this you need to make Earley think that -EPS- is a normal terminal,
        you can do that by setting eps_symbol to None when calling earley.
    """
    
    def __init__(self, n: int, strict=False, eps_str='-EPS-', wildcard_str='-WILDCARD-'):
        """
        :param n: length constraint
        :param strict: if True, accepts exactly n insertions, if False, accepts up to n insertions.
        """
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        super(InsertionConstraint, self).__init__()
        assert n >=0 , 'We better use n > 0.'
        self.add_state(initial=True, final=not strict)  # we start by adding an initial state
        self.add_arc(0, 0, wildcard_str)
        for i in range(n):
            self.add_state(final=not strict)  # then we add a state for each unit of length
            self.add_arc(i, i + 1, eps_str)  # and an arc labelled with a WILDCARD
            self.add_arc(i + 1, i + 1, wildcard_str)  # and an arc labelled with a WILDCARD
        # we always make the last state final
        self.make_final(n)
        self._eps_str = eps_str
        self._wildcard_str = wildcard_str
                
    def destinations(self, origin: int, label: str) -> set:
        """Return the destination from a certain `origin` state with a certain `label` (-1 means no destination available)"""
        if origin < self.nb_states():
            if label == self._eps_str:
                return super(InsertionConstraint, self).destinations(origin, label)
            else:  # if not eps, we match any word
                return super(InsertionConstraint, self).destinations(origin, self._wildcard_str)
        else:
            return set()