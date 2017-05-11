from cfg import *
from collections import defaultdict


def get_terminal_string(symbol: Symbol):
    """Returns the python string underlying a certain terminal (thus unwrapping all span annotations)"""
    if not symbol.is_terminal():
        raise ValueError('I need a terminal, got %s of type %s' %
                         (symbol, type(symbol)))
    return symbol.root().obj()


def get_bispans(symbol: Span):
    """
    Returns the bispans associated with a symbol. 

    The first span returned corresponds to paths in the source FSA (typically a span in the source sentence),
     the second span returned corresponds to either
        a) paths in the target FSA (typically a span in the target sentence)
        or b) paths in the length FSA
    depending on the forest where this symbol comes from.
    """
    if not isinstance(symbol, Span):
        raise ValueError('I need a span, got %s of type %s' %
                         (symbol, type(symbol)))
    s, start2, end2 = symbol.obj()  # this unwraps the target or length annotation
    _, start1, end1 = s.obj()  # this unwraps the source annotation
    return (start1, end1), (start2, end2)


def simple_features(edge: Rule, src_fsa: FSA, eps=Terminal('-EPS-'),
                    sparse_del=False, sparse_ins=False, sparse_trans=False) -> dict:
    """
    Featurises an edge given
        * rule and spans
        * src sentence as an FSA
        * TODO: target sentence length n
        * TODO: extract IBM1 dense features
    crucially, note that the target sentence y is not available!    
    """
    fmap = defaultdict(float)
    if len(edge.rhs) == 2:  # binary rule
        fmap['type:binary'] += 1.0
        # here we could have sparse features of the source string as a function
        # of spans being concatenated
        (ls1, ls2), (lt1, lt2) = get_bispans(edge.rhs[0])  # left of RHS
        (rs1, rs2), (rt1, rt2) = get_bispans(edge.rhs[1])  # right of RHS
        # TODO: double check these, assign features, add some more
        if ls1 == ls2:  # deletion of source left child
            pass
        if rs1 == rs2:  # deletion of source right child
            pass
        if ls2 == rs1:  # monotone
            pass
        if ls1 == rs2:  # inverted
            pass
    else:  # unary
        symbol = edge.rhs[0]
        if symbol.is_terminal():  # terminal rule
            fmap['type:terminal'] += 1.0
            # we could have IBM1 log probs for the traslation pair or ins/del
            (s1, s2), (t1, t2) = get_bispans(symbol)
            src_word = src_fsa.label(s1, s2)
            tgt_word = get_terminal_string(symbol)
            if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
                fmap['type:deletion'] += 1.0
                # dense versions (for initial development phase)
                # TODO: use IBM1 prob
                # ff['ibm1:del:logprob'] +=
                # sparse version
                if sparse_del:
                    fmap['del:%s' % src_word] += 1.0
            else:
                if s1 == s2:  # has not consumed any source word, must be an eps rule
                    fmap['type:insertion'] += 1.0
                    # dense version
                    # TODO: use IBM1 prob
                    # ff['ibm1:ins:logprob'] +=
                    # sparse version
                    if sparse_ins:
                        fmap['ins:%s' % tgt_word] += 1.0
                else:
                    fmap['type:translation'] += 1.0
                    # dense version
                    # TODO: use IBM1 prob
                    # ff['ibm1:x2y:logprob'] +=
                    # ff['ibm1:y2x:logprob'] +=
                    # sparse version
                    if sparse_trans:
                        fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0
        else:  # S -> X
            fmap['top'] += 1.0
    return fmap


def featurize_edges(forest, src_fsa,
                    sparse_del=False, sparse_ins=False, sparse_trans=False,
                    eps=Terminal('-EPS-')) -> dict:
    edge2fmap = dict()
    for edge in forest:
        edge2fmap[edge] = simple_features(
            edge, src_fsa, eps, sparse_del, sparse_ins, sparse_trans)
    return edge2fmap


def weight_function(edge, fmap, wmap) -> float:
    pass  # dot product of fmap and wmap  (working in log-domain)


def top_sort(forest: CFG) -> list:
    """Returns ordered list of nodes according to topsort order in an acyclic forest"""
    pass


def inside_algorithm(forest: CFG, tsort: list, edge_weights: dict) -> dict:
    """Returns the inside weight of each node"""
    pass


def outside_algorithm(forest: CFG, tsort: list, edge_weights: dict, inside: dict) -> dict:
    """Returns the outside weight of each node"""
    pass


def expected_feature_vector(forest: CFG, inside: dict, outside: dict, edge_features: dict) -> dict:
    """Returns an expected feature vector (here a sparse python dictionary)"""
    pass
