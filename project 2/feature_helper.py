from libitg import *
from collections import defaultdict
from nltk.util import skipgrams
import numpy as np
import sys
from alg import *


def get_terminal_string(symbol: Symbol):
    """Returns the python string underlying a certain terminal (thus unwrapping all span annotations)"""
    if not symbol.is_terminal():
        raise ValueError('I need a terminal, got %s of type %s' % (symbol, type(symbol)))
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
        raise ValueError('I need a span, got %s of type %s' % (symbol, type(symbol)))
    s, start2, end2 = symbol.obj()  # this unwraps the target or length annotation
    _, start1, end1 = s.obj()  # this unwraps the source annotation
    return (start1, end1), (start2, end2)


def get_source_word(fsa: FSA, origin: int, destination: int) -> str:
    """Returns the python string representing a source word from origin to destination (assuming there's a single one)"""
    labels = list(fsa.labels(origin, destination))
    assert len(labels) == 1, 'Use this function only when you know the path is unambiguous, found %d labels %s for (%d, %d)' % (len(labels), labels, origin, destination)
    return labels[0]


def get_target_word(symbol: Symbol):
    """Returns the python string underlying a certain terminal (thus unwrapping all span annotations)"""
    if not symbol.is_terminal():
        raise ValueError('I need a terminal, got %s of type %s' % (symbol, type(symbol)))
    return symbol.root().obj()


def simple_features(edge: Rule, src_fsa: FSA, weights_ibm, skip_dict, eps=Terminal('-EPS-'),
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
        # here we could have sparse features of the source string as a function of spans being concatenated
        (ls1, ls2), (lt1, lt2) = get_bispans(edge.rhs[0])  # left of RHS
        (rs1, rs2), (rt1, rt2) = get_bispans(edge.rhs[1])  # right of RHS        
        # TODO: double check these, assign features, add some more
        if ls1 == ls2:  # deletion of source left child
            fmap['type:del_lhs'] += 1.0
        if rs1 == rs2:  # deletion of source right child
            fmap['type:del_rhs'] += 1.0
        if ls2 == rs1:  # monotone
            fmap['type:mon'] += 1.0
        if ls1 == rs2:  # inverted
            fmap['type:inv'] += 1.0
        
        #Span features:
        fmap['type:span_source_lhs'] += (ls2-ls1)
        fmap['type:span_source_rhs'] += (rs2-rs1)
        fmap['type:span_target_lhs'] += (lt2-lt1)
        fmap['type:span_target_rhs'] += (rt2-rt1)
        
    else:  # unary
        symbol = edge.rhs[0]
        if symbol.is_terminal():  # terminal rule
            fmap['type:terminal'] += 1.0
            # we could have IBM1 log probs for the traslation pair or ins/del
            (s1, s2), (t1, t2) = get_bispans(symbol)
            if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
                src_word = get_source_word(src_fsa, s1, s2)
                fmap['type:deletion'] += 1.0
                # dense versions (for initial development phase)
                # TODO: use IBM1 prob
                if (src_word, eps) in weights_ibm.keys():
                    ibm_prob = weights_ibm[(src_word, eps)]
                    fmap['ibm1:del:logprob'] += ibm_prob
                # sparse version
                if sparse_del:
                    fmap['del:%s' % src_word] += 1.0
            else:
                tgt_word = get_target_word(symbol)
                if s1 == s2:  # has not consumed any source word, must be an eps rule
                    fmap['type:insertion'] += 1.0
                    # dense version
                    # TODO: use IBM1 prob
                    if (eps, tgt_word) in weights_ibm.keys():
                        ibm_prob = weights_ibm[(eps, tgt_word)]
                        map['ibm1:ins:logprob'] += ibm_prob                    
                    # sparse version
                    if sparse_ins:
                        fmap['ins:%s' % tgt_word] += 1.0
                else:
                    src_word = get_source_word(src_fsa, s1, s2)
                    fmap['type:translation'] += 1.0
                    # dense version
                    # TODO: use IBM1 prob
                    if (src_word, tgt_word) in weights_ibm.keys():
                        ibm_prob = weights_ibm[(src_word, tgt_word)]
                        fmap['ibm1:x2y:logprob'] += ibm_prob
                    #ff['ibm1:y2x:logprob'] += 
                    # sparse version                    
                    if sparse_trans:
                        fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0
                            
                    #skip bigrams:
                    for key in skip_dict.keys():
                        if src_word in key:
                            fmap['skip:%s' % (src_word)] += skip_dict[key]
        else:  # S -> X
            fmap['top'] += 1.0
    return fmap


def skip_bigrams(chinese) -> dict:
    skip_dict = defaultdict(int)
    for sen in chinese:
        skips = list(skipgrams(sen.split(),2, 1))
        for skip in skips:
            skip_dict[skip] += 1
    return skip_dict

def featurize_edges(forest, src_fsa, weights_ibm, skip_dict,
                    sparse_del=False, sparse_ins=False, sparse_trans=False,
                    eps=Terminal('-EPS-')) -> dict:
    edge2fmap = dict()
    for edge in forest:
        edge2fmap[edge] = simple_features(edge, src_fsa, weights_ibm, skip_dict, eps, sparse_del, sparse_ins, sparse_trans)
    return edge2fmap

# Returns the dot product of the weights
def weight_function(edge, fmap, wmap) -> float:
    sum = 0
    for k, v in fmap[edge].items():
        sum += wmap[k] * v
    return sum


def get_weight_f(fmap, wmap):
    return lambda x: weight_function(x, fmap, wmap)


def weight(rule: Rule, chinese: list, weights: dict, cfg: CFG):
    if len(rule.rhs) == 1:
        t = rule.rhs[0]
        if t in cfg.terminals:
            cn = chinese[rule.lhs._start]
            return weights[cn, t.root().obj()]
    return 1


def toposort(cfg: CFG):
    S = set(cfg.nonterminals)
    S = S.union(cfg.terminals)
    # for rule in cfg:
    #     for symbol in rule.rhs:
    #         S.remove(symbol)
    L = []
    temp = set()
    def visit(n):
        if n in temp:
            print('ERROR: Not a cyclic graph!')
        elif n in S:
            temp.add(n)
            for rule in cfg.get(n):
                for m in rule.rhs:
                    visit(m)
            temp.remove(n)
            S.remove(n)
            L.append(n)

    while S:
        n = S.pop()
        S.add(n)
        visit(n)
    return L


# Instead of passing the weights dictionary here, we need to compute the _log potential_
# This is the dot product of the feature weights and the feature vector at some edge
# This feature vector is created for each edge before this.
def inside_value(cfg: CFG, fweight):
    std = toposort(cfg)
    Iplus = {}
    Imax = {}
    for v in std:
        if v in cfg.terminals:
            Iplus[v] = 0
        elif v in cfg.nonterminals:
            rules = cfg.get(v)
            if not rules:
                Iplus[v] = -sys.maxsize
                Imax[v] = -sys.maxsize
            else:
                s = 0
                mx = -sys.maxsize
                for rule in rules:
                    prod = fweight(rule)  # fweight(rule)
                    for symbol in rule.rhs:
                        prod += Iplus[symbol]
                    s = np.logaddexp(s, prod)
                    mx = max(prod, mx)
                Iplus[v] = s
                Imax[v] = mx
    return Iplus, Imax


def outside_value(cfg: CFG, I: dict, fweight):
    std = toposort(cfg)
    O = {}
    for v in std:
        O[v] = 0
    O['S'] = 1  # Root node
    for v in reversed(cfg):
        rules = cfg.get(v)
        for e in rules:
            for u in e.rhs:
                k = fweight(e) + O[v]
                for s in e.rhs:
                    if s is not u:
                        k += I[s]
                O[u] = np.logaddexp(O[u], k)
    return O


def expected_features(forest: CFG, edge_features: dict, wmap: dict) -> dict:
    weight_f = get_weight_f(edge_features, wmap)
    Iplus, Imax = inside_value(forest, weight_f)
    outside = outside_value(forest, Iplus, weight_f)
    expf = defaultdict(float)
    for rule in forest:
        k = outside[rule.lhs]
        for v in rule.rhs:
            k *= Iplus[v]
        for f, v in edge_features[rule].items():
            expf[f] += k * v
    return expf, Imax


def viterbi(Imax, dxn, weight):
    std = toposort(dxn)
    u = std[-1]
    def iternew(u):
        queue = [u]
        while queue:
            u = queue.pop()
            mx1 = -sys.maxsize
            argmax1 = None
            for r in dxn[u]:
                mx2 = -sys.maxsize
                for v in r.rhs:
                    if v.is_terminal():
                        if 1 > mx2:
                            mx2 = 1
                    elif Imax[v] > mx2:
                        mx2 = Imax[v]
                mx2 += weight[r]
                if mx2 > mx1:
                    argmax1 = r
                    mx1 = mx2
            yield argmax1
            for v in reversed(argmax1.rhs):  # Ensure leftmost derivation
                if not v.is_terminal():
                    queue.append(v)
    cfg = CFG(iternew(u))
    return language_of_cfg(cfg, u)


def gradient(dxn: CFG, dxy: CFG, src_fsa: FSA, weight: dict, weights_ibm: dict, skip_dict) -> dict:
    fmapxn = featurize_edges(dxn, src_fsa, weights_ibm, skip_dict)

    expfxn, Imax = expected_features(dxn, fmapxn, weight)
    print(viterbi(Imax, dxn, weight))

    if len(dxy) == 0:
        print('Skipping ungenerated y')
        return
    fmapxy = featurize_edges(dxy, src_fsa, weights_ibm, skip_dict)
    expfxy, _ = expected_features(dxy, fmapxy, weight)

    gradient = defaultdict(float)
    features = set(expfxn.keys())
    features.union(expfxy.keys())
    for f in features:
        gradient[f] = expfxy[f] - expfxn[f]
    return gradient
