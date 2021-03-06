from libitg import *
from collections import defaultdict
from nltk.util import skipgrams
import numpy as np
import sys
from alg import *
import pickle
import random
import math
import scipy.stats


def get_terminal_string(symbol: Symbol):
    """Returns the python string underlying a certain terminal (thus unwrapping all span annotations)"""
    if not symbol.is_terminal():
        raise ValueError('I need a terminal, got %s of type %s' % (symbol, type(symbol)))
    return symbol.root().obj()

def get_spans(symbol: Span):
    if not isinstance(symbol, Span):
        raise ValueError('I need a span, got %s of type %s' % (symbol, type(symbol)))
    sym, start2, end2 = symbol.obj()  # this unwraps the target or length annotation
    return sym, start2, end2

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
    sym, start1, end1 = s.obj()  # this unwraps the source annotation
    return (sym, start1, end1), (start2, end2)

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


def simple_features(edge: Rule, src_fsa: FSA, weights_ibm, skip_dict, use_bispans=False, eps=Terminal('-EPS-'),
                    sparse_del=False, sparse_ins=False, sparse_trans=False, skip_grams=False) -> dict:
    """
    Featurises an edge given
        * rule and spans
        * src sentence as an FSA
        * use_bispans is used when the target y is available
        * TODO: target sentence length n
        * TODO: extract IBM1 dense features
    crucially, note that the target sentence y is not available!    
    """
    fmap = defaultdict(float)
    if len(edge.rhs) == 2:  # binary rule
        fmap['type:binary'] += 1.0
        # here we could have sparse features of the source string as a function of spans being concatenated
        if use_bispans:
            (l_sym, ls1, ls2), (lt1, lt2) = get_bispans(edge.rhs[0])  # left of RHS
            (r_sym, rs1, rs2), (rt1, rt2) = get_bispans(edge.rhs[1])  # right of RHS
        else:
            l_sym, ls1, ls2 = get_spans(edge.rhs[0])  # left of RHS
            r_sym, rs1, rs2 = get_spans(edge.rhs[1])  # right of RHS

        fmap['type:span_source_lhs'] += (ls2-ls1)
        fmap['type:span_source_rhs'] += (rs2-rs1)

        if ls1 == ls2:  # deletion of source left child
           fmap['type:del_lhs'] += 1.0
           # fmap['type:source_length'] += 1.0
        if rs1 == rs2:  # deletion of source right child
           fmap['type:del_rhs'] += 1.0
           # fmap['type:source_length'] += 1.0
        if ls2 == rs1:  # monotone
            fmap['type:mon'] += 1.0
        if ls1 == rs2:  # inverted
            fmap['type:inv'] += 1.0

        if skip_grams:
            if ls2-ls1 == 1 and rs2-rs1 == 1:
                w1, w2 = get_source_word(src_fsa, ls1, ls2), get_source_word(src_fsa, rs1, rs2)
                fmap['skip:%s/%s' % (w1, w2)] += 1.0


        if l_sym == Nonterminal('I') or r_sym == Nonterminal('I'):
            fmap['type:insertion'] += 1.0
            # fmap['type:target_length'] += 1.0
        if l_sym == Nonterminal('T') or r_sym == Nonterminal('T'):
            fmap['type:translation'] += 1.0
            # fmap['type:target_length'] += 1.0
            # fmap['type:source_length'] += 1.0
        if l_sym == Nonterminal('D') or r_sym == Nonterminal('D'):
            fmap['type:deletion'] += 1.0
            # fmap['type:source_length'] += 1.0
                
    else:  # unary
        symbol = edge.rhs[0]
        if symbol.is_terminal():  # terminal rule
            fmap['type:terminal'] += 1.0
            # we could have IBM1 log probs for the translation pair or ins/del
            if use_bispans:
                (sym, s1, s2), (t1, t2) = get_bispans(symbol)
            else:
                sym, s1, s2 = get_spans(symbol)
            if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
                src_word = get_source_word(src_fsa, s1, s2)
                fmap['type:deletion'] += 1.0
                # dense versions (for initial development phase)
                # TODO: use IBM1 prob
                fmap['ibm1:del:logprob'] += weights_ibm[(src_word, eps)]
                # sparse version
                if sparse_del:
                    fmap['del:%s' % src_word] += 1.0
            else:
                tgt_word = get_target_word(symbol)
                if s1 == s2:  # has not consumed any source word, must be an eps rule
                    fmap['type:insertion'] += 1.0
                    
                    # dense version
                    # TODO: use IBM1 prob
                    fmap['ibm1:ins:logprob'] += weights_ibm[(eps, tgt_word)]
                    # sparse version
                    if sparse_ins:
                        fmap['ins:%s' % tgt_word] += 1.0
                else:
                    src_word = get_source_word(src_fsa, s1, s2)
                    fmap['type:translation'] += 1.0
                    # dense version
                    # TODO: use IBM1 prob
                    fmap['ibm1:x2y:logprob'] += weights_ibm[(src_word, tgt_word)]
                    #ff['ibm1:y2x:logprob'] += 
                    # sparse version                    
                    if sparse_trans:
                        fmap['trans:%s/%s' % (src_word, tgt_word)] += 1.0

                    if skip_grams:
                        #skip bigrams:
                        try:
                            next_word = get_source_word(src_fsa, s1+1, s2+1)
                            fmap['skip:%s%s' % (src_word, next_word)] = skip_dict[(src_word, next_word)]
                        except AssertionError:
                            pass
                            
                        try:
                            second_word = get_source_word(src_fsa, s1+2, s2+2)
                            fmap['skip:%s%s' % (src_word, second_word)] = skip_dict[(src_word, second_word)]
                        except AssertionError:
                            pass

        elif symbol.root().obj()[0] == 'D' and len(symbol.root().obj()) == 1:
            fmap['type:deletion'] += 1.0
            # fmap['type:source_length'] += 1.0
        elif symbol.root().obj()[0] == 'T':
            fmap['type:translation'] += 1.0
            # fmap['type:source_length'] += 1.0
            # fmap['type:target_length'] += 1.0
        else:  # S -> X
            if edge.lhs == Nonterminal('D(x)') or Nonterminal('D(x, y)'):
                # here lhs is the root of the intersected forest: S' 
                # do not weight this edge
                # note that Earley introduces S' -> S
                # thus this edge is not a real clique in the CRF, 
                # it's just a convenience that Earley adds to the forest
                # in order to guarantee its root is unique
                pass  
            else:
                pass
    return fmap


def skip_bigrams(chinese) -> dict:
    skip_dict = defaultdict(int)
    for sen in chinese:
        skips = list(skipgrams(sen.split(),2, 1))
        for skip in skips:
            skip_dict[skip] += 1
    return skip_dict

def featurize_edges(forest, src_fsa, weights_ibm, skip_dict, use_bispans=False,
                    sparse_del=False, sparse_ins=False, sparse_trans=False,
                    use_skip_dict = True, eps=Terminal('-EPS-')) -> dict:
    edge2fmap = dict()
    for edge in forest:
        edge2fmap[edge] = simple_features(edge, src_fsa, weights_ibm, skip_dict, use_bispans, eps, sparse_del, sparse_ins, sparse_trans, use_skip_dict)
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
    Iplus = {} #defaultdict(float)
    Imax = {}
    for v in std:
        rules = cfg.get(v)
        if not rules and v.is_terminal():
            Iplus[v] = 0
            Imax[v] = 0
        else:
            s = -np.inf
            mx = -np.inf
            for rule in rules:
                prod = fweight(rule)  # fweight(rule)
                for symbol in rule.rhs:
                    prod += Iplus[symbol]
                # s += math.exp(prod)
                s = np.logaddexp(s, prod)
                mx = max(prod, mx)
            Iplus[v] = s
            Imax[v] = mx
    return Iplus, Imax, Iplus[std[-1]]

def derivations(cfg: CFG):
    std = toposort(cfg)
    Iplus = defaultdict(float)
    for v in std:
        rules = cfg.get(v)
        if not rules:
            Iplus[v] = 1
        else:
            s = 0
            for rule in rules:
                prod = 1  # fweight(rule)
                for symbol in rule.rhs:
                    prod *= Iplus[symbol]
                s += prod
            Iplus[v] = s
    return Iplus[std[-1]]

def inside_value_antilog(cfg: CFG, fweight):
    std = toposort(cfg)
    Iplus = defaultdict(float)
    for v in std:
        rules = cfg.get(v)
        if not rules:
            Iplus[v] = 1
        else:
            s = 0
            for rule in rules:
                prod = math.exp(fweight(rule))  # fweight(rule)
                for symbol in rule.rhs:
                    prod *= Iplus[symbol]
                s += prod
            Iplus[v] = s
    return Iplus, Iplus[std[-1]]


def outside_value(cfg: CFG, I: dict, fweight):
    std = toposort(cfg)
    O = {}
    for v in std:
        O[v] = -np.inf
    O[std[-1]] = 0  # Root node
    for v in reversed(std):
        for e in cfg.get(v):
            for u in e.rhs:
                k = fweight(e) + O[v]
                for s in e.rhs:
                    if s != u:
                        k += I[s]
                O[u] = np.logaddexp(O[u], k)
    return O


def outside_value_antilog(cfg: CFG, I: dict, fweight):
    std = toposort(cfg)
    O = {}
    for v in std:
        O[v] = 0
    O[std[-1]] = 1  # Root node
    for v in reversed(std):
        for e in cfg.get(v):
            for u in e.rhs:
                k = math.exp(fweight(e)) * O[v]
                for s in e.rhs:
                    if s != u:
                        k *= I[s]
                O[u] += k
    return O


def expected_features(forest: CFG, edge_features: dict, wmap: dict) -> dict:
    weight_f = get_weight_f(edge_features, wmap)
    Iplus, Imax, tot = inside_value(forest, weight_f)
    outside = outside_value(forest, Iplus, weight_f)
    expf = defaultdict(float)
    for rule in forest:
        k = outside[rule.lhs] + weight_f(rule)
        for v in rule.rhs:
            k += Iplus[v]
        k = math.exp(k - tot)
        for f, v in edge_features[rule].items():
            expf[f] += k * v
    return expf, tot


def expected_features_antilog(forest: CFG, edge_features: dict, wmap: dict) -> dict:
    weight_f = get_weight_f(edge_features, wmap)
    Iplus, tot = inside_value_antilog(forest, weight_f)
    outside = outside_value_antilog(forest, Iplus, weight_f)
    expf = defaultdict(float)
    for rule in forest:
        k = outside[rule.lhs]
        for v in rule.rhs:
            k *= Iplus[v]
        k = math.exp(math.log(k) - math.log(tot))
        for f, v in edge_features[rule].items():
            expf[f] += k * v
    # for f in expf.keys():
    #     expf[f] = math.log(expf[f])
    return expf, math.log(tot)

def viterbi(Imax, dxn, wmap, edge_features):
    fweight = get_weight_f(edge_features, wmap)
    std = toposort(dxn)
    u = std[-1]

    def iternew(u):
        queue = [u]
        while queue:
            u = queue.pop()
            mx1 = -sys.maxsize
            argmax1 = None
            for r in dxn[u]:
                mx2 = 0
                for v in r.rhs:
                    if not v.is_terminal():
                        mx2 += Imax[v]
                        #print(r)
                        #print(Imax[v])
                    #else:
                        #print(r)
                        #print(Imax[v])
                mx2 += fweight(r)
                if mx2 > mx1:
                    argmax1 = r
                    mx1 = mx2
            for v in reversed(argmax1.rhs):  # Ensure leftmost derivation
                if not v.is_terminal():
                    queue.append(v)
                else:
                    if v.obj()[2]-v.obj()[1] > 0:
                        yield v.obj()[0].obj()
    return ' '.join(filter(lambda x: x != '-EPS-', iternew(u)))

def sampling(Iplus, dxn, wmap, edge_features):
    fweight = get_weight_f(edge_features, wmap)
    std = toposort(dxn)
    u = std[-1]
    def iternew(u):
        queue = [u]
        while queue:
            u = queue.pop()
            probabilities = []
            argmax = None
            for r in dxn[u]:
                mx2 = 0
                for v in r.rhs:
                    if not v.is_terminal():
                        mx2 += Iplus[v]
                mx2 += fweight(r)
                probabilities.append((mx2 - Iplus[u], r))
            samp = math.log(random.random())
            s = -np.inf
            for log_prob, r in probabilities:
                s = np.logaddexp(s, log_prob)
                if samp <= s:
                    argmax = r
                    break
            for v in reversed(argmax.rhs):  # Ensure leftmost derivation
                if not v.is_terminal():
                    queue.append(v)
                else:
                    if v.obj()[2]-v.obj()[1] > 0:
                        yield v.obj()[0].obj()
    return ' '.join(filter(lambda x: x != '-EPS-', iternew(u)))


def gradient(dxn: CFG, dxy: CFG, src_fsa: FSA, weight: dict, weights_ibm: dict, skip_dict, index,
             get_features=False, sigma=0.0001, sparse=False, use_skipdict=False,
             fmapxn=None, fmapxy=None) -> dict:
    if not fmapxn:
        fmapxn = featurize_edges(dxn, src_fsa, weights_ibm, skip_dict, sparse_del=sparse, sparse_trans=sparse, sparse_ins=sparse, use_skip_dict=use_skipdict)
    if not fmapxy:
        fmapxy = featurize_edges(dxy, src_fsa, weights_ibm, skip_dict, sparse_del=sparse, sparse_trans=sparse, sparse_ins=sparse, use_skip_dict=use_skipdict, use_bispans=True)

    expfxn, totxn = expected_features(dxn, fmapxn, weight)

    expfxy, totxy = expected_features(dxy, fmapxy, weight)

    gd = defaultdict(float)
    features = set(expfxn.keys())
    features.union(expfxy.keys())

    for f in features:
        gd[f] = expfxy[f] - expfxn[f] - weight[f] / (sigma * sigma)

    sm = (1/(sigma * sigma)) * sum([-(x * x) for x in weight.values()])
    return gd, totxy - totxn + sm


def likelihood(dxn: CFG, dxy: CFG, src_fsa: FSA, weight: dict, weights_ibm: dict, skip_dict,
             sigma=0.0001, sparse=False, use_skipdict=False, fmapxn=None, fmapxy=None) -> dict:
    if not fmapxn:
        fmapxn = featurize_edges(dxn, src_fsa, weights_ibm, skip_dict, sparse_del=sparse, sparse_trans=sparse, sparse_ins=sparse, use_skip_dict=use_skipdict)
    if not fmapxy:
        fmapxy = featurize_edges(dxy, src_fsa, weights_ibm, skip_dict, sparse_del=sparse, sparse_trans=sparse, sparse_ins=sparse, use_skip_dict=use_skipdict, use_bispans=True)
    weight_f = get_weight_f(fmapxn, weight)
    Iplusxn, Imax, Irootxn = inside_value(dxn, weight_f)
    weight_f = get_weight_f(fmapxy, weight)
    Irootxy = 0
    if dxy:
        Iplusxy, _, Irootxy = inside_value(dxy, weight_f)

    regres = 0
    for k, v in weight.items():
        regres += v * v
    sm = (1/(sigma * sigma)) * sum([-(x * x) for x in weight.values()])
    return Irootxy - Irootxn

def predict(dx: CFG, fmapx, weight: dict, type='v') -> str:
    weight_f = get_weight_f(fmapx, weight)
    Iplus, Imax, _ = inside_value(dx, weight_f)
    if type == 'sampling':
        return sampling(Iplus, dx, weight, fmapx)
    else:
        return viterbi(Imax, dx, weight, fmapx)