from libitg import *
from collections import defaultdict
from nltk.util import skipgrams


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
            src_word_ = src_fsa.labels(s1, s2)
            for c in src_word_:
                src_word = c
                break
            tgt_word = get_terminal_string(symbol)
            if symbol.root() == eps:  # symbol.root() gives us a Terminal free of annotation
                fmap['type:deletion'] += 1.0
                # dense versions (for initial development phase)
                # TODO: use IBM1 prob
                if (src_word, tgt_word) in weights_ibm.keys():
                    ibm_prob = weights_ibm[(src_word, tgt_word)]
                    fmap['ibm1:del:logprob'] += ibm_prob
                # sparse version
                if sparse_del:
                    fmap['del:%s' % src_word] += 1.0
            else:                
                if s1 == s2:  # has not consumed any source word, must be an eps rule
                    fmap['type:insertion'] += 1.0
                    # dense version
                    # TODO: use IBM1 prob
                    if (src_word, tgt_word) in weights_ibm.keys():
                        ibm_prob = weights_ibm[(src_word, tgt_word)]
                        map['ibm1:ins:logprob'] += ibm_prob                    
                    # sparse version
                    if sparse_ins:
                        fmap['ins:%s' % tgt_word] += 1.0
                else:
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
    skip_dict = dict()
    for sen in chinese:
        skips = list(skipgrams(sen.split(),2, 1))
        for skip in skips:
            skip_dict[skip] += 1
    return skip_dict

def featurize_edges(forest, src_fsa, weights_ibm,
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
    I = {}
    for v in std:
        if v in cfg.terminals:
            I[v] = 1
        elif v in cfg.nonterminals:
            rules = cfg.get(v)
            if not rules:
                I[v] = 0
            else:
                s = 0
                for rule in rules:
                    prod = fweight(rule)
                    for symbol in rule.rhs:
                        prod *= I[symbol]
                    s += prod
                I[v] = s
    return I


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
                k = fweight(e)*O[v]
                for s in e.rhs:
                    if s is not u:
                        k *= I[s]
                O[u] += k
    return O


def expected_features(forest: CFG, edge_features: dict, wmap: dict) -> dict:
    weight_f = get_weight_f(edge_features, wmap)
    inside = inside_value(forest, weight_f)
    outside = outside_value(forest, inside, weight_f)
    expf = defaultdict(float)
    for rule in forest:
        k = outside[rule.lhs]
        for v in rule.rhs:
            k *= inside[v]
        for f, v in edge_features[rule].items():
            expf[f] += k * v
    return expf


def gradient(dxn: CFG, dxy: CFG, src_fsa: FSA, weight: dict, weights_ibm: dict) -> dict:
    fmapxn = featurize_edges(dxn, src_fsa, weights_ibm)
    fmapxy = featurize_edges(dxy, src_fsa, weights_ibm)

    expfxn = expected_features(dxn, fmapxn, weight)
    expfxy = expected_features(dxy, fmapxy, weight)

    gradient = defaultdict(float)
    features = set(expfxn.keys())
    features.union(expfxy.keys())
    for f in features:
        gradient[f] = expfxy[f] - expfxn[f]
    return gradient


# def viterbi(dxn: CFG, src_fsa: FSA, weight: dict) -> CFG:
#     fmapxn = featurize_edges(dxn, src_fsa)
