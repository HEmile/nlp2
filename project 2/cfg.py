from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
class Symbol:
    pass

class Terminal(Symbol):
    pass

class Nonterminal(Symbol):
    pass

class Symbol:
    """
    A symbol in a grammar. In this class we basically wrap a certain type of object and treat it as a symbol.
    """

    def __init__(self):
        pass

    def is_terminal(self) -> bool:
        """Whether or not this is a terminal symbol"""
        pass

    def root(self) -> Symbol:
        """Some symbols are represented as a hierarchy of symbols, this method returns the root of that hierarchy."""
        pass

    def obj(self) -> object:
        """Returns the underlying python object."""
        pass

    def translate(self, target) -> Symbol:
        """Translate the underlying python object of the root symbol and return a new Symbol"""
        pass

class Terminal(Symbol):
    """
    Terminal symbols are words in a vocabulary.
    """

    def __init__(self, symbol: str):
        assert type(symbol) is str, 'A Terminal takes a python string, got %s' % type(symbol)
        self._symbol = symbol

    def is_terminal(self):
        return True

    def root(self) -> Terminal:
        # Terminals are not hierarchical symbols
        return self

    def obj(self) -> str:
        """The underlying python string"""
        return self._symbol

    def translate(self, target) -> Terminal:
        return Terminal(target)

    def __str__(self):
        return "'%s'" % self._symbol

    def __repr__(self):
        return 'Terminal(%r)' % self._symbol

    def __hash__(self):
        return hash(self._symbol)

    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol

    def __ne__(self, other):
        return not (self == other)

class Nonterminal(Symbol):
    """
    Nonterminal symbols are variables in a grammar.
    """

    def __init__(self, symbol: str):
        assert type(symbol) is str, 'A Nonterminal takes a python string, got %s' % type(symbol)
        self._symbol = symbol

    def is_terminal(self):
        return False

    def root(self) -> Nonterminal:
        # Nonterminals are not hierarchical symbols
        return self

    def obj(self) -> str:
        """The underlying python string"""
        return self._symbol

    def translate(self, target) -> Nonterminal:
        return Nonterminal(target)

    def __str__(self):
        return "[%s]" % self._symbol

    def __repr__(self):
        return 'Nonterminal(%r)' % self._symbol

    def __hash__(self):
        return hash(self._symbol)

    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol

    def __ne__(self, other):
        return not (self == other)

class Span(Symbol):
    pass

class Span(Symbol):
    """
    A span can be a terminal, a nonterminal, or a span wrapped around two integers.
    Internally, we represent spans with tuples of the kind (symbol, start, end).

    Example:
        Span(Terminal('the'), 0, 1)
        Span(Nonterminal('[X]'), 0, 1)
        Span(Span(Terminal('the'), 0, 1), 1, 2)
        Span(Span(Nonterminal('[X]'), 0, 1), 1, 2)
    """

    def __init__(self, symbol: Symbol, start: int, end: int):
        assert isinstance(symbol, Symbol), 'A span takes an instance of Symbol, got %s' % type(symbol)
        self._symbol = symbol
        self._start = start
        self._end = end

    def is_terminal(self):
        # a span delegates this to an underlying symbol
        return self._symbol.is_terminal()

    def root(self) -> Symbol:
        # Spans are hierarchical symbols, thus we delegate
        return self._symbol.root()

    def obj(self) -> (Symbol, int, int):
        """The underlying python tuple (Symbol, start, end)"""
        return (self._symbol, self._start, self._end)

    def translate(self, target) -> Span:
        return Span(self._symbol.translate(target), self._start, self._end)

    def __str__(self):
        return "%s:%s-%s" % (self._symbol, self._start, self._end)

    def __repr__(self):
        return 'Span(%r, %r, %r)' % (self._symbol, self._start, self._end)

    def __hash__(self):
        return hash((self._symbol, self._start, self._end))

    def __eq__(self, other):
        return type(self) == type(other) and self._symbol == other._symbol and self._start == other._start and self._end == other._end

    def __ne__(self, other):
        return not (self == other)

from collections import defaultdict

class Rule(object):
    """
    A rule is a container for a LHS symbol and a sequence of RHS symbols.
    """

    def __init__(self, lhs: Symbol, rhs: list):
        """
        A rule takes a LHS symbol and a list/tuple of RHS symbols
        """
        assert isinstance(lhs, Symbol), 'LHS must be an instance of Symbol'
        assert len(rhs) > 0, 'If you want an empty RHS, use an epsilon Terminal'
        assert all(isinstance(s, Symbol) for s in rhs), 'RHS must be a sequence of Symbol objects'
        self._lhs = lhs
        self._rhs = tuple(rhs)

    def __eq__(self, other):
        return type(self) == type(other) and self._lhs == other._lhs and self._rhs == other._rhs

    def __ne__(self, other):
        return not (self == other)

    def __hash__(self):
        return hash((self._lhs, self._rhs))

    def __str__(self):
        return '%s ||| %s' % (self._lhs, ' '.join(str(s) for s in self._rhs))

    def __repr__(self):
        return 'Rule(%r, %r)' % (self._lhs, self._rhs)

    @property
    def lhs(self):
        return self._lhs

    @property
    def rhs(self):
        return self._rhs

    @property
    def arity(self):
        return len(self._rhs)

class CFG:
    """
    A CFG is nothing but a container for rules.
    We group rules by LHS symbol and keep a set of terminals and nonterminals.
    """

    def __init__(self, rules=[]):
        self._rules = []
        self._rules_by_lhs = defaultdict(list)
        self._terminals = set()
        self._nonterminals = set()
        # organises rules
        for rule in rules:
            self._rules.append(rule)
            self._rules_by_lhs[rule.lhs].append(rule)
            self._nonterminals.add(rule.lhs)
            for s in rule.rhs:
                if s.is_terminal():
                    self._terminals.add(s)
                else:
                    self._nonterminals.add(s)

    @property
    def nonterminals(self):
        return self._nonterminals

    @property
    def terminals(self):
        return self._terminals

    def __len__(self):
        return len(self._rules)

    def __getitem__(self, lhs):
        return self._rules_by_lhs.get(lhs, frozenset())

    def get(self, lhs, default=frozenset()):
        """rules whose LHS is the given symbol"""
        return self._rules_by_lhs.get(lhs, default)

    def can_rewrite(self, lhs):
        """Whether a given nonterminal can be rewritten.

        This may differ from ``self.is_nonterminal(symbol)`` which returns whether a symbol belongs
        to the set of nonterminals of the grammar.
        """
        return len(self[lhs]) > 0

    def __iter__(self):
        """iterator over rules (in arbitrary order)"""
        return iter(self._rules)

    def items(self):
        """iterator over pairs of the kind (LHS, rules rewriting LHS)"""
        return self._rules_by_lhs.items()

    def __str__(self):
        lines = []
        for lhs, rules in self.items():
            for rule in rules:
                lines.append(str(rule))
        return '\n'.join(lines)

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


def inside_value(cfg: CFG):
    sorted = toposort(cfg)
    I = {}
    for v in sorted:
        if v in cfg.terminals:
            I[v] = 1
        elif v in cfg.nonterminals:
            rules = cfg.get(v)
            if not rules:
                I[v] = 0
            else:
                s = 0
                for rule in rules:
                    prod = 1
                    for symbol in rule.rhs:
                        prod *= I[symbol]
                    s += prod
                I[v] = s
    return I[sorted[-1]]

def outside_value(cfg: CFG, I):
    sorted = toposort(cfg)
    O = {}
    for v in sorted:
        O[v] = 0
    O['S'] = 1 #Root node
    for v in reversed(cfg):
        rules = cfg.get(v)
        for e in rules:
            for u in e.rhs:
                k = weight[e]*O[v]
                for s in e.rhs: 
                    if s is not u:
                        k *= I[s]
                O[u] += k
    return O







