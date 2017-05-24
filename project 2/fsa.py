from collections import defaultdict

class FSA:
    """
    A container for arcs. This implements a non-deterministic unweighted FSA.
    """

    class State:
        def __init__(self):
            self.by_destination = defaultdict(set)
            self.by_label = defaultdict(set)

        def __eq__(self, other):
            return type(self) == type(other) and self.by_destination == other.by_destination and self.by_label == other.by_label

        def __ne__(self, other):
            return not (self == other)

    def __init__(self):
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        # each state is a tuple (by_destination and by_label)
        #  by_destination is a dictionary that maps from destination to a set of labels
        #  by_label is a dictionary that maps from label to a set of destinations
        self._states = []
        self._initial = set()
        self._final = set()
        self._arcs = set()

    def __eq__(self, other):
        return type(self) == type(other) and self._states == other._states and self._initial == other._initial and self._final == other._final

    def __ne__(self, other):
        return not (self == other)

    def nb_states(self):
        """Number of states"""
        return len(self._states)

    def nb_arcs(self):
        """Number of arcs"""
        return len(self._arcs)

    def add_state(self, initial=False, final=False) -> int:
        """Add a state marking it as initial and/or final and return its 0-based id"""
        sid = len(self._states)
        self._states.append(FSA.State())
        #self._arcs.append(defaultdict(str))
        if initial:
            self.make_initial(sid)
        if final:
            self.make_final(sid)
        return sid

    def add_arc(self, origin, destination, label: str):
        """Add an arc between `origin` and `destination` with a certain label (states should be added before calling this method)"""
        self._states[origin].by_destination[destination].add(label)
        self._states[origin].by_label[label].add(destination)
        self._arcs.add((origin, destination, label))

    def destinations(self, origin: int, label: str) -> set:
        if origin >= len(self._states):
            return set()
        return self._states[origin].by_label.get(label, set())

    def labels(self, origin: int, destination: int) -> set:
        """Return the label of an arc or None if the arc does not exist"""
        if origin >= len(self._arcs):
            return set()
        return self._states[origin].by_destination.get(destination, set())

    def make_initial(self, state: int):
        """Mark a state as initial"""
        self._initial.add(state)

    def is_initial(self, state: int) -> bool:
        """Test whether a state is initial"""
        return state in self._initial

    def make_final(self, state: int):
        """Mark a state as final/accepting"""
        self._final.add(state)

    def is_final(self, state: int) -> bool:
        """Test whether a state is final/accepting"""
        return state in self._final

    def iterinitial(self):
        """Iterates over initial states"""
        return iter(self._initial)

    def iterfinal(self):
        """Iterates over final states"""
        return iter(self._final)

    def iterarcs(self, origin: int, group_by='destination') -> dict:
        if origin + 1 < self.nb_states():
            return self._states[origin].by_destination.items() if group_by == 'destination' else self._states[origin].by_label.items()
        return dict()

    def __str__(self):
        lines = ['states=%d' % self.nb_states(),
                 'initial=%s' % ' '.join(str(s) for s in self._initial),
                 'final=%s' % ' '.join(str(s) for s in self._final),
                 'arcs=%d' % self.nb_arcs()]
        for origin, state in enumerate(self._states):
            for destination, labels in sorted(state.by_destination.items(), key=lambda pair: pair[0]):
                for label in sorted(labels):
                    lines.append('origin=%d destination=%d label=%s' % (origin, destination, label))
        return '\n'.join(lines)


class LengthConstraint(FSA):
    """
    A container for arcs. This implements a deterministic unweighted FSA.
    """

    def __init__(self, n: int, strict=False):
        """
        :param n: length constraint
        :param strict: if True, accepts the language \Sigma^n, if False, accepts union of \Sigma^i for i from 0 to n
        """
        # each state is represented as a collection of outgoing arcs
        # which are organised in a dictionary mapping a label to a destination state
        super(LengthConstraint, self).__init__()
        self.add_state(initial=True, final=not strict)
        for i in range(n):
            self.add_state(final=not strict)
            self.add_arc(i, i + 1, '-WILDCARD-')
        # we always make the last state final
        self.make_final(n)

    def destination(self, origin: int, label: str) -> int:
        """Return the destination from a certain `origin` state with a certain `label` (-1 means no destination available)"""
        if origin + 1 < self.nb_states():
            outgoing = self._states[origin]
            if not outgoing:
                return -1
            return origin + 1
        else:
            return -1


def make_fsa(string: str) -> FSA:
    """Converts a sentence (string) to an FSA (labels are python str objects)"""
    fsa = FSA()
    fsa.add_state(initial=True)
    for i, word in enumerate(string.split()):
        fsa.add_state()  # create a destination state
        fsa.add_arc(i, i + 1, word)  # label the arc with the current word
    fsa.make_final(fsa.nb_states() - 1)
    return fsa

def make_val_fsa(strings: list) -> FSA:
    """Converts a sentence (string) to an FSA (labels are python str objects)"""
    fsa = FSA()
    j = 0
    for string in strings:
        fsa.add_state(initial=True)
        for word in string.split():
            fsa.add_state()  # create a destination state
            fsa.add_arc(j, j + 1, word)  # label the arc with the current word
            j += 1
        j += 1
        fsa.make_final(fsa.nb_states() - 1)
    return fsa