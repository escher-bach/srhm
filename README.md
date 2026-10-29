# Self-Similar Hierarchy Models

Two companion notes on hierarchical generative models whose rules are shared
across scale, and on what that sharing makes learnable without labels and
without backpropagation. Both papers and every experiment behind their
quantitative claims are in this repository.

## Papers

**[Self-Similar Recursive Grids](papers/srg.pdf)** ([source](papers/srg.tex))
formalizes the recursive grid: a grid of `n` cells with configuration `C` and
outside state `f(C)`, where grids are themselves cells of larger grids and the
same cell map `f` is used at every position and every scale. Structurally this
is a recurrent network unrolled in scale rather than time, which gives `n^L`
root-to-leaf trajectories and `(n^L - 1)/(n - 1)` copies of one local problem
per input. The note proves that the invariance group of an exchangeable SRG is
the iterated wreath product `Aut(T_L)`; that every invariant covariance has
exactly `L+1` eigenvalues, one per scale, so linear learning dynamics decouple
by scale; that a discrete SRG is a tree automaton whose minimal state space is
given by a Myhill-Nerode theorem, with scale tying as the precise condition
under which cross-scale copying is valid; and that for self-similar generative
processes Bayes-optimal inference is itself an SRG whose kernel is identifiable
from the law of a single bottom grid. Three local learning rules follow with
guarantees: frozen-input scale-equilibrium learning, scale-Bellman
bootstrapping with error linear in depth, and node-parallel evolution
strategies. Headline demonstration: a single unlabelled tree with 19,683
leaves yields a root classifier within 0.2 points of the Bayes oracle, with
zero labels and no backpropagation.

**[Learning a Self-Similar Grammar from One Example](papers/srhm.pdf)**
([source](papers/srhm.tex)) is the stochastic-process counterpart, answering
Open Problem 9.2 of the above. The Random Hierarchy Model is a synthetic
compositional task: a depth-`L` context-free grammar with `v` symbols, `m`
synonymous rules per symbol and branching `s`. Published results draw the
rules independently at every level; this note studies the *tied* model, in
which one grammar is used at every level. In the untied model every learner is
capped at root accuracy `1/v + P/(vm)` from `P` labelled examples, even with
unlimited unlabelled data and an oracle for all lower-level rules, because the
top rule has no context. With tying, clustering bottom tuples by their sibling
context recovers the grammar, which is a statistical form of the
Myhill-Nerode quotient, and the consistent namings of those clusters form
exactly a coset of the automorphism group of the valid-tuple set, trivial with
high probability. A single unlabelled example of sufficient depth therefore
determines the grammar exactly. The note also establishes graceful degradation
under partial tying, a depth-independent error floor and its matching
upper bound, the same pipeline in linear-Gaussian and bounded nonlinear
Gaussian trees, an identification theorem for arbitrary tied tree processes,
and a Kesten-Stigum explanation of why backpropagation's label requirement
grows with depth. On CIFAR-10 the tied tree is an exact density reaching
4.75 bits/dim with zero labels.

## Experiments

[`experiments/`](experiments) holds the model implementations, the experiment
drivers and the recorded results each claim in the papers cites.

Models: `srhm.py` (tied and untied grammars), `ptrhm.py` (partial tying),
`lgtree.py` (linear-Gaussian trees), `ngt.py` and `ngt_nd.py` (bounded
nonlinear Gaussian trees), `imtree.py` and `scenetree.py` (image trees),
`codebook*.py` (state codebooks), `selfsim.py`, `treelearn.py`.

Drivers are `exp_*.py`, `code_*.py` and `check_*.py`; `demo_one_tree.py` and
`demo_sshp.py` reproduce the headline single-example demonstrations, and
`checks.py` verifies the SRG algebra. Recorded outputs sit beside them as
`*.jsonl`, `*.log` and `*.json`; `test_*.py` are unit tests. Experiments
marked `.colab.log` were run on a hosted GPU; the rest are CPU results.

Most drivers need NumPy and SciPy; the CIFAR, codebook and backpropagation
comparisons additionally need PyTorch.
