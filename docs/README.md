1) Goal and claim we’re testing

We want to distinguish:
	•	Memorization / shortcutting: high IID accuracy at fixed length k, but brittle under invariances, counterfactuals, and length shifts.
	•	NC¹-style algorithmic computation: depth is used to implement iterated composition (balanced-tree style), with strong invariances and systematic length generalization.

We will not claim “this model is in NC¹” from accuracy alone. We will claim:
	•	“behavior is consistent with algorithmic composition” vs
	•	“behavior is consistent with memorization / distributional shortcuts.”

⸻

2) Task definition

2.1 Problem: group word evaluation

Given a finite group G (primary: S_5, also include abelian controls), and a sequence (word)
(g_1, g_2, \dots, g_k) \in G^k
predict the product:
y = g_1 g_2 \cdots g_k \in G

This is the canonical NC¹-hard benchmark when G is non-solvable (e.g. S_5).

⸻

3) Input/output format and tokenization (BOS/EOS etc.)

3.1 Vocabulary
	•	One token per group element: GE_0 ... GE_{|G|-1}.
	•	Special tokens:
	•	BOS (begin-of-sequence)
	•	EOS (end-of-sequence)
	•	PAD (if batching)
	•	(optional) SEP if you ever join multiple words, but not needed here.

3.2 Sequence encoding

For a word length k, input tokens are:

BOS GE_{i1} GE_{i2} ... GE_{ik} EOS

Output is a single class: the index of the product element GE_{iy}.

Notes:
	•	Use a standard classifier head on the final hidden state for EOS (or pooled representation). Don’t overcomplicate: keep evaluation comparable across architectures.
	•	Keep positional encoding consistent per model. For “TC⁰-like transformer,” you will likely use standard learned or sinusoidal positions, but see invariance checks below.

⸻

4) Positive vs negative checks (succinct list)

4.1 Negative checks (memorization / lack of generalization)

These are red flags. One or two might be ambiguous; several together is strong evidence.
	1.	Length extrapolation cliff

	•	Train on lengths ≤ k, test on k+1, 2k.
	•	Signal: sharp accuracy drop, often with high confidence.

	2.	Conjugation invariance failure

	•	For random h \in G, map each token g_i \mapsto h g_i h^{-1}. True label transforms to h y h^{-1}.
	•	Signal: accuracy/confidence drops or outputs don’t transform correctly.

	3.	Generator / element relabeling sensitivity

	•	Apply a random bijection (permutation) of element IDs consistently to inputs AND labels.
	•	Signal: model breaks (indicates dependence on element IDs, not algebra).

	4.	Counterfactual swaps / perturbations

	•	Swap two distant blocks with matched local statistics (same multiset of elements) but different order.
	•	Flip one token at position j.
	•	Signal: output changes incorrectly or not enough; sensitivity varies strongly with position (e.g., early tokens barely matter).

	5.	Hard-negative matched-statistics failures

	•	Construct pairs of words with identical element counts / bigram stats but different products.
	•	Signal: model predicts same class or uses frequency heuristics.

	6.	Nearest-neighbor reliance

	•	In embedding space, test accuracy correlates strongly with distance to closest training example.
	•	Signal: exemplar retrieval behavior.

	7.	Membership inference succeeds

	•	Attack predicts whether a sample was in training using confidence/loss.
	•	Signal: significantly above chance.

	8.	Overconfidence OOD

	•	Under any of the above shifts, wrong predictions remain high-confidence.

4.2 Positive checks (NC¹-style composition)

These are “green flags” indicating actual compositional computation.
	1.	Smooth length generalization

	•	Accuracy degrades gradually with k, not abruptly. Calibration remains reasonable.

	2.	Conjugation equivariance / invariance holds

	•	Under g_i \mapsto h g_i h^{-1}, prediction transforms as y \mapsto h y h^{-1} with no accuracy drop.

	3.	Relabeling robustness

	•	Consistent remapping of element IDs leaves performance essentially unchanged after minimal adjustment (in strict form, you can train with random relabelings to encourage this).

	4.	Uniform positional sensitivity

	•	Single-token perturbations have comparable influence regardless of position; early tokens matter.

	5.	Layerwise partial-product signal

	•	Probes can decode partial products of increasingly long spans from deeper layers (span growth with depth is the key).

	6.	Distribution shift robustness

	•	Train/test across different word distributions (uniform, cancellation-heavy, subgroup-restricted) with stable accuracy.

	7.	No nearest-neighbor dependence

	•	Accuracy does not collapse in low-density regions of input space; weak correlation with training similarity.

⸻

5) Dataset generation

You already have generation tooling in word-problem (see README). Here’s the required structure to support the checks.

5.1 Groups

Include:
	•	Primary hard group: S_5 (non-solvable; NC¹-complete word problem).
	•	Controls:
	•	Abelian: Z_n (easy; helps debug).
	•	Solvable non-abelian: e.g. A_4 \times Z_5 (intermediate behavior).

5.2 Train/test splits (must be sequence-disjoint)

For each k, split by sequence identity: no exact word should appear in both train and test.

Two regimes:
	•	IID regime (for sanity, not decisive)
	•	Algorithmic regime (the decisive one)

Algorithmic regime recommended split

Train on:
	•	all lengths 2..k OR only \{2,k\} if using --strict_len to make extrapolation sharper.

Test sets:
	•	Length OOD: k+1, 2k (newly generated).
	•	Conjugation OOD: conjugate every sequence by random h.
	•	Relabeling OOD: apply fresh random element-ID permutations to every example.
	•	Matched-stat hard negatives: specially constructed (see below).
	•	Distribution shifts: cancellation-heavy, subgroup-restricted.

5.3 How to sample words

Base generator: uniform random words in G^k, without replacement when possible.

Additional distributions (needed for shift tests)
	1.	Cancellation-heavy
Generate words with deliberate local inverses:

	•	sample random elements a_1,...,a_m
	•	interleave with inverses: a_1, a_1^{-1}, a_2, a_2^{-1}, ...
	•	then add random “noise” elements so not trivially reducible

	2.	Subgroup-restricted
Pick a subgroup H \le G (for S_5, pick a known subgroup like a dihedral subgroup; your algebra library likely has subgroup generation).
Generate words entirely in H. This tests whether the model relies on global distribution properties.
	3.	Matched-stat pairs
Goal: two sequences with same easy statistics but different products.
Practical method:

	•	sample a base word w
	•	create w' by permuting tokens inside blocks (same multiset, different order)
	•	reject if product accidentally matches
	•	optionally constrain to preserve bigram histogram approximately (harder; good but not required initially)

5.4 Conjugation and relabeling transforms

These must be implemented as post-processing on the dataset so labels remain correct.
	•	Conjugation by h:
	•	input: [g_1,\dots,g_k]
	•	output: [h g_1 h^{-1}, \dots, h g_k h^{-1}]
	•	label transforms: y \mapsto h y h^{-1}
	•	Relabeling:
	•	sample a random bijection \pi: G \to G
	•	map every token GE_i to GE_{π(i)} consistently
	•	map label similarly

Important: do these relabelings per-example or per-batch for robust invariance stress; do them per-dataset for a clean train/test mismatch.

⸻

6) What to log (minimum)

Per example:
	•	group, length k
	•	input word IDs
	•	label ID
	•	distribution tag (uniform/cancellation/subgroup/matched-stat/etc.)
	•	for OOD variants: the transformation metadata (conjugating element h, relabeling permutation id, etc.)
	•	model logits, predicted class, confidence (max softmax)
	•	loss

Aggregate:
	•	accuracy by length
	•	accuracy by distribution tag
	•	accuracy under each invariance transform
	•	calibration (ECE or reliability plots)
	•	for perturbation tests: accuracy vs perturbed position

⸻

7) Probes / internal diagnostics (required for “positive” case)

You need at least one internal test that shows iterative composition rather than dataset fitting.

7.1 Partial-product probes (recommended)

For each layer ℓ, train a linear probe to predict:
	•	product of prefix length m, for multiple m
	•	or product of a window / block

Key expectation for NC¹-like behavior:
	•	as layer ℓ increases, the probe can decode products of longer spans (often roughly doubling span per layer in well-structured models).

Implementation sketch:
	•	Freeze model.
	•	For each training sample, compute hidden states at each layer.
	•	Define targets:
	•	p_m = g_1 \cdots g_m for m \in \{2,4,8,\dots,k\}
	•	Train linear classifiers for each (layer, m) pair.
	•	Plot probe accuracy as heatmap: layers vs span size.

Shortcut models: probe success won’t show monotone span growth.

7.2 Nearest-neighbor dependence
	•	Extract penultimate embeddings.
	•	For each test example compute distance to nearest training embedding.
	•	Plot accuracy vs distance quantile.

⸻

8) Required sweeps (so conclusions are not fragile)

This is what the intern should run as a baseline.

8.1 Architecture sweeps

A) TC⁰-like transformer sweep

Hold width roughly fixed, vary depth:
	•	num_layers ∈ {1, 2, 4, 6, 8} (choose max that still “feels constant” for your narrative)
	•	fixed hidden dim (e.g., 256/512), fixed heads
	•	no recurrence, no external memory

Also sweep:
	•	positional encoding variant (learned vs sinusoidal) because it changes shortcut space
	•	attention pattern (full vs local) if available

B) NC¹-capable model sweep (DeltaNet)

Vary the parameter that corresponds to iterative capacity:
	•	depth / number of compositional steps
	•	any gating / update rank parameter
	•	keep token embedding dim comparable to transformer

8.2 Data sweeps

For each group and base length k:
	•	k ∈ {4, 6, 8, 10, 12} (pick based on compute; the key is at least 2–3 different k)
	•	training samples per k (e.g., 100k–1M depending on |G|^k feasibility)
	•	train on:
	•	uniform only
	•	uniform + cancellation
	•	uniform + cancellation + subgroup
	•	test on:
	•	IID held-out
	•	length OOD: k+1, 2k
	•	conjugation OOD
	•	relabeling OOD
	•	matched-stat counterfactuals

8.3 Seed sweeps

At least 3 seeds per config (5 ideal). Memorization effects can look “clean” on one seed and vanish on another.

⸻

9) Expected outcomes (what “success” looks like)

TC⁰-like transformer (negative profile)
	•	high IID accuracy at fixed k (often quite high)
	•	sharp failure on:
	•	length OOD
	•	conjugation/relabeling OOD
	•	matched-stat counterfactuals
	•	overconfident wrong predictions OOD
	•	strong nearest-neighbor dependence
	•	partial-product probes do not show span growth by layer

NC¹-capable model (positive profile)
	•	better length extrapolation
	•	robust conjugation/relabeling behavior (or at least far less brittle)
	•	counterfactual sensitivity uniform across positions
	•	partial-product probe shows increasing span with depth
	•	reduced nearest-neighbor dependence

⸻

10) Practical notes for the intern (common pitfalls)
	1.	Do not trust IID accuracy. The entire point is that it can be misleading.
	2.	Disjointness matters. Ensure sequences are not duplicated across splits.
	3.	Transformations must preserve labels (especially conjugation).
	4.	Control tasks (e.g., Z_n) should be solved by everyone; if not, fix pipeline before interpreting anything.
	5.	Report by slice: length, distribution tag, and invariance condition.

⸻

11) Minimal “first milestone” plan (1 week)

If bandwidth is tight, do this first:
	1.	Group = S_5, lengths k=6 train, test on k=6 IID and k=7 OOD
	2.	Train:
	•	transformer depth 2 vs depth 6
	•	DeltaNet baseline
	3.	Evaluate:
	•	IID accuracy
	•	length OOD accuracy
	•	conjugation OOD accuracy
	•	single-token flip sensitivity vs position
	4.	Do partial-product probes for m \in \{2,4,6\}

That alone will typically separate memorization vs composition.
