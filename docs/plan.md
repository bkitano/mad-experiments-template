1. Get Dataset generation working
- Goal: should generate sequences with BOS/EOS/PAD tokens.

2. Plug the dataset into a FSA to show 100% accuracy
- Goal: should show 100% ID and OOD accuracy.

3. Implement DeltaNet from `fla` with `accelerate` on Modal + wandb
- Goal: should show high ID and OOD accuracy.
- Questions to investigate
  - what is the nature of the failures for OOD? Mechanistically, what are they getting wrong?
    - try grouping and categorizing the sequences it gets wrong.
  - Can we extract the FSA and compare it to the ground truth?
    - extracting the FSA means
      - identifying the monoid rules from the matrix updates...? I'm still not sure how we incorporate the additive term
  - Can I predict from theory how it will fail?
    - something along the lines of "theory says there is differential performance along the boundary of inputs based on training duration"; ie some cases may be harder than others? What would some cases look like: longer sequences, sequences that have fewer corresponding training set subsequences.