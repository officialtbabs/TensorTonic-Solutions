def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """

    BASE_VOCAB_SIZE = 256

    # Convert corpus to byte representation
    sequences = [list(item.encode("utf-8")) for item in corpus]

    # Initializing base byte vocabulary
    initial_vocab = {
        token_id: [token_id] for token_id in range(BASE_VOCAB_SIZE)
    }
    
    vocab = []
    merges = []

    new_token_id = BASE_VOCAB_SIZE

    for new_token_id in range(BASE_VOCAB_SIZE, vocab_size):
        pair_frequencies = {}

        # Counting adjacent pairs
        for sequence in sequences:
            # Computing pairs in a sequence
            pairs = zip(sequence[:-1], sequence[1:])
    
            # Updating pair frequencies
            for pair in pairs:
                pair_frequencies[pair] = pair_frequencies.get(pair, 0) + 1

        if not pair_frequencies:
            break
            
        # Getting most frequent pair 
        most_frequent_pair = max(pair_frequencies, key=lambda pair: (pair_frequencies[pair], pair_frequencies[pair],
                initial_vocab[pair[0]],
                initial_vocab[pair[1]]))

        left, right = most_frequent_pair

        # Update the initial vocabulary with the most frequent pair byte representation
        initial_vocab[new_token_id] = initial_vocab[left] + initial_vocab[right]
        
        vocab.append([new_token_id, initial_vocab.get(new_token_id)])
        
        merges.append([left, right, new_token_id])

        # Apply the merge of the most frequuent pair to the sequence
        for sequence_index, sequence in enumerate(sequences):
            merged_sequence =[]
            i = 0
            
            while i < len(sequence):
                if i < len(sequence) - 1 and sequence[i] == left and sequence[i + 1] == right:
                    merged_sequence.append(new_token_id)
                    i += 2
                else:
                    merged_sequence.append(sequence[i])
                    i += 1

            sequences[sequence_index] = merged_sequence
        
    return {
        "vocab": vocab,
        "merges": merges
    }
