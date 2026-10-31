def create_nsp_pairs(documents: list, pair_specs: list) -> list:
    pairs = []
    for spec in pair_specs:
        same_document = spec["doc_a"] == spec["doc_b"]
        consecutive = spec["sent_b"] == spec["sent_a"] + 1
        pairs.append({
            "sentence_a": documents[spec["doc_a"]][spec["sent_a"]],
            "sentence_b": documents[spec["doc_b"]][spec["sent_b"]],
            "is_next": int(same_document and consecutive),
        })
    return pairs