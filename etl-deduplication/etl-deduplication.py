def deduplicate(records, key_columns, strategy):
    """
    Deduplicate records by key columns using the given strategy.
    """
    # Write code here
    result = {}

    for record in records:
        # Create tuple key from key_columns
        key = tuple(record[col] for col in key_columns)

        if key not in result:
            result[key] = record

        else:
            if strategy == "last":
                result[key] = record

            elif strategy == "most_complete":
                current = result[key]

                # Count None values
                current_none = sum(v is None for v in current.values())
                new_none = sum(v is None for v in record.values())

                # Keep record with fewer None values
                if new_none < current_none:
                    result[key] = record

            # "first" keeps existing record, so do nothing

    return list(result.values())