def validate_records(records, schema):
    """
    Validate records against a schema definition.
    """
    # Write code here
    results = []

    for record_index, record in enumerate(records):
        errors = []

        for rule in schema:
            column = rule["column"]
            expected_type = rule["type"]

            # 1. Missing column
            if column not in record:
                errors.append(f"{column}: missing")
                continue

            value = record[column]

            # 2. Null check
            if value is None:
                if not rule.get("nullable", False):
                    errors.append(f"{column}: null")
                continue

            # 3. Type check
            type_valid = False

            if expected_type == "float":
                # float accepts int and float, but not bool
                type_valid = type(value) in (int, float)

            elif expected_type == "int":
                type_valid = type(value) is int

            elif expected_type == "str":
                type_valid = type(value) is str

            elif expected_type == "bool":
                type_valid = type(value) is bool

            if not type_valid:
                errors.append(
                    f"{column}: expected {expected_type}, "
                    f"got {type(value).__name__}"
                )
                continue

            # 4. Range check
            if type(value) in (int, float):
                min_value = rule.get("min")
                max_value = rule.get("max")

                if min_value is not None and value < min_value:
                    errors.append(f"{column}: out of range")
                elif max_value is not None and value > max_value:
                    errors.append(f"{column}: out of range")

        results.append(
            (record_index, len(errors) == 0, errors)
        )

    return results