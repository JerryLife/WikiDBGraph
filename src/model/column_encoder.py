import numpy as np
import pandas as pd
import mmh3

modes = [
    "cross_encoder",
    "header_values_default",
    "header_values_prefix",
    "header_values_repeat",
    "header_values_verbose",
    "header_only",
    "header_values_verbose_notype",
    "header_values_columnvaluepair_notype",
    "header_header_values_repeat_notype",
    "header_values_default_notype",
    "database_table_column_values",
    "table_column_values",
]

sampling_modes = [
    "random",
    "frequent",
    "mixed",
    "weighted",
    "priority_sampling",
    "consistent_sampling",
]
PHI_FRACTION = 0.6180339887

def fibonacci_hash(x):
    result = (x * PHI_FRACTION) % 1  # Take fractional part
    return result

def get_samples(values, n=15, mode="priority_sampling"):
    """
    Sample values from a pandas Series using different strategies.

    Args:
        values: pandas Series containing the values to sample
        n: number of samples to return (default: 15)
        mode: sampling strategy ('random', 'frequent', or 'mixed')
            - 'random': completely random sampling from unique values
            - 'frequent': only the most frequent values
            - 'mixed': combination of frequent and diverse values
            - 'weighted': weighted sampling based on value counts
            - 'priority_sampling': uses priority sampling based on frequency and hash of the values
            - 'consistent_sampling': consistent uniform sampling based on hash of the values

    Returns:
        List of string representations of sampled values
    """
    unique_values = values.dropna().unique()
    total_unique = len(unique_values)

    # If total unique values are fewer than n, return them all
    if total_unique <= n:
        return sorted([str(val) for val in unique_values])

    if mode == "random":
        # Completely random sampling
        random_indices = np.random.choice(total_unique, size=n, replace=False)
        sampled_values = unique_values[random_indices]
        tokens = sorted(sampled_values)

    elif mode == "frequent":
        # Only most frequent values
        value_counts = values.dropna().value_counts()
        tokens = value_counts.head(n).index.tolist()
        tokens.sort()

    elif mode == "mixed":
        # Mix of most frequent and evenly spaced values
        n_frequent = n // 2
        value_counts = values.dropna().value_counts()
        most_frequent_values = value_counts.head(n_frequent).index.tolist()

        # Calculate evenly spaced samples for diversity
        n_diverse = n - n_frequent
        spacing_interval = max(1, total_unique // n_diverse)
        diverse_values = unique_values[::spacing_interval][:n_diverse]

        # Combine frequent and diverse samples, remove duplicates
        # tokens = sorted(set(most_frequent_values + list(diverse_values)))
        tokens = sorted(set(map(str, most_frequent_values + list(diverse_values))))

    elif mode == "weighted":
        # Weighted sampling based on value counts
        value_counts = values.dropna().value_counts(sort=False)
        weights = value_counts / value_counts.sum()
        sampled_indices = np.random.choice(
            total_unique, size=n, replace=False, p=weights
        )
        sampled_values = unique_values[sampled_indices]
        tokens = sampled_values

    elif mode == "priority_sampling":
        value_counts = values.dropna().value_counts(sort=False)

        # Calculate priorities: qi = freq / hash(value)
        priorities = pd.Series(
            {
                val: freq / fibonacci_hash(mmh3.hash(str(val), 42))
                for val, freq in value_counts.items()
            }
        )

        # Select the top elements based on priority scores
        sampled_values = priorities.nlargest(n).index.tolist()
        tokens = sampled_values

    elif mode == "consistent_sampling":
        value_counts = values.dropna().value_counts(sort=False)

        priorities = pd.Series(
            {
                val: fibonacci_hash(mmh3.hash(str(val), 42))
                for val in value_counts.keys()
            }
        )

        # Select the top elements based on priority scores
        sampled_values = priorities.nlargest(n).index.tolist()
        tokens = sampled_values

    else:
        raise ValueError(
            f"Unsupported mode: {mode}. Use 'random', 'frequent', 'mixed','weighted', 'priority_sampling' or 'consistent_sampling'"
        )

    return [str(token) for token in tokens]




class ColumnEncoder:
    def __init__(
        self,
        tokenizer=None,
        encoding_mode="table_header_values_repeat",
        sampling_mode="mixed",
        n_samples=10,
    ):
        if tokenizer is not None:
            self._tokenizer = tokenizer
            self.cls_token = getattr(tokenizer, "cls_token", "")
            self.sep_token = getattr(tokenizer, "sep_token", "")
            self.eos_token = getattr(tokenizer, "eos_token", "")

        self._serialization_methods = {
            "cross_encoder": self._serialize_cross_encoder,
            "database_table_column_values": self._serialize_database_table_column_values,
            "table_column_values": self._serialize_table_column_values,
            "table_header_values_repeat": self._serialize_table_header_values_repeat,
            "header_values_default": self._serialize_header_values_default,
            "header_values_prefix": self._serialize_header_values_prefix,
            "header_values_repeat": self._serialize_header_values_repeat,
            "header_values_verbose": self._serialize_header_values_verbose,
            "header_only": self._serialize_header_only,
            "header_values_verbose_notype": self._serialize_header_values_verbose_notype,
            "header_values_columnvaluepair_notype": self._serialize_header_values_columnvaluepair_notype,
            "header_header_values_repeat_notype": self._serialize_header_values_repeat_notype,
            "header_values_default_notype": self._serialize_header_values_default,
        }

        if encoding_mode not in self._serialization_methods:
            raise ValueError(
                f"Unsupported encoding mode: {encoding_mode}. Supported modes are: {list(self._serialization_methods.keys())}"
            )
        if sampling_mode not in sampling_modes:
            raise ValueError(
                f"Unsupported sampling mode: {sampling_mode}. Supported modes are: {sampling_modes}"
            )

        self.encoding_mode = encoding_mode
        self.sampling_mode = sampling_mode
        self.n_samples = n_samples

    def encode(self, df, col):
        """Encodes the column of a DataFrame using the selected serialization method."""
        header = col
        tokens = get_samples(df[col], n=self.n_samples, mode=self.sampling_mode)
        # data_type = detect_column_type(df[col])
        # return self._serialization_methods[self.encoding_mode](
        #     header, data_type, tokens
        # )
        return self._serialization_methods[self.encoding_mode](
            header, tokens
        )

    def _serialize_header_values_verbose(self, header, data_type, tokens):
        """Serializes with detailed column header, type, and token values."""
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Type: {data_type}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
        )

    def _serialize_header_values_default(self, header, data_type, tokens):
        """Serializes with default format including header, type, and tokens."""
        return (
            f"{self.cls_token}"
            f"{header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_header_values_prefix(self, header, data_type, tokens):
        """Serializes with prefixed labels for header, datatype, and values."""
        return (
            f"{self.cls_token}"
            f"header:{header}{self.sep_token}"
            f"datatype:{data_type}{self.sep_token}"
            f"values:{', '.join(tokens)}"
        )

    def _serialize_header_values_repeat(self, header, data_type, tokens):
        """Serializes with repeated header for emphasis."""
        repeated_header = self.sep_token.join([header] * 5)
        return (
            f"{self.cls_token}"
            f"{repeated_header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_table_header_values_repeat(self, header, tokens):
        """Serializes with repeated header for emphasis."""
        table_name, header = header.split("::")
        repeated_header = self.sep_token.join([header] * 3)
        return (
            f"{self.cls_token}"
            f"{table_name}{self.sep_token}"
            f"{repeated_header}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_database_table_column_values(self, header, tokens):
        """Serializes with repeated header for emphasis."""
        db_name, table_name, header = header.split("::")
        # repeated_header = [header] * 3
        return (
            f"Database: {db_name} / "
            f"Table: {table_name} / "
            f"Column: {header} / "
            f"Sample values: {tokens}"
        )

    def _serialize_table_column_values(self, header, tokens):
        """Serializes with repeated header for emphasis."""
        table_name, header = header.split("::")
        return (
            f"Table: {table_name} / "
            f"Column: {header} / "
            f"Sample values: {tokens}"
        )

    def _serialize_database_table_column_values_refined(self, header, tokens):
        """Serializes with repeated header for emphasis."""
        db_name, table_name, header = header.split("::")
        # repeated_header = [header] * 3
        return (
            f"Column: {header} "
            f"(from table: {table_name} in database: {db_name}): "
            f"Sample values: {tokens}"
        )
    def _serialize_cross_encoder(self, header, tokens):
        """Serializes with repeated header for emphasis."""
        table_name, header = header.split("::")
        repeated_header = header
        return (
            f"table: {table_name}|"
            f"column: {repeated_header}|"
            f"sample values: {tokens}"
        )

    def _serialize_header_only(self, header, data_type, tokens):
        """Serializes with header only."""
        return f"{self.cls_token}" f"{header}" f"{self.eos_token}"

    def _serialize_header_values_verbose_notype(self, header, data_type, tokens):
        """Serializes with simple format including header and tokens."""
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
            f"{self.eos_token}"
        )

    def _serialize_header_values_columnvaluepair_notype(
        self, header, data_type, tokens
    ):

        tokens = [f"{header}:{token}" for token in tokens]
        return (
            f"{self.cls_token}"
            f"Column: {header}{self.sep_token}"
            f"Values: {self.sep_token.join(tokens)}{self.sep_token}"
            f"{self.eos_token}"
        )

    def _serialize_header_values_repeat_notype(self, header, data_type, tokens):
        """Serializes with repeated header for emphasis."""
        repeated_header = self.sep_token.join([header] * 5)
        return (
            f"{self.cls_token}"
            f"{repeated_header}{self.sep_token}"
            f"{data_type}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )

    def _serialize_header_values_default_notype(self, header, data_type, tokens):

        return (
            f"{self.cls_token}"
            f"{header}{self.sep_token}"
            f"{self.sep_token.join(tokens)}"
        )
