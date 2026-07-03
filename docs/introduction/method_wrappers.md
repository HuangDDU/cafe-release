# Wrapper Types (TODO)

The **wrapper** is Cafe's abstraction for unifying diverse trajectory inference methods. Every method's output is converted into one of 8 standard wrapper types, each defining a specific data format (stored in `raw_wrapper_dict`). Methods within the same wrapper type share the same visualization and embedding approach.

## Linear.
