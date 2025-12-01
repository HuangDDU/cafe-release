from contextlib import ContextDecorator


class temporary_obsm_key(ContextDecorator):
    """
    A context manager replace key for adata.obsm temporarily.
    """

    def __init__(self, adata, key, value):
        self.adata = adata
        self.key = key
        self.value = value
        self.original_value = None
        self.key_existed = False

    def __enter__(self):
        if self.key in self.adata.obsm:
            self.key_existed = True
            self.original_value = self.adata.obsm[self.key]

        self.adata.obsm[self.key] = self.value
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.key_existed:
            self.adata.obsm[self.key] = self.original_value
        else:
            del self.adata.obsm[self.key]
