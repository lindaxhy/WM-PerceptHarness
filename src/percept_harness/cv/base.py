"""Provider-independent computer-vision evidence errors."""


class CvProviderError(RuntimeError):
    """A sanitized CV provider boundary failure."""


class CvOutOfMemoryError(CvProviderError):
    """A provider-reported allocation failure eligible for one retry."""
