# Stage 01: Language Understanding, Normalization, Neural Rewriting & Intent Classification
from .arat5_rewriter import AraT5DialectRewriter
from .query_preprocessor import QueryPreprocessor

__all__ = ["AraT5DialectRewriter", "QueryPreprocessor"]

# Backward compatibility alias
MSANormalizerPipeline = QueryPreprocessor
