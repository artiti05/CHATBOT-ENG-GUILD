# Stage 01: Language Understanding, Normalization, Neural Rewriting & Intent Classification
from .query_preprocessor import QueryPreprocessor
from .arat5_rewriter import AraT5DialectRewriter

# Backward compatibility alias
MSANormalizerPipeline = QueryPreprocessor
