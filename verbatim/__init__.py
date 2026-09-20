"""Extractive, grounded question-answering over a resume PDF.

No text is generated anywhere in this package. A BM25 shortlist narrows the
corpus, TypeSafe's System One primitives (Noul) rerank and gate the
candidates, and the final "answer" is always the source text itself.
"""
