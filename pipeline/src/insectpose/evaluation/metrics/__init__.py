"""Metriques. Une metrique = un module + un nom enregistre (CONVENTIONS.md §7.2).

Signature imposee : fn(bundle: EvalBundle) -> list[dict] (lignes du contrat 4).
Aucune metrique ne lit un modele, un log de framework ou un fichier hors bundle.
"""
