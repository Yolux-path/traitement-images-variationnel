# Projet_MTI

TP2 de MTI : débruitage, déconvolution et inpainting par approches variationnelles.

- `TP2_jupyter_empty.ipynb` : le notebook (source de vérité).
- `notebook_export/TP2_notebook.py` : export `.py` du notebook, régénéré à chaque commit. Ne pas le modifier à la main.
- `TP2_questions_papier.tex` : réponses aux questions « sur papier ».

## Installer le hook d'export (une fois, après le clone)

```sh
ln -s ../../hooks/pre-commit .git/hooks/pre-commit
```

Sans ce hook, l'export `.py` n'est plus mis à jour quand le notebook change.
