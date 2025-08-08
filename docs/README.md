To edit the docs, install `mkdocs-material` via 
```
pip install mkdocs-material
```

To preview changes to the docs,
```
mkdocs serve
```

After editing, build a static site with
```
mkdocs build
```

<!-- To publish, run this command from the directory containing the mkdocs.yml file:
```
mkdocs gh-deploy --force
``` -->

To publish, run
```
sh build_docs.sh
```
to build the static site, and commit and push the changes to the `signed-heat-3d` repo. Deploy to GitHub Pages via
```
mkdocs gh-deploy --force
```