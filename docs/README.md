# Evol Virtual Creature Documentation Site

This directory contains a static GitHub Pages documentation site.

To publish it:

1. Push the repository to GitHub.
2. Open the repository settings.
3. Go to **Pages**.
4. Choose **Deploy from a branch**.
5. Select the branch and the `/docs` folder.
6. Save the settings.

GitHub will serve `docs/index.html` as the project documentation page.


## Editing the Documentation

- `index.html`: main user-facing guide and concept overview.
- `api-reference.json`: structured API content rendered by `api.html`.
- `api.html`: static renderer for the API JSON. Use `api.html?module=evol_virtual_creature.genes` to link directly to one module.
- `styles.css`: documentation site styling.
