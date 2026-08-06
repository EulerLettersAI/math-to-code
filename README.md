# Math to Code

Executable educational lessons that connect mathematical notation to working
code. The site is designed for publication at:

<https://eulerlettersai.github.io/math-to-code/>

## Work locally

Install [Quarto](https://quarto.org/docs/get-started/) and Python 3.11, then:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
quarto preview
```

To work directly in the notebooks:

```bash
jupyter lab
```

## Publish

Pushing to `main` runs `.github/workflows/publish.yml`. In the GitHub
repository settings, set **Pages → Build and deployment → Source** to
**GitHub Actions** once. The workflow then renders and deploys the site.

## First lesson

[Classical and quantum kernel classifiers](lessons/classical-vs-quantum/index.ipynb)
compares a classical RBF support-vector classifier with a quantum-kernel
support-vector classifier using the same dataset and split.

