# Contributing to Floki

Thank you for considering contributing to **Floki**! Your ideas and efforts mean a lot to us, and together, we can make this project even better. This guide will walk you through everything you need to know to get started and make an impact.

---

## Code of Conduct

By contributing, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please take a moment to read it before starting.

---

## How to Contribute

### 1. Report Issues or Suggest Features

If you encounter a bug or have an idea for a new feature:

1. Check the [issue tracker](https://github.com/Cyb3rWard0g/floki/issues) to see if it’s already reported.
2. If not, open a new issue with:
   - A clear title and description.
   - Steps to reproduce (for bugs).
   - Why the feature would be useful (for feature requests).

### 2. Set Up Your Development Environment

Follow these steps to get started with the project:

1. **Fork the Repository**: Click the `Fork` button on the top-right of the GitHub page to create your own copy of the repository.
2. **Clone Your Fork**:

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

3. Set Up a Virtual Environment (optional but recommended):

```python
python -m venv .venv
source .venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

4. Install the Project in Editable Mode:

```bash
pip install -e .
```

5. Check the installed package version to confirm:

```bash
pip list | grep floki-ai
```

### 3. Create a New Branch

Always create a new branch for your changes to keep things organized:

```bash
git checkout -b feature/your-feature-name
```

Follow these branch naming conventions:

* feature/short-description for new features.
* bugfix/short-description for bug fixes.
* hotfix/short-description for urgent fixes.

### 4. Make Changes

* Focus on a single feature or fix per branch.
* Add tests if your change involves functionality.
* Update documentation if necessary.

### 5. Commit Your Changes

Write clear, meaningful commit messages:

```bash
git add .
git commit -m "feat: add user authentication feature"
```

### 6. Push Your Branch

Push your changes to your forked repository:

```bash
git push origin feature/your-feature-name
```

### 7. Open a Pull Request

1. Go to your forked repository on GitHub and click `New Pull Request.`
2. Ensure your branch is being merged into the main branch.
3. Include a clear title and description of your changes.
4. Add any related issue numbers (e.g., Closes #123).

### 8. Building the Documentation Locally

To contribute to Floki's documentation, you can preview the docs locally using Docker. Follow these steps:

1. Ensure you have Docker installed and running on your system. If not, download Docker here.
2. Navigate to the project's root directory where the documentation files are located.
3. Run the following command to start a local documentation server:

```bash
docker run --rm -it -p 8000:8000 -v ${PWD}:/docs squidfunk/mkdocs-material
```

4. Open your browser and go to the following URL to view the docs:

```
http://0.0.0.0:8000/floki/
```

5. Any changes you make to the Markdown files under the /docs folder will automatically reflect in the preview.

## Reviewing Pull Requests

* Be responsive to feedback.
* Make any requested changes promptly.
* Once approved, a maintainer will merge your pull request.

## Cleaning Up Local and Remote Branches

Once your Pull Request (PR) has been merged or closed, it’s a good practice to clean up your local and remote branches. This helps keep your development environment and fork organized.

### Delete Local Branch

```bash
git branch -D feature/your-feature-name
```

### Delete Remote Branch

```bash
git push origin --delete feature/your-feature-name
```

## Need Help?

If you have any questions, open an issue with the question label.

Thank you for contributing! 🎉