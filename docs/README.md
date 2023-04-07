# Documentation Build Instructions

SYCLomatic documentation is written using [reStructuredText](http://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html) and built using [Sphinx](http://sphinx-doc.org/). Follow the instructions in this README to build the documentation locally.

## Requirements

The following must be installed to build the documentation:

- Python 3
- Sphinx
- Theme: sphinx_book_theme

## Contribute to the Documentation

Please make yourself familiar with the SYCLomatic [contribution guidelines](https://github.com/oneapi-src/SYCLomatic/blob/SYCLomatic/CONTRIBUTING.md) before submitting a contribution.

For documentation contributions in particular: 

- The [DOC] tag should be present in the commit message title. 
- Contributions should be tested (build with no errors) before submitting a PR.
- PRs should be granular and purpose specific: One PR should contain one unit of change. 
- Submit content changes separate from formatting changes. 

## Build the Documentation

You can build the documentation locally to preview your changes and make sure there are no errors. In the `docs` directory of your local SYCLomatic repository run:

	make html

The generated HTML will be located in `docs/_build/html`.

When testing changes in the documentation, make sure to remove the previous build before building again. To do this run:

	make clean



