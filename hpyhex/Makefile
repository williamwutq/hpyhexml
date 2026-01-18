# Makefile for building and publishing the hpyhex package

help:
	@echo "Make commands:"
	@echo "  clean   - Remove the dist directory"
	@echo "  build   - Build the package"
	@echo "  publish - Publish the package to PyPI"

clean:
	@rm -rf dist

build:
	@python -m build

publish:
	@python -m twine upload dist/*