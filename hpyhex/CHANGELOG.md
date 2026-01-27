# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-15 [YANKED]

### Added
- Initial release of HappyHex, but package has no content. This version is yanked.

## [0.2.0] - 2026-01-16

### Added
- Implemented core classes for hexagonal grid system.
- Added all supported API functions for game core.
- Added convenient Game class to manage game state.

## Fixed
- Fixed build system configuration in pyproject.toml and build scripts.

## [0.2.0.1] - 2026-01-18

### Added
- Provided "Serialization Guide" section in documentation in core classes (hex module).
- Added version string to __init__.py.

### Fixed
- Fix Piece.all_pieces() method to return list instead of generator.

## [UNRELEASED]

### Added
- Expanded documentation with detailed description of the API, including all classes and functions.

### Changed
- Refactor solve_radius method in HexEngine for O(1) performance. The original implementation had O(sqrt(n)) complexity due to solving quadratic equation by trial and error. The new implementation uses direct mathematical formula for constant time complexity.

