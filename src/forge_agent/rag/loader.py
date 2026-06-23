"""Markdown document loader for local knowledge bases."""

from __future__ import annotations

from pathlib import Path

from forge_agent.rag.document import Document, DocumentMetadata


class MarkdownLoader:
    """Load Markdown files from a local directory."""

    _markdown_suffixes = {".md", ".markdown"}

    def load_dir(self, directory: str | Path) -> list[Document]:
        """Load all Markdown files under a directory.

        Files are returned in deterministic path order so tests and index output
        remain stable across runs.
        """
        root = Path(directory)

        if not root.exists():
            raise FileNotFoundError(f"Knowledge base directory does not exist: {root}")

        if not root.is_dir():
            raise NotADirectoryError(f"Knowledge base path is not a directory: {root}")

        markdown_files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and self._is_markdown_file(path)
        )

        return [self.load_file(path, root=root) for path in markdown_files]

    def load_file(self, path: str | Path, root: str | Path | None = None) -> Document:
        """Load one Markdown file as a Document."""
        source_path = Path(path)

        if not source_path.exists():
            raise FileNotFoundError(f"Markdown file does not exist: {source_path}")

        if not source_path.is_file():
            raise IsADirectoryError(f"Markdown path is not a file: {source_path}")

        if not self._is_markdown_file(source_path):
            raise ValueError(f"Unsupported Markdown file suffix: {source_path}")

        content = source_path.read_text(encoding="utf-8")
        relative_path = self._relative_path(source_path, root)
        title = self._extract_title(content) or source_path.stem

        metadata = DocumentMetadata(
            source_path=source_path.as_posix(),
            relative_path=relative_path,
            source_id=f"markdown:{relative_path}",
            title=title,
            mtime=source_path.stat().st_mtime,
        )

        return Document(content=content, metadata=metadata)

    @classmethod
    def _is_markdown_file(cls, path: Path) -> bool:
        return path.suffix.lower() in cls._markdown_suffixes

    @staticmethod
    def _relative_path(path: Path, root: str | Path | None) -> str:
        if root is None:
            return path.name

        root_path = Path(root)

        try:
            return path.relative_to(root_path).as_posix()
        except ValueError:
            return path.name

    @staticmethod
    def _extract_title(content: str) -> str | None:
        for line in content.splitlines():
            stripped = line.strip()

            if stripped.startswith("# ") and len(stripped) > 2:
                return stripped[2:].strip()

        return None
