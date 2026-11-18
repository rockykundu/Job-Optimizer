from __future__ import annotations

import re
from io import BytesIO

from pypdf import PdfReader


class PdfProcessingError(ValueError):
    pass


def extract_pdf_text(
    content: bytes,
    maximum_pages: int,
    maximum_characters: int,
) -> str:
    if not content.startswith(b"%PDF-"):
        raise PdfProcessingError("The uploaded file is not a valid PDF.")
    try:
        reader = PdfReader(BytesIO(content), strict=False)
        if reader.is_encrypted:
            try:
                decrypt_result = reader.decrypt("")
            except Exception as exc:
                raise PdfProcessingError(
                    "Password-protected PDFs are not supported."
                ) from exc
            if not decrypt_result:
                raise PdfProcessingError(
                    "Password-protected PDFs are not supported."
                )
        if len(reader.pages) == 0:
            raise PdfProcessingError("The uploaded PDF contains no pages.")
        if len(reader.pages) > maximum_pages:
            raise PdfProcessingError(
                f"The resume exceeds the {maximum_pages}-page limit."
            )

        pages: list[str] = []
        character_count = 0
        for page in reader.pages:
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue
            character_count += len(page_text)
            if character_count > maximum_characters:
                raise PdfProcessingError(
                    "The extracted resume text exceeds the permitted size."
                )
            pages.append(page_text)
    except PdfProcessingError:
        raise
    except Exception as exc:
        raise PdfProcessingError(
            "The PDF could not be read. Upload a valid text-based PDF."
        ) from exc

    text = re.sub(r"\s+", " ", "\n".join(pages)).strip()
    if len(text) < 50:
        raise PdfProcessingError(
            "Not enough text was extracted. Scanned image-only PDFs are not supported."
        )
    return text
