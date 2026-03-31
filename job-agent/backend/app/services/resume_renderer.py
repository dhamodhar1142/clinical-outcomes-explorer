from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .storage_service import StorageService


class ResumeRenderer:
    def __init__(self) -> None:
        self.storage = StorageService()

    def save_resume_docx(self, application_id: str, filename: str, content: str) -> str:
        return self._write_docx(self.storage.application_dir(application_id) / filename, content)

    def save_resume_pdf(self, application_id: str, filename: str, content: str) -> str:
        return self._write_pdf(self.storage.application_dir(application_id) / filename, content)

    def save_cover_letter_docx(self, application_id: str, filename: str, content: str) -> str:
        return self._write_docx(self.storage.application_dir(application_id) / filename, content)

    def save_cover_letter_pdf(self, application_id: str, filename: str, content: str) -> str:
        return self._write_pdf(self.storage.application_dir(application_id) / filename, content)

    def save_tailoring_prompt(self, application_id: str, filename: str, content: str) -> str:
        return self.storage.write_text(self.storage.application_dir(application_id) / filename, content)

    def _write_docx(self, path: Path, content: str) -> str:
        paragraphs = "".join(
            f"<w:p><w:r><w:t xml:space=\"preserve\">{self._escape_xml(line)}</w:t></w:r></w:p>"
            for line in content.splitlines()
        )
        document_xml = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<w:document xmlns:wpc=\"http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas\" "
            "xmlns:mc=\"http://schemas.openxmlformats.org/markup-compatibility/2006\" "
            "xmlns:o=\"urn:schemas-microsoft-com:office:office\" "
            "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
            "xmlns:m=\"http://schemas.openxmlformats.org/officeDocument/2006/math\" "
            "xmlns:v=\"urn:schemas-microsoft-com:vml\" "
            "xmlns:wp14=\"http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing\" "
            "xmlns:wp=\"http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing\" "
            "xmlns:w10=\"urn:schemas-microsoft-com:office:word\" "
            "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" "
            "xmlns:w14=\"http://schemas.microsoft.com/office/word/2010/wordml\" "
            "xmlns:wpg=\"http://schemas.microsoft.com/office/word/2010/wordprocessingGroup\" "
            "xmlns:wpi=\"http://schemas.microsoft.com/office/word/2010/wordprocessingInk\" "
            "xmlns:wne=\"http://schemas.microsoft.com/office/word/2006/wordml\" "
            "xmlns:wps=\"http://schemas.microsoft.com/office/word/2010/wordprocessingShape\" mc:Ignorable=\"w14 wp14\">"
            "<w:body>"
            f"{paragraphs}"
            "<w:sectPr><w:pgSz w:w=\"12240\" w:h=\"15840\"/><w:pgMar w:top=\"1440\" w:right=\"1440\" w:bottom=\"1440\" w:left=\"1440\"/></w:sectPr>"
            "</w:body></w:document>"
        )
        content_types = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
            "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml\"/>"
            "</Types>"
        )
        rels = (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>"
            "</Relationships>"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", rels)
            archive.writestr("word/document.xml", document_xml)
        return str(path)

    def _write_pdf(self, path: Path, content: str) -> str:
        lines = [line[:90] for line in content.splitlines() if line.strip()] or [content[:90]]
        text_commands = []
        y = 780
        for line in lines[:45]:
            escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            text_commands.append(f"1 0 0 1 40 {y} Tm ({escaped}) Tj")
            y -= 16
        stream = "BT /F1 11 Tf " + " ".join(text_commands) + " ET"
        objects = [
            "1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
            "2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
            "3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
            "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
            f"5 0 obj << /Length {len(stream.encode('utf-8'))} >> stream\n{stream}\nendstream endobj",
        ]
        pdf = "%PDF-1.4\n"
        offsets = [0]
        for obj in objects:
            offsets.append(len(pdf.encode("utf-8")))
            pdf += obj + "\n"
        xref_start = len(pdf.encode("utf-8"))
        pdf += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
        for offset in offsets[1:]:
            pdf += f"{offset:010d} 00000 n \n"
        pdf += f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pdf.encode("utf-8"))
        return str(path)

    def _escape_xml(self, text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )
