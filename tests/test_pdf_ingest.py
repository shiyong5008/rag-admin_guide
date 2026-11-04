import unittest

from pdf_ingest import _clean_text_block, chunk_elements


class CleanTextBlockTests(unittest.TestCase):
    def test_skips_lines_matching_known_footer_suffixes(self) -> None:
        block = {
            "bbox": (0, 0, 100, 100),
            "lines": [
                {"spans": [{"text": "Updated Vendor net Inc."}]},
                {"spans": [{"text": "Product Analyzer 7.0.0 Administration Guide"}]},
                {"spans": [{"text": "Keep this text."}]},
            ],
        }

        text, is_code = _clean_text_block(block, "Document")

        self.assertEqual(text, "Keep this text.")
        self.assertFalse(is_code)


class ChunkElementsTests(unittest.TestCase):
    def test_text_chunks_keep_page_and_section_boundaries(self) -> None:
        chunks = chunk_elements(
            [
                {"kind": "text", "text": "First page text.", "page": 12, "section": "Setup"},
                {"kind": "text", "text": "Second page text.", "page": 13, "section": "Setup"},
                {"kind": "text", "text": "New section text.", "page": 14, "section": "Devices"},
            ]
        )

        self.assertEqual(
            [(chunk["page_start"], chunk["section"]) for chunk in chunks],
            [(12, "Setup"), (13, "Setup"), (14, "Devices")],
        )

    def test_code_and_image_elements_stay_atomic(self) -> None:
        code = "```text\nconfig system global\nend\n```"
        chunks = chunk_elements(
            [
                {"kind": "text", "text": "Configure the device.", "page": 20, "section": "System"},
                {"kind": "code", "text": code, "page": 20, "section": "System"},
                {
                    "kind": "image",
                    "text": "Image in System. Nearby text: Dashboard settings.",
                    "page": 21,
                    "section": "System",
                    "image_path": "data/images/page_0021_image_01.png",
                    "image_ext": "png",
                },
            ]
        )

        self.assertEqual(chunks[1]["kind"], "code")
        self.assertEqual(chunks[1]["text"], code)
        self.assertEqual(chunks[2]["kind"], "image")
        self.assertEqual(chunks[2]["page_start"], 21)
        self.assertEqual(chunks[2]["image_path"], "data/images/page_0021_image_01.png")


if __name__ == "__main__":
    unittest.main()