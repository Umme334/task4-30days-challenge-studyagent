import unittest
from unittest.mock import MagicMock, patch
import io
from pypdf import PdfWriter

class TestPdfProcessing(unittest.TestCase):
    
    def create_dummy_pdf(self):
        """Helper to create a valid in-memory PDF."""
        writer = PdfWriter()
        writer.add_blank_page(width=72, height=72)
        output = io.BytesIO()
        writer.write(output)
        output.seek(0)
        return output

    def test_pdf_reader_integration(self):
        """Test that we can actually read a generated PDF using pypdf."""
        from pypdf import PdfReader
        
        dummy_pdf = self.create_dummy_pdf()
        reader = PdfReader(dummy_pdf)
        
        self.assertEqual(len(reader.pages), 1)

if __name__ == '__main__':
    unittest.main()
