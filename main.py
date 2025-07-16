import os
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod

class PDFProcessor:
    def __init__(self,
                 model_dir: str,
                 layoutreader_model_dir: str,
                 config_path: str,
                 output_dir: str = "output"):
        self.model_dir = model_dir
        self.layoutreader_model_dir = layoutreader_model_dir
        self.config_path = config_path
        self.output_dir = output_dir

        # Prepare output directories
        self.image_dir = os.path.join(self.output_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_writer = FileBasedDataWriter(self.image_dir)
        self.md_writer = FileBasedDataWriter(self.output_dir)

    def process_pdf(self, pdf_file_path: str):
        name_without_suff = os.path.splitext(os.path.basename(pdf_file_path))[0]

        # Read PDF bytes
        reader = FileBasedDataReader("")
        pdf_bytes = reader.read(pdf_file_path)

        # Create Dataset Instance
        ds = PymuDocDataset(pdf_bytes)

        # Perform inference
        if ds.classify() == SupportedPdfParseMethod.OCR:
            infer_result = ds.apply(doc_analyze, ocr=True)
            pipe_result = infer_result.pipe_ocr_mode(self.image_writer)
        else:
            infer_result = ds.apply(doc_analyze, ocr=False)
            pipe_result = infer_result.pipe_txt_mode(self.image_writer)

        # Save results
        infer_result.draw_model(os.path.join(self.output_dir, f"{name_without_suff}_model.pdf"))
        pipe_result.draw_layout(os.path.join(self.output_dir, f"{name_without_suff}_layout.pdf"))
        pipe_result.draw_span(os.path.join(self.output_dir, f"{name_without_suff}_spans.pdf"))

        # Dump markdown and content list
        pipe_result.dump_md(self.md_writer, f"{name_without_suff}.md", self.image_dir)
        pipe_result.dump_content_list(self.md_writer, f"{name_without_suff}_content_list.json", self.image_dir)

if __name__ == "__main__":
    # Example usage
    processor = PDFProcessor(
        model_dir="/home/dave/.cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0/snapshots/38e484355b9acf5654030286bf72490e27842a3c/models",
        layoutreader_model_dir="/home/dave/.cache/huggingface/hub/models--hantian--layoutreader/snapshots/641226775a0878b1014a96ad01b9642915136853",
        config_path="/home/dave/magic-pdf.json",
        output_dir="output"
    )

    processor.process_pdf("test.pdf")

