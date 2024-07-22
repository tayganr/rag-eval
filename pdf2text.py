import os
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.

    Parameters:
    pdf_path (str): The path to the PDF file from which to extract text.

    Returns:
    str: The extracted text from the PDF.
    """
    text = ''
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

# Example usage
pdf_path = 'YOUR_PATH_TO_THE_PDF/YOUR_PDF.pdf'
extracted_text = extract_text_from_pdf(pdf_path)

# Save extracted text to a file with the same filename but .txt extension
txt_filename = os.path.splitext(os.path.basename(pdf_path))[0] + '.txt'
txt_path = os.path.join('data', 'source_documents_txt', txt_filename)
with open(txt_path, 'w') as file:
    file.write(extracted_text)

print(f"Extracted text saved to: {txt_path}")
