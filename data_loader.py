import fitz


def load_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_file:
        for page_num in range(len(pdf_file)):
            page = pdf_file.load_page(page_num)
            text += page.get_text()
    return text


def load_text_from_txt(txt_path):
    with open(txt_path, 'r', encoding='utf-8') as f:
        return f.read()


def load_corpus(sources):
    corpus = ""
    for source in sources:
        if source.endswith('.pdf'):
            corpus += load_text_from_pdf(source)
        elif source.endswith('.txt'):
            corpus += load_text_from_txt(source)
        # Add more formats as needed
    return corpus 