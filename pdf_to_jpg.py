from pdf2image import convert_from_path

def convert_pdf_to_jpg(pdf_path, output_folder):
    images = convert_from_path(pdf_path)
    
    for i, image in enumerate(images):
        image.save(f"{output_folder}/page_{i + 1}.jpg", "JPEG")

if __name__ == "__main__":
    pdf_file_path = "./mappa_prova.pdf"
    output_folder_path = "./cartella_di_output"

    convert_pdf_to_jpg(pdf_file_path, output_folder_path)