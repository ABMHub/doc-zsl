import os
import random
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageOps

def create_image_grid(folder_path, output_filename):
    # Lista todas as imagens na pasta
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif')
    images = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
    
    if len(images) < 4:
        raise ValueError("A pasta deve conter pelo menos 4 imagens.")
    
    # Escolhe 4 imagens aleatoriamente
    selected_images = random.sample(images, 4)
    
    # Carrega as imagens
    img_paths = [os.path.join(folder_path, img) for img in selected_images]
    img_list = [Image.open(img_path) for img_path in img_paths]
    
      # Mantém a proporção A4
    
    # Redimensiona as imagens para a proporção A4
    border_size = 5  # Define a espessura da linha preta
    img_list = [ImageOps.expand(img.convert('L'), border=border_size, fill='black') for img in img_list]
    
    # Criar a figura
    fig, axes = plt.subplots(2, 2, figsize=(12, 16), gridspec_kw={'wspace': 0.05, 'hspace': 0.05})  # Ajusta a figura para a proporção A4
    
    for ax, img, path in zip(axes.ravel(), img_list, img_paths):
        ax.imshow(img, cmap='gray')
        
        ax.axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0)
    output_path = os.path.abspath(os.path.join(folder_path, output_filename))
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close(fig)
    # plt.show()  # Descomente esta linha para visualizar a imagem na tela
    
create_image_grid("./separacao-rvl_cdip/haleton_material_safety_data_sheet", "./plot/class_example_1.png")

