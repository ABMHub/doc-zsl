#
# Move todos os arquivos presentes nas subpastas dessa pasta
# para a pasta raiz. Deleta as pastas vazias
#

folder = "/mnt/c/Users/lucas/Datasets/rvl-cdip"

import os
import shutil

def flatten(root_folder, current_folder):
    contents = os.listdir(current_folder)
    for elem in contents:
        elem_path = os.path.join(current_folder, elem)
        if os.path.isdir(elem_path):
            flatten(root_folder, elem_path)
            try: # se tiver arquivo repetido vai dar erro deletando a pasta
                os.rmdir(elem_path)
            except:
                pass

        else:
            if root_folder == current_folder: continue
            try: # pode ocorrer erro no move por arquivo repetido
                shutil.move(elem_path, root_folder)
            except: 
                print(f"erro no arquivo {elem_path}")

flatten(folder, folder)
