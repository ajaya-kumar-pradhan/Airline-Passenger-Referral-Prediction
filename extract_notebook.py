import json

def extract_code(notebook_path, output_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    code_cells = []
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            code_cells.append(source)
            
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n\n# %% NEW CELL\n\n".join(code_cells))
    
    print(f"Extracted {len(code_cells)} code cells to {output_path}")

if __name__ == "__main__":
    extract_code(r"C:\Users\ajaya\Downloads\Copy_of_Book_Recommendation_System_(Unsupervised_Learning_Project).ipynb", "notebook_logic.py")
