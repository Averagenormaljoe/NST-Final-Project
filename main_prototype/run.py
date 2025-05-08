import os


notebook_path = "book_prototype.ipynb"  


command = f"python -m jupyter nbconvert --execute --to notebook --inplace {notebook_path}"


success = os.system(command)
if success != 0:
    print("Notebook execution completed!")
else:
    print("Notebook execution failed!")