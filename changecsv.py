import pandas as pd

# Load the dataset
file_path = r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\app\data\BackupCrop_Recommendation.csv'
df = pd.read_csv(file_path)

# Remove the 'Rainfall' column
df = df.drop(columns=['Rainfall'])

# Save the updated dataset
updated_file_path = r'C:\Users\ACER\OneDrive - mail.unnes.ac.id\katalis\updated_dataset.csv'
df.to_csv(updated_file_path, index=False)

print("Column removed and file saved as 'updated_dataset.csv'")
