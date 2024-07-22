import pandas as pd  
import json  
  
# Load the CSV file  
csv_file_path = 'your_file.csv'  # Replace with your actual CSV file path  
df = pd.read_csv(csv_file_path)  
  
# Inspect the DataFrame to ensure columns are read correctly  
print("DataFrame columns:", df.columns)  
  
# Ensure the columns are correctly recognized  
expected_columns = ['question', 'contexts', 'ground_truth']  
for col in expected_columns:  
    if col not in df.columns:  
        raise ValueError(f"Expected column '{col}' not found in CSV file")  
  
# Initialize the JSON structure  
data = {  
    "question": df['question'].tolist(),  
    "contexts": df['contexts'].apply(lambda x: [x] if x else []).tolist(),  
    "ground_truth": df['ground_truth'].tolist()  
}  
  
# Convert the dictionary to a JSON string  
json_data = json.dumps(data, indent=4)  
  
# Save the JSON data to a file  
json_file_path = 'output.json'  # Replace with your desired output file path  
with open(json_file_path, 'w') as json_file:  
    json_file.write(json_data)  
  
print(f"JSON data has been saved to {json_file_path}")  
