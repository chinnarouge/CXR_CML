import pandas as pd

# Load the dataset
file_path = (
    "/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/validation/valid_fold_5.csv"
)
df = pd.read_csv(file_path)

# List of all medical conditions (columns with labels)
condition_columns = df.columns[6:]


# Function to generate the report text for each image
def generate_report(row):
    report = []
    normal_present = False
    support_devices_present = False
    for condition in condition_columns:
        if row[condition] == 1:
            report.append(f"{condition} is present.")
            # else:
            # report.append(f"{condition} is absent.")
    

    '''if not support_devices_present:
        report.append("Support Devices is absent.")'''
        
    return " ".join(report)


# Apply the function to each row to generate the reports
df["impression"] = df.apply(generate_report, axis=1)

# Select only the 'fpath' and 'report' columns for the final output
output_df = df[["fpath", "impression"]]

# Save the output to an Excel file
output_excel_path = "/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/validation/validation_text_fold_5.csv"
output_df.to_csv(output_excel_path, index=False)

print(f"Report generated and saved to {output_excel_path}")
