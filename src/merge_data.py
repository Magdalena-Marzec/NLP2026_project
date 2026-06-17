import pandas as pd
import os

# =====================================================================
# CONFIGURATION ARGUMENTS
# =====================================================================
# Replace these mock filenames with your actual unzipped GEO file paths.
INPUT_FILES = [
"GSM2647305_w1118_AC_m_r1.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647306_w1118_AC_m_r2.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647307_w1118_AC_m_r3.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2817328_w1118_AC_m_r4.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647311_w1118_DG_m_r1.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647312_w1118_DG_m_r2.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647313_w1118_DG_m_r3.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2817330_w1118_DG_m_r4.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647317_w1118_GE_m_r1.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647318_w1118_GE_m_r2.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647319_w1118_GE_m_r3.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2817332_w1118_GE_m_r4.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647323_w1118_GO_m_r1.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647324_w1118_GO_m_r2.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647325_w1118_GO_m_r3.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2817334_w1118_GO_m_r4.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647329_w1118_HD_m_r1.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647330_w1118_HD_m_r2.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647331_w1118_HD_m_r3.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2817336_w1118_HD_m_r4.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647335_w1118_RE_m_r1.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647336_w1118_RE_m_r2.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647337_w1118_RE_m_r3.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2817338_w1118_RE_m_r4.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647341_w1118_TX_m_r1.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647342_w1118_TX_m_r2.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2647343_w1118_TX_m_r3.htseq_reverse.HiSAT2.FB.txt.gz",
"GSM2817340_w1118_TX_m_r4.htseq_reverse.HiSAT2.FB.txt.gz",
]

# The condition list acts as metadata reference for your samples.
# Matching the exact index order of the INPUT_FILES above.
CONDITIONS = [
    "ABDOMEN",
    "ABDOMEN",
    "ABDOMEN",
    "ABDOMEN",
    "DIGESTIVE",
    "DIGESTIVE",
    "DIGESTIVE",
    "DIGESTIVE",
    "GENITALIA",
    "GENITALIA",
    "GENITALIA",
    "GENITALIA",
    "GONADS",
    "GONADS",
    "GONADS",
    "GONADS",
    "HEAD",
    "HEAD",
    "HEAD",
    "HEAD",
    "REPRODUCTIVE",
    "REPRODUCTIVE",
    "REPRODUCTIVE",
    "REPRODUCTIVE",
    "THORAX",
    "THORAX",
    "THORAX",
    "THORAX",
]

OUTPUT_FILE = "w1118_raw_counts_matrix.csv"
DATA_DIR = "./data/"


# =====================================================================
# PROCESSING FUNCTION
# =====================================================================
def build_count_matrix(file_paths, conditions, output_path, data_dir=None):
    """
    Reads individual 2-column count files, aligns them by Gene ID, 
    and concatenates them into a single (Genes x Samples) matrix.
    
    Parameters:
        file_paths (list of str): List of filenames or relative/absolute paths.
        conditions (list of str): List of sample conditions (same length/order as file_paths).
        output_path (str): Output CSV file path for the merged matrix.
        data_dir (str, optional): Optional directory/folder to prepend to each filename in file_paths.
    """
    dataframes = []

    if len(file_paths) != len(conditions):
        raise ValueError(f"Mismatch: {len(file_paths)} files vs {len(conditions)} conditions")    

    for filepath, condition in zip(file_paths, conditions):
        # If a data directory is provided, prepend it to the filename
        if data_dir is not None:
            full_path = os.path.join(data_dir, filepath)
        else:
            full_path = filepath

        # Extract a clean sample name from the filename to use as the column header
        base_name = os.path.basename(filepath).replace('.txt', '').replace('.csv', '')

        # Combine condition and basename for highly readable columns 
        # (e.g., "HEAD_GSM2647329_w1118_HD_m_r1")
        sample_col_name = f"{condition}_{base_name}"

        # Load the file. GEO HTSeq files usually lack headers and are tab-separated.
        # If your files are strictly comma-separated, change sep='\t' to sep=','
        try:
            df = pd.read_csv(
                full_path, 
                sep='\t', 
                header=None, 
                names=['Gene_ID', sample_col_name],
                compression="gzip"
            )

            # Set the Gene ID as the index. This ensures that when we concatenate,
            # pandas matches the counts to the correct gene perfectly.
            df.set_index('Gene_ID', inplace=True)
            dataframes.append(df)

            print(f"Loaded {sample_col_name} -> {len(df)} genes.")

        except FileNotFoundError:
            print(f"ERROR: Could not find file -> {full_path}")
            return

    print("\nAligning and concatenating all samples...")
    # Concatenate all dataframes horizontally (axis=1) joining on the Gene_ID index
    merged_matrix = pd.concat(dataframes, axis=1)

    # Optional: HTSeq files often contain meta-rows at the bottom starting with '__'
    # (e.g., __no_feature, __ambiguous). We filter those out to keep only true FBgn IDs.
    merged_matrix = merged_matrix[~merged_matrix.index.str.startswith('__')]

    # Save the final matrix to CSV
    merged_matrix.to_csv(output_path)
    print(f"Success! Matrix saved to {output_path} with shape {merged_matrix.shape}")

    return merged_matrix

# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    build_count_matrix(INPUT_FILES, CONDITIONS, OUTPUT_FILE, DATA_DIR)
