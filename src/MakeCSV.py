import os
import pandas as pd
import random
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def find_audio_files(base_dir, extensions=('.wav', '.mp3', '.flac', '.ogg')):
    """
    Find all audio files in the directory structure and organize them by class.
    
    Args:
        base_dir: Root directory containing class folders
        extensions: Tuple of valid audio file extensions
        
    Returns:
        dict: Dictionary mapping class names to lists of audio file paths
    """
    audio_files_by_class = {}
    
    # Walk through the directory
    for root, dirs, files in os.walk(base_dir):
        # Get class name from directory name
        class_name = os.path.basename(root)
        
        # Skip the base directory itself
        if root == base_dir:
            continue
        
        # Find audio files in the current directory
        audio_files = [
            os.path.join(root, file) 
            for file in files 
            if file.lower().endswith(extensions)
        ]
        
        # If we found audio files, add them to our dictionary
        if audio_files:
            # Use relative paths from the base directory
            rel_paths = [os.path.relpath(path, base_dir) for path in audio_files]
            audio_files_by_class[class_name] = rel_paths
    
    return audio_files_by_class

def generate_contrastive_pairs(audio_files_by_class, num_pairs_per_class=100, same_ratio=0.5):
    """
    Generate pairs of audio files for contrastive learning.
    
    Args:
        audio_files_by_class: Dictionary mapping class names to lists of audio files
        num_pairs_per_class: Number of pairs to generate per class
        same_ratio: Ratio of same-class pairs (1) to different-class pairs (0)
        
    Returns:
        list: List of tuples (audio_path1, audio_path2, label)
    """
    pairs = []
    classes = list(audio_files_by_class.keys())
    
    # For each class
    for class_name in tqdm(classes, desc="Generating pairs"):
        files = audio_files_by_class[class_name]
        
        if len(files) < 2:
            print(f"Warning: Class {class_name} has less than 2 files. Skipping.")
            continue
        
        # Calculate how many same-class and different-class pairs to generate
        num_same = int(num_pairs_per_class * same_ratio)
        num_diff = num_pairs_per_class - num_same
        
        # Generate same-class pairs (label=1)
        for _ in range(num_same):
            # Randomly select two different files from the same class
            if len(files) >= 2:
                file1, file2 = random.sample(files, 2)
                pairs.append((file1, file2, 1))
        
        # Generate different-class pairs (label=0)
        for _ in range(num_diff):
            # Select a file from the current class
            file1 = random.choice(files)
            
            # Select a random different class
            other_classes = [c for c in classes if c != class_name]
            if not other_classes:
                continue
                
            other_class = random.choice(other_classes)
            
            # Select a random file from the other class
            if audio_files_by_class[other_class]:
                file2 = random.choice(audio_files_by_class[other_class])
                pairs.append((file1, file2, 0))
    
    # Shuffle the pairs
    random.shuffle(pairs)
    return pairs

def save_pairs_to_csv(pairs, output_file):
    """Save the generated pairs to a CSV file."""
    df = pd.DataFrame(pairs, columns=['audio_path1', 'audio_path2', 'label'])
    df.to_csv(output_file, index=False)
    print(f"Saved {len(pairs)} pairs to {output_file}")
    return df

def create_train_val_split(pairs, train_csv, val_csv, test_size=0.2, random_state=42):
    """Split the pairs into training and validation sets."""
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=test_size, random_state=random_state, stratify=[p[2] for p in pairs]
    )
    
    # Save to CSV files
    train_df = save_pairs_to_csv(train_pairs, train_csv)
    val_df = save_pairs_to_csv(val_pairs, val_csv)
    
    return train_df, val_df

def analyze_dataset(audio_files_by_class):
    """Print statistics about the dataset."""
    print("\nDataset Analysis:")
    print("-" * 40)
    
    total_files = 0
    for class_name, files in audio_files_by_class.items():
        num_files = len(files)
        total_files += num_files
        print(f"Class: {class_name} - {num_files} files")
    
    print("-" * 40)
    print(f"Total classes: {len(audio_files_by_class)}")
    print(f"Total audio files: {total_files}")
    print("-" * 40)

def main():
    # parser = argparse.ArgumentParser(description="Generate contrastive pairs CSV from audio directory")
    # parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio class folders")
    # parser.add_argument("--output_dir", type=str, default="data", help="Directory to save CSV files")
    # parser.add_argument("--pairs_per_class", type=int, default=100, help="Number of pairs to generate per class")
    # parser.add_argument("--same_ratio", type=float, default=0.5, help="Ratio of same-class pairs to different-class pairs")
    # parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    # parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    # args = parser.parse_args()
    
    # Set random seed
    # random.seed(args.random_seed)
    
    # Create output directory if it doesn't exist
    # os.makedirs(args.output_dir, exist_ok=True)
    
    os.makedirs("./data", exist_ok=True)
    # Find audio files by class
    # print(f"Scanning audio files in {args.audio_dir}...")
    
    print(f"Scanning audio files in {'./Audio_dataset'}...")
    # audio_files_by_class = find_audio_files(args.audio_dir)
    
    audio_files_by_class = find_audio_files("./Audio_dataset")
    
    # Check if we found any files
    if not audio_files_by_class:
        print("No audio files found. Please check the directory structure.")
        return
    
    # Analyze the dataset
    analyze_dataset(audio_files_by_class)
    
    # Generate pairs
    # print(f"\nGenerating {args.pairs_per_class} pairs per class with {args.same_ratio*100}% same-class ratio...")
    print(f"\nGenerating {100} pairs per class with {0.5*100}% same-class ratio...")
    pairs = generate_contrastive_pairs(
        audio_files_by_class, 
        num_pairs_per_class=100, 
        same_ratio=0.5
    )
    
    # Create train/val split and save to CSV
    # train_csv = os.path.join(args.output_dir, "contrastive_train.csv")
    # val_csv = os.path.join(args.output_dir, "contrastive_val.csv")
    
    train_csv = os.path.join("./data", "contrastive_train.csv")
    val_csv = os.path.join("./data", "contrastive_val.csv")
    
    # print(f"\nSplitting into training ({100-args.val_split*100}%) and validation ({args.val_split*100}%) sets...")
    # create_train_val_split(pairs, train_csv, val_csv, test_size=args.val_split, random_state=args.random_seed)
    
    print(f"\nSplitting into training ({100-0.2*100}%) and validation ({0.2*100}%) sets...")
    create_train_val_split(pairs, train_csv, val_csv, test_size=0.2, random_state=42)
    
    print("\nDone! CSV files are ready for contrastive learning.")
    print(f"Training CSV: {train_csv}")
    print(f"Validation CSV: {val_csv}")

if __name__ == "__main__":
    main()