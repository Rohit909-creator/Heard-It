import os
import pandas as pd
import random
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def find_audio_files(base_dir, extensions=('.wav', '.mp3', '.flac', '.ogg', '.opus')):
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

def generate_triplets(audio_files_by_class, num_triplets_per_class=100):
    """
    Generate triplets of audio files for triplet learning.
    
    Args:
        audio_files_by_class: Dictionary mapping class names to lists of audio files
        num_triplets_per_class: Number of triplets to generate per class
        
    Returns:
        list: List of tuples (anchor_path, positive_path, negative_path)
    """
    triplets = []
    classes = list(audio_files_by_class.keys())
    
    # For each class
    for class_name in tqdm(classes, desc="Generating triplets"):
        files = audio_files_by_class[class_name]
        
        # Need at least 2 files in the same class for anchor and positive
        if len(files) < 2:
            print(f"Warning: Class {class_name} has less than 2 files. Skipping.")
            continue
        
        # Generate triplets for this class
        for _ in range(num_triplets_per_class):
            # Select anchor and positive from the same class (must be different files)
            if len(files) >= 2:
                anchor, positive = random.sample(files, 2)
                
                # Select a random different class for negative
                other_classes = [c for c in classes if c != class_name]
                if not other_classes:
                    continue
                    
                negative_class = random.choice(other_classes)
                
                # Select a random file from the negative class
                if audio_files_by_class[negative_class]:
                    negative = random.choice(audio_files_by_class[negative_class])
                    triplets.append((anchor, positive, negative))
    
    # Shuffle the triplets
    random.shuffle(triplets)
    return triplets

def save_triplets_to_csv(triplets, output_file):
    """Save the generated triplets to a CSV file."""
    df = pd.DataFrame(triplets, columns=['anchor_path', 'positive_path', 'negative_path'])
    df.to_csv(output_file, index=False)
    print(f"Saved {len(triplets)} triplets to {output_file}")
    return df

def create_train_val_split(triplets, train_csv, val_csv, test_size=0.2, random_state=42):
    """Split the triplets into training and validation sets."""
    train_triplets, val_triplets = train_test_split(
        triplets, test_size=test_size, random_state=random_state
    )
    
    # Save to CSV files
    train_df = save_triplets_to_csv(train_triplets, train_csv)
    val_df = save_triplets_to_csv(val_triplets, val_csv)
    
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
    # parser = argparse.ArgumentParser(description="Generate triplet CSV from audio directory")
    # parser.add_argument("--audio_dir", type=str, required=True, help="Directory containing audio class folders")
    # parser.add_argument("--output_dir", type=str, default="data", help="Directory to save CSV files")
    # parser.add_argument("--triplets_per_class", type=int, default=100, help="Number of triplets to generate per class")
    # parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    # parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    
    # args = parser.parse_args()
    
    # Set random seed
    random.seed(42)
    
    # Create output directory if it doesn't exist
    # os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    
    # Find audio files by class
    print(f"Scanning audio files in {'./Audio_dataset2'}...")
    audio_files_by_class = find_audio_files('./Audio_dataset2')
    
    # Check if we found any files
    if not audio_files_by_class:
        print("No audio files found. Please check the directory structure.")
        return
    
    # Analyze the dataset
    analyze_dataset(audio_files_by_class)
    
    # Generate triplets
    print(f"\nGenerating {100} triplets per class...")
    triplets = generate_triplets(
        audio_files_by_class, 
        num_triplets_per_class=100
    )
    
    # Create train/val split and save to CSV
    train_csv = os.path.join('./data', "triplet_train.csv")
    val_csv = os.path.join('./data', "triplet_val.csv")
    
    print(f"\nSplitting into training ({100-0.2*100}%) and validation ({0.2*100}%) sets...")
    create_train_val_split(triplets, train_csv, val_csv, test_size=0.2, random_state=42)
    
    print("\nDone! CSV files are ready for triplet learning.")
    print(f"Training CSV: {train_csv}")
    print(f"Validation CSV: {val_csv}")

if __name__ == "__main__":
    main()