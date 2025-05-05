import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from Preprocessing import load_and_preprocess_audio_file
from Trainer import ResNetMelLite
from Utils import EnhancedSimilarityMatcher, Matcher

class WakewordEvaluator:
    def __init__(self, model_path, classes_path, output_dir="./evaluation_results"):
        """Initialize the evaluator with model and classes paths"""
        self.model_path = model_path
        self.classes_path = classes_path
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load classes
        with open(classes_path, 'r') as f:
            s = f.read()
        self.classes = s.split("\n")
        if self.classes[-1] == '':  # Remove empty last element if present
            self.classes = self.classes[:]
        self.num_classes = len(self.classes)
        print(f"Loaded {self.num_classes} classes from {classes_path}")

        # Load model
        self.model = ResNetMelLite.load_from_checkpoint(model_path, num_classes=self.num_classes)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("Model loaded on GPU")
        else:
            print("Model loaded on CPU")

        # Initialize matcher
        self.matcher = Matcher()
        
    def get_embeddings(self, audio_path):
        """Extract embeddings from audio file"""
        try:
            mel_spec = load_and_preprocess_audio_file(audio_path, max_duration=1.0)
            mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32)
            mel_spec_tensor = mel_spec_tensor.unsqueeze(1)
            
            if torch.cuda.is_available():
                mel_spec_tensor = mel_spec_tensor.cuda()
                
            # Get classification output
            with torch.no_grad():
                classification_output = self.model(mel_spec_tensor)
                
            # Get embeddings by removing the final classification layer
            original_fc = self.model.model.fc[4]
            self.model.model.fc[4] = nn.Sequential()
            
            with torch.no_grad():
                embeddings = self.model(mel_spec_tensor)
                
            # Restore the original model
            self.model.model.fc[4] = original_fc
            
            return {
                "embeddings": embeddings,
                "classification": classification_output,
                "success": True
            }
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return {
                "embeddings": None,
                "classification": None,
                "success": False
            }
    
    def evaluate_single_file(self, audio_path):
        """Evaluate a single audio file and return results"""
        result = self.get_embeddings(audio_path)
        
        if not result["success"]:
            return {
                "file": audio_path,
                "predicted_class": "ERROR",
                "confidence": 0.0,
                "top_5_classes": [],
                "top_5_confidences": []
            }
            
        classification = result["classification"]
        
        # Get top prediction
        probabilities = F.softmax(classification, dim=1)
        confidence, index = torch.max(probabilities, dim=1)
        predicted_class = self.classes[index.item()]
        
        # Get top 5 predictions
        top_values, top_indices = torch.topk(probabilities, min(5, self.num_classes), dim=1)
        top_classes = [self.classes[idx.item()] for idx in top_indices[0]]
        top_confidences = [val.item() for val in top_values[0]]
        
        return {
            "file": os.path.basename(audio_path),
            "predicted_class": predicted_class,
            "confidence": confidence.item(),
            "top_5_classes": top_classes,
            "top_5_confidences": top_confidences
        }
    
    def compare_files(self, file1, file2):
        """Compare two audio files and return similarity"""
        result1 = self.get_embeddings(file1)
        result2 = self.get_embeddings(file2)
        
        if not (result1["success"] and result2["success"]):
            return {
                "file1": os.path.basename(file1),
                "file2": os.path.basename(file2),
                "cosine_similarity": 0.0,
                "matcher_similarity": 0.0
            }
            
        emb1 = result1["embeddings"]
        emb2 = result2["embeddings"]
        
        cosine_sim = torch.cosine_similarity(emb1, emb2, dim=1).item()
        matcher_sim = self.matcher.match(emb1, emb2)[1]
        
        return {
            "file1": os.path.basename(file1),
            "file2": os.path.basename(file2),
            "cosine_similarity": cosine_sim,
            "matcher_similarity": matcher_sim
        }
    
    def evaluate_files(self, file_list):
        """Evaluate a list of audio files and return results as a dataframe"""
        results = []
        
        print("Evaluating individual files...")
        for file_path in tqdm(file_list):
            if os.path.exists(file_path):
                result = self.evaluate_single_file(file_path)
                results.append(result)
            else:
                print(f"File not found: {file_path}")
                
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Add formatted columns for display
        df['confidence_pct'] = df['confidence'].apply(lambda x: f"{x:.2%}")
        df['top_5'] = df.apply(lambda row: "\n".join([f"{cls} ({conf:.2%})" for cls, conf in 
                                          zip(row['top_5_classes'], row['top_5_confidences'])]), axis=1)
        
        return df
    
    def compare_file_pairs(self, file_pairs):
        """Compare pairs of audio files and return results as a dataframe"""
        results = []
        
        print("Comparing file pairs...")
        for file1, file2 in tqdm(file_pairs):
            if os.path.exists(file1) and os.path.exists(file2):
                result = self.compare_files(file1, file2)
                results.append(result)
            else:
                print(f"One or both files not found: {file1}, {file2}")
                
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Add formatted columns for display
        df['cosine_similarity_pct'] = df['cosine_similarity'].apply(lambda x: f"{x:.2%}")
        df['matcher_similarity_pct'] = df['matcher_similarity'].apply(lambda x: f"{x:.2%}")
        
        return df
    
    def evaluate_and_display(self, file_list, file_pairs=None):
        """Evaluate files, compare pairs, and display results"""
        # Evaluate individual files
        eval_df = self.evaluate_files(file_list)
        
        # Display individual file results
        print("\n=== Individual File Evaluation ===")
        display_df = eval_df[['file', 'predicted_class', 'confidence_pct']]
        print(tabulate(display_df, headers='keys', tablefmt='grid'))
        
        # Display detailed results with top 5
        print("\n=== Detailed Classification Results ===")
        detailed_df = eval_df[['file', 'predicted_class', 'confidence_pct', 'top_5']]
        print(tabulate(detailed_df, headers='keys', tablefmt='grid', maxcolwidths=[20, 15, 10, 30]))
        
        # Compare pairs if provided
        if file_pairs:
            comparison_df = self.compare_file_pairs(file_pairs)
            
            print("\n=== File Pair Comparison ===")
            comp_display_df = comparison_df[['file1', 'file2', 'cosine_similarity_pct', 'matcher_similarity_pct']]
            print(tabulate(comp_display_df, headers='keys', tablefmt='grid'))
        
        # Save results to CSV
        csv_path = os.path.join(self.output_dir, 'evaluation_results.csv')
        eval_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        if file_pairs:
            pairs_csv_path = os.path.join(self.output_dir, 'pairs_comparison.csv')
            comparison_df.to_csv(pairs_csv_path, index=False)
            print(f"Pair comparison saved to {pairs_csv_path}")
        
        # Generate visualizations
        self._generate_visualizations(eval_df, comparison_df if file_pairs else None)
        
        return eval_df, comparison_df if file_pairs else None
    
    def _generate_visualizations(self, eval_df, comparison_df=None):
        """Generate visualizations for the evaluation results"""
        # Create confidence distribution plot
        plt.figure(figsize=(10, 6))
        sns.histplot(eval_df['confidence'], bins=20, kde=True)
        plt.title('Confidence Score Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plot_path = os.path.join(self.output_dir, 'confidence_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Create class distribution plot
        plt.figure(figsize=(12, 8))
        class_counts = eval_df['predicted_class'].value_counts()
        sns.barplot(x=class_counts.index, y=class_counts.values)
        plt.title('Predicted Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'class_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        
        # Create similarity comparison plot if comparison data exists
        if comparison_df is not None and not comparison_df.empty:
            plt.figure(figsize=(10, 6))
            plt.scatter(comparison_df['cosine_similarity'], comparison_df['matcher_similarity'])
            plt.title('Cosine Similarity vs Matcher Similarity')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Matcher Similarity')
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add file pair labels
            for i, row in comparison_df.iterrows():
                plt.annotate(f"{row['file1']}\nvs\n{row['file2']}", 
                            (row['cosine_similarity'], row['matcher_similarity']),
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center',
                            fontsize=8)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'similarity_comparison.png')
            plt.savefig(plot_path)
            plt.close()

if __name__ == "__main__":
    # Define model and classes file paths here
    # model_path = "./lightning_logs/version_28/checkpoints/epoch=8-step=28107.ckpt"
    # classes_path = "./mswc3_cache/classes.txt"
    
    # Alternative paths (uncomment to use)
    model_path = "./lightning_logs/version_27/checkpoints/epoch=14-step=46560.ckpt" # sam is detected threshold avg = 0.64987453 or 0.6498745325952768
    classes_path = "./mswc_cache/classes.txt"
    # model_path = "./lightning_logs/version_27/checkpoints/epoch=9-step=31040.ckpt" # sam is not detected threshold avg = 0.79239375 or 0.7917719706892967
    # classes_path = "./mswc_cache/classes.txt"
    
    output_dir = "./evaluation_results"
    
    # Define audio files to test
    test_files = [
        "./Audios4testing/sam_1.wav",
        "./Audios4testing/sam_2.wav", 
        "./Audios4testing/shambu_1.wav",
        "./Audios4testing/shambu_2.wav",
        "./Audios4testing/alexa_1.wav",
        "./Audios4testing/alexa_2.wav",
        "./Audios4testing/shiva_1.wav",
        "./Audios4testing/shiva_2.wav",
        "./Audios4testing/munez_2.wav",
        "./Audios4testing/munez_3.wav",
        "./Audios4testing/nigga2.wav",
        "./Audios4testing/nigga3.wav",
    ]
    
    # Define pairs to compare (pairs of files to check similarity)
    # Each sublist contains files that should be similar to each other
    wakeword_groups = [
        # Sam wakeword variations
        ["./Audios4testing/sam_1.wav", "./Audios4testing/sam_2.wav"],
        
        # Shambu wakeword variations
        ["./Audios4testing/shambu_1.wav", "./Audios4testing/shambu_2.wav"],
        
        # Alexa wakeword variations
        ["./Audios4testing/alexa_1.wav", "./Audios4testing/alexa_2.wav", "./Audios4testing/alexa_3.wav", "./Audios4testing/alexa_4.wav", "./Audios4testing/alexa_5.wav",],
        
        # Shiva wakeword variations
        ["./Audios4testing/shiva_1.wav", "./Audios4testing/shiva_2.wav"],
        
        # Munez wakeword variations
        ["./Audios4testing/munez_1.wav", "./Audios4testing/munez_2.wav"],
        
        # Nigga wakeword variations
        ["./Audios4testing/nigga2.wav", "./Audios4testing/nigga3.wav"],
        
        # Kallan wakeword variations
        ["./Audios4testing/kallan_1.wav", "./Audios4testing/kallan_2.wav"],
        
        # Hello wakeword variations
        ["./Audios4testing/Hello1.mp3", "./Audios4testing/Hello2.mp3"],
        
        # Eliza wakeword variations
        ["./Audios4testing/Eliza1.mp3", "./Audios4testing/Eliza2.mp3"],
        
        # Skywalker wakeword variations
        ["./Audios4testing/Skywalker_en-AU-jimm.mp3", "./Audios4testing/Skywalker_en-AU-kylie.mp3"],
        
        # Thunderbolt wakeword variations
        ["./Audios4testing/Thunderbolt_en-AU-jimm.mp3", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"],
        
        # Thunderbolt wakeword variations
        ["./Audios4testing/Quasar_en-AU-jimm.mp3", "./Audios4testing/Qausar_en-AU-kylie.mp3"],
    ]
    
    # Generate all possible pairs within each group
    file_pairs = []
    for group in wakeword_groups:
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                file_pairs.append((group[i], group[j]))
    
    # Also add some cross-wakeword comparisons to verify they're different
    # cross_wakeword_pairs = [
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/shambu_1.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/alexa_1.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/munez_2.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/munez_3.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/alexa_2.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/shiva_1.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/shiva_2.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/nigga1.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/nigga2.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/kallan_1.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/kallan_2.wav"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/Qausar_en-AU-kylie.mp3"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/Qausar_en-AU-jimm.mp3"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/Thunderbolt_en-AU-jimm.mp3"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/Skywalker_en-AU-kylie.mp3"),
    #     ("./Audios4testing/sam_1.wav", "./Audios4testing/Skywalker_en-AU-jimm.mp3"),
        
        
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/shambu_1.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/alexa_1.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/alexa_2.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/munez_2.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/munez_3.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/shiva_1.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/shiva_2.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/nigga1.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/nigga2.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/kallan_1.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/kallan_2.wav"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/Qausar_en-AU-kylie.mp3"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/Qausar_en-AU-jimm.mp3"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/Thunderbolt_en-AU-jimm.mp3"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/Skywalker_en-AU-kylie.mp3"),
    #     ("./Audios4testing/sam_2.wav", "./Audios4testing/Skywalker_en-AU-jimm.mp3"),
        
        
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/shambu_1.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/alexa_1.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/alexa_2.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/munez_2.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/munez_3.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/nigga1.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/nigga2.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/kallan_1.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/kallan_2.wav"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/Qausar_en-AU-kylie.mp3"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/Qausar_en-AU-jimm.mp3"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/Thunderbolt_en-AU-jimm.mp3"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/Skywalker_en-AU-kylie.mp3"),
    #     ("./Audios4testing/shiva_1.wav", "./Audios4testing/Skywalker_en-AU-jimm.mp3"),
        
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/shambu_1.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/alexa_1.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/alexa_2.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/shiva_1.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/shiva_2.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/nigga1.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/nigga2.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/kallan_1.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/kallan_2.wav"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/Qausar_en-AU-kylie.mp3"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/Qausar_en-AU-jimm.mp3"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/Thunderbolt_en-AU-jimm.mp3"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/Skywalker_en-AU-kylie.mp3"),
    #     ("./Audios4testing/munez_2.wav", "./Audios4testing/Skywalker_en-AU-jimm.mp3"),
    
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/shambu_1.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/munez_2.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/munez_3.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/shiva_1.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/shiva_2.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/nigga1.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/nigga2.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/kallan_1.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/kallan_2.wav"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/Qausar_en-AU-kylie.mp3"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/Qausar_en-AU-jimm.mp3"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/Thunderbolt_en-AU-jimm.mp3"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/Skywalker_en-AU-kylie.mp3"),
    #     ("./Audios4testing/alexa_1.wav", "./Audios4testing/Skywalker_en-AU-jimm.mp3"),
    
    # ]
    
    # Optimized cross-wakeword comparison pairs
    cross_wakeword_pairs = [
        # Sam vs other wakewords (covering different types)
        ("./Audios4testing/sam_1.wav", "./Audios4testing/shambu_1.wav"),
        ("./Audios4testing/sam_2.wav", "./Audios4testing/alexa_3.wav"),
        ("./Audios4testing/sam_1.wav", "./Audios4testing/shiva_2.wav"),
        ("./Audios4testing/sam_2.wav", "./Audios4testing/munez_1.wav"),
        ("./Audios4testing/sam_1.wav", "./Audios4testing/nigga2.wav"),
        ("./Audios4testing/sam_2.wav", "./Audios4testing/kallan_1.wav"),
        ("./Audios4testing/sam_1.wav", "./Audios4testing/Hello2.mp3"),
        ("./Audios4testing/sam_2.wav", "./Audios4testing/Eliza1.mp3"),
        ("./Audios4testing/sam_1.wav", "./Audios4testing/Skywalker_en-AU-jimm.mp3"),
        ("./Audios4testing/sam_2.wav", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"),
        
        # Shambu vs other wakewords
        ("./Audios4testing/shambu_1.wav", "./Audios4testing/alexa_1.wav"),
        ("./Audios4testing/shambu_2.wav", "./Audios4testing/nigga1.wav"),
        ("./Audios4testing/shambu_1.wav", "./Audios4testing/Qausar_en-AU-jimm.mp3"),
        
        # Shiva vs other wakewords
        ("./Audios4testing/shiva_1.wav", "./Audios4testing/munez_2.wav"),
        ("./Audios4testing/shiva_2.wav", "./Audios4testing/kallan_2.wav"),
        ("./Audios4testing/shiva_1.wav", "./Audios4testing/Eliza2.mp3"),
        
        # Alexa vs other wakewords
        ("./Audios4testing/alexa_2.wav", "./Audios4testing/nigga3.wav"),
        ("./Audios4testing/alexa_4.wav", "./Audios4testing/Hello1.mp3"),
        ("./Audios4testing/alexa_5.wav", "./Audios4testing/Thunderbolt_en-AU-jimm.mp3"),
        
        # Munez vs other wakewords
        ("./Audios4testing/munez_1.wav", "./Audios4testing/kallan_1.wav"),
        ("./Audios4testing/munez_2.wav", "./Audios4testing/Skywalker_en-AU-kylie.mp3"),
        
        # Nigga vs other wakewords
        ("./Audios4testing/nigga1.wav", "./Audios4testing/Hello2.mp3"),
        ("./Audios4testing/nigga3.wav", "./Audios4testing/Qausar_en-AU-kylie.mp3"),
        
        # Kallan vs other wakewords
        ("./Audios4testing/kallan_2.wav", "./Audios4testing/Eliza1.mp3"),
        
        # MP3 wakewords comparisons
        ("./Audios4testing/Hello1.mp3", "./Audios4testing/Skywalker_en-AU-jimm.mp3"),
        ("./Audios4testing/Eliza2.mp3", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"),
        ("./Audios4testing/Skywalker_en-AU-kylie.mp3", "./Audios4testing/Qausar_en-AU-jimm.mp3"),
        ("./Audios4testing/Thunderbolt_en-AU-jimm.mp3", "./Audios4testing/Qausar_en-AU-kylie.mp3")
    ]
    file_pairs.extend(cross_wakeword_pairs)
    
    # Create evaluator
    evaluator = WakewordEvaluator(
        model_path=model_path,
        classes_path=classes_path,
        output_dir=output_dir
    )
    
    # Run evaluation
    evaluator.evaluate_and_display(test_files, file_pairs)