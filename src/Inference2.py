import torch
import torch.nn.functional as F
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# Assuming these imports work from the original code
from Preprocessing import load_and_preprocess_audio_file
from Trainer import ResNetMelLite
from Utils import Matcher

class MultiModelEvaluator:
    """Evaluates multiple models on the same audio dataset for comparison"""
    
    def __init__(self, model_configs, output_dir="./model_comparison_results"):
        """
        Initialize evaluator with multiple model configurations
        
        Args:
            model_configs: List of dicts with 'model_path' and 'classes_path' keys
            output_dir: Directory to save results
        """
        self.model_configs = model_configs
        self.output_dir = output_dir
        self.models = {}
        self.classes_maps = {}
        self.matcher = Matcher()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load all models
        print(f"Loading {len(model_configs)} models...")
        for idx, config in enumerate(model_configs):
            model_id = f"model_{idx}"
            model_path = config['model_path']
            classes_path = config['classes_path']
            model_name = config.get('name', f"Model {idx+1}")
            
            # Load classes
            with open(classes_path, 'r') as f:
                classes = f.read().split('\n')
                # if classes[-1] == '':  # Remove empty last element if present
                #     classes = classes[:-1]
            
            # Load model
            model = ResNetMelLite.load_from_checkpoint(model_path, num_classes=len(classes))
            model.eval()
            if torch.cuda.is_available():
                model = model.cuda()
            
            self.models[model_id] = {
                'model': model,
                'classes': classes,
                'path': model_path,
                'name': model_name
            }
            self.classes_maps[model_id] = classes
            
            print(f"Loaded {model_name}: {len(classes)} classes from {classes_path}")
    
    def _process_audio(self, audio_path, model_id):
        """Process a single audio file with a specific model"""
        model_data = self.models[model_id]
        model = model_data['model']
        classes = model_data['classes']
        
        try:
            # Preprocess audio
            mel_spec = load_and_preprocess_audio_file(audio_path, max_duration=1.0)
            mel_spec_tensor = torch.tensor([mel_spec], dtype=torch.float32).unsqueeze(1)
            
            if torch.cuda.is_available():
                mel_spec_tensor = mel_spec_tensor.cuda()
            
            # Get classification output
            with torch.no_grad():
                classification_output = model(mel_spec_tensor)
            
            # Get embeddings
            original_fc = model.model.fc[4]
            model.model.fc[4] = torch.nn.Sequential()
            
            with torch.no_grad():
                embeddings = model(mel_spec_tensor)
            
            # Restore original model
            model.model.fc[4] = original_fc
            
            # Get predictions
            probabilities = F.softmax(classification_output, dim=1)
            confidence, index = torch.max(probabilities, dim=1)
            predicted_class = classes[index.item()]
            
            # Get top 5 predictions
            top_k = min(5, len(classes))
            top_values, top_indices = torch.topk(probabilities, top_k, dim=1)
            top_classes = [classes[idx.item()] for idx in top_indices[0]]
            top_confidences = [val.item() for val in top_values[0]]
            
            return {
                'model_id': model_id,
                'file': os.path.basename(audio_path),
                'predicted_class': predicted_class,
                'confidence': confidence.item(),
                'top_classes': top_classes,
                'top_confidences': top_confidences,
                'embeddings': embeddings,
                'success': True
            }
            
        except Exception as e:
            print(f"Error processing {audio_path} with {model_id}: {str(e)}")
            return {
                'model_id': model_id,
                'file': os.path.basename(audio_path),
                'predicted_class': "ERROR",
                'confidence': 0.0,
                'top_classes': [],
                'top_confidences': [],
                'embeddings': None,
                'success': False
            }
    
    def compare_files(self, file1, file2, model_id):
        """Compare two audio files using a specific model"""
        result1 = self._process_audio(file1, model_id)
        result2 = self._process_audio(file2, model_id)
        
        if not (result1['success'] and result2['success']):
            return {
                'model_id': model_id,
                'file1': os.path.basename(file1),
                'file2': os.path.basename(file2),
                'cosine_similarity': 0.0,
                'matcher_similarity': 0.0
            }
        
        emb1 = result1['embeddings']
        emb2 = result2['embeddings']
        
        cosine_sim = torch.cosine_similarity(emb1, emb2, dim=1).item()
        matcher_sim = self.matcher.match(emb1, emb2)[1]
        
        return {
            'model_id': model_id,
            'file1': os.path.basename(file1),
            'file2': os.path.basename(file2),
            'cosine_similarity': cosine_sim,
            'matcher_similarity': matcher_sim
        }
    
    def evaluate_files(self, file_list, file_pairs=None):
        """Evaluate all audio files with all models"""
        results = []
        comparison_results = []
        
        start_time = time.time()
        
        # Evaluate individual files with each model
        for model_id in self.models:
            model_name = self.models[model_id]['name']
            print(f"\nEvaluating files with {model_name}...")
            
            for file_path in tqdm(file_list):
                if os.path.exists(file_path):
                    result = self._process_audio(file_path, model_id)
                    results.append(result)
                else:
                    print(f"File not found: {file_path}")
        
        # Compare file pairs with each model if provided
        if file_pairs:
            for model_id in self.models:
                model_name = self.models[model_id]['name']
                print(f"\nComparing file pairs with {model_name}...")
                
                for file1, file2 in tqdm(file_pairs):
                    if os.path.exists(file1) and os.path.exists(file2):
                        result = self.compare_files(file1, file2, model_id)
                        comparison_results.append(result)
                    else:
                        print(f"One or both files not found: {file1}, {file2}")
        
        # Create DataFrames
        results_df = pd.DataFrame(results)
        
        # Add model name column
        results_df['model_name'] = results_df['model_id'].apply(lambda mid: self.models[mid]['name'])
        
        # Add formatted columns for display
        results_df['confidence_pct'] = results_df['confidence'].apply(lambda x: f"{x:.2%}")
        
        if file_pairs:
            comparison_df = pd.DataFrame(comparison_results)
            comparison_df['model_name'] = comparison_df['model_id'].apply(lambda mid: self.models[mid]['name'])
            comparison_df['cosine_similarity_pct'] = comparison_df['cosine_similarity'].apply(lambda x: f"{x:.2%}")
            comparison_df['matcher_similarity_pct'] = comparison_df['matcher_similarity'].apply(lambda x: f"{x:.2%}")
        else:
            comparison_df = None
        
        print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds")
        
        return results_df, comparison_df
    
    def analyze_and_display(self, results_df, comparison_df=None):
        """Analyze and display results, generate visualizations"""
        # Save raw results to CSV
        results_csv_path = os.path.join(self.output_dir, 'all_model_evaluations.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"\nRaw results saved to {results_csv_path}")
        
        if comparison_df is not None:
            comparison_csv_path = os.path.join(self.output_dir, 'all_model_comparisons.csv')
            comparison_df.to_csv(comparison_csv_path, index=False)
            print(f"Comparison results saved to {comparison_csv_path}")
        
        # Model performance summary
        model_summary = results_df.groupby('model_name')['confidence'].agg(
            mean_confidence=np.mean,
            median_confidence=np.median,
            min_confidence=np.min,
            max_confidence=np.max
        ).reset_index()
        
        model_summary['mean_confidence'] = model_summary['mean_confidence'].apply(lambda x: f"{x:.2%}")
        model_summary['median_confidence'] = model_summary['median_confidence'].apply(lambda x: f"{x:.2%}")
        model_summary['min_confidence'] = model_summary['min_confidence'].apply(lambda x: f"{x:.2%}")
        model_summary['max_confidence'] = model_summary['max_confidence'].apply(lambda x: f"{x:.2%}")
        
        print("\n=== Model Performance Summary ===")
        print(tabulate(model_summary, headers='keys', tablefmt='grid'))
        
        # Audio file classification summary across models
        file_summary = results_df.pivot_table(
            index='file', 
            columns='model_name', 
            values=['predicted_class', 'confidence'],
            aggfunc={'predicted_class': lambda x: x.iloc[0], 'confidence': np.mean}
        )
        
        # Format confidence values
        for model_name in file_summary['confidence'].columns:
            file_summary[('confidence', model_name)] = file_summary[('confidence', model_name)].apply(lambda x: f"{x:.2%}")
        
        # Save file summary
        file_summary_path = os.path.join(self.output_dir, 'file_classification_summary.csv')
        file_summary.to_csv(file_summary_path)
        
        print("\n=== File Classification Summary (excerpt) ===")
        # Display only the first few rows for readability
        print(tabulate(file_summary.head(10), headers='keys', tablefmt='grid'))
        print(f"Full summary saved to {file_summary_path}")
        
        # If we have comparison data, analyze similarity metrics across models
        if comparison_df is not None:
            # Get average similarity metrics per model
            similarity_summary = comparison_df.groupby('model_name').agg(
                avg_cosine=('cosine_similarity', np.mean),
                avg_matcher=('matcher_similarity', np.mean),
                min_cosine=('cosine_similarity', np.min),
                max_cosine=('cosine_similarity', np.max),
                similarity_pairs=('file1', 'count')
            ).reset_index()
            
            # Format percentages
            similarity_summary['avg_cosine'] = similarity_summary['avg_cosine'].apply(lambda x: f"{x:.2%}")
            similarity_summary['avg_matcher'] = similarity_summary['avg_matcher'].apply(lambda x: f"{x:.2%}")
            similarity_summary['min_cosine'] = similarity_summary['min_cosine'].apply(lambda x: f"{x:.2%}")
            similarity_summary['max_cosine'] = similarity_summary['max_cosine'].apply(lambda x: f"{x:.2%}")
            
            print("\n=== Similarity Metrics Across Models ===")
            print(tabulate(similarity_summary, headers='keys', tablefmt='grid'))
            
            # Save similarity summary
            similarity_summary_path = os.path.join(self.output_dir, 'similarity_summary.csv')
            similarity_summary.to_csv(similarity_summary_path, index=False)
            print(f"Similarity summary saved to {similarity_summary_path}")
            
            # Generate comparison visualizations
            self._generate_similarity_comparisons(comparison_df)
        
        # Generate performance visualizations
        self._generate_model_performance_visualizations(results_df)
        
        return model_summary, file_summary, similarity_summary if comparison_df is not None else None
    
    def _generate_model_performance_visualizations(self, results_df):
        """Generate visualizations for model performance"""
        # Model confidence comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='model_name', y='confidence', data=results_df)
        plt.title('Confidence Distribution by Model')
        plt.xlabel('Model')
        plt.ylabel('Confidence Score')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_confidence_comparison.png'))
        plt.close()
        
        # Model-specific confidence distributions
        plt.figure(figsize=(15, 10))
        for i, model_id in enumerate(self.models):
            model_name = self.models[model_id]['name']
            model_data = results_df[results_df['model_id'] == model_id]
            
            plt.subplot(2, len(self.models)//2 + len(self.models)%2, i+1)
            sns.histplot(model_data['confidence'], bins=15, kde=True)
            plt.title(f'{model_name}')
            plt.xlabel('Confidence')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'model_confidence_distributions.png'))
        plt.close()
    
    def _generate_similarity_comparisons(self, comparison_df):
        """Generate visualizations for file pair similarities across models"""
        # Filter to include only certain types of pairs for clearer visualization
        same_word_pairs = comparison_df[
            comparison_df.apply(lambda x: x['file1'].split('_')[0] == x['file2'].split('_')[0], axis=1)
        ]
        
        # Generate scatter plot for each model comparing cosine vs matcher similarity
        plt.figure(figsize=(15, 10))
        for i, model_id in enumerate(self.models):
            model_name = self.models[model_id]['name']
            model_data = comparison_df[comparison_df['model_id'] == model_id]
            
            plt.subplot(2, len(self.models)//2 + len(self.models)%2, i+1)
            plt.scatter(model_data['cosine_similarity'], model_data['matcher_similarity'], alpha=0.7)
            plt.title(f'{model_name}')
            plt.xlabel('Cosine Similarity')
            plt.ylabel('Matcher Similarity')
            plt.grid(True, linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'similarity_comparison_by_model.png'))
        plt.close()
        
        # Generate bar chart comparing average similarities for same-word pairs
        avg_sim_by_model = same_word_pairs.groupby('model_name').agg(
            avg_cosine=('cosine_similarity', np.mean),
            avg_matcher=('matcher_similarity', np.mean)
        ).reset_index()
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(avg_sim_by_model))
        width = 0.35
        
        plt.bar(x - width/2, avg_sim_by_model['avg_cosine'], width, label='Avg Cosine Similarity')
        plt.bar(x + width/2, avg_sim_by_model['avg_matcher'], width, label='Avg Matcher Similarity')
        
        plt.xlabel('Model')
        plt.ylabel('Average Similarity')
        plt.title('Average Similarity Metrics for Same-Word Pairs by Model')
        plt.xticks(x, avg_sim_by_model['model_name'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'avg_similarity_by_model.png'))
        plt.close()


def run_multi_model_evaluation():
    """Main function to run evaluation across multiple models"""
    # Define model configurations
    model_configs = [
        {
            'name': 'v31_ep11',
            'model_path': "./lightning_logs/version_31/checkpoints/epoch=11-step=136404.ckpt",
            'classes_path': "./mswc3_cache/classes.txt"
        },
        {
            'name': 'v31_ep9',
            'model_path': "./lightning_logs/version_31/checkpoints/epoch=9-step=113670.ckpt",
            'classes_path': "./mswc3_cache/classes.txt"
        },
        {
            'name': 'v33_ep27',
            'model_path': "./lightning_logs/version_33/checkpoints/epoch=27-step=318276.ckpt",
            'classes_path': "./mswc3_cache/classes.txt"
        },
        {
            'name': 'v33_ep24',
            'model_path': "./lightning_logs/version_33/checkpoints/epoch=24-step=284175.ckpt",
            'classes_path': "./mswc3_cache/classes.txt"
        },
        {
            'name': 'v33_ep19',
            'model_path': "./lightning_logs/version_33/checkpoints/epoch=19-step=227340.ckpt",
            'classes_path': "./mswc3_cache/classes.txt"
        },
        {
            'name': 'v33_ep18',
            'model_path': "./lightning_logs/version_33/checkpoints/epoch=18-step=215973.ckpt",
            'classes_path': "./mswc3_cache/classes.txt"
        },
        {
            'name': 'v33_ep16',
            'model_path': "./lightning_logs/version_33/checkpoints/epoch=16-step=193239.ckpt",
            'classes_path': "./mswc3_cache/classes.txt"
        },
        {
            'name': 'v27_ep14_sam_detected',
            'model_path': "./lightning_logs/version_27/checkpoints/epoch=14-step=46560.ckpt",
            'classes_path': "./mswc_cache/classes.txt"
        },
        {
            'name': 'v27_ep9_sam_not_detected',
            'model_path': "./lightning_logs/version_27/checkpoints/epoch=9-step=31040.ckpt",
            'classes_path': "./mswc_cache/classes.txt"
        }
    ]
    
    # Define audio files to test
    test_files = [
        "./Audios4testing/sam_1.wav",
        "./Audios4testing/sam_2.wav", 
        "./Audios4testing/shambu_1.wav",
        "./Audios4testing/shambu_2.wav",
        "./Audios4testing/alexa_1.wav",
        "./Audios4testing/alexa_2.wav",
        "./Audios4testing/alexa_3.wav", 
        "./Audios4testing/alexa_4.wav", 
        "./Audios4testing/alexa_5.wav",
        "./Audios4testing/shiva_1.wav",
        "./Audios4testing/shiva_2.wav",
        "./Audios4testing/munez_1.wav",
        "./Audios4testing/munez_2.wav",
        "./Audios4testing/munez_3.wav",
        "./Audios4testing/nigga1.wav",
        "./Audios4testing/nigga2.wav",
        "./Audios4testing/nigga3.wav",
        "./Audios4testing/kallan_1.wav",
        "./Audios4testing/kallan_2.wav",
        "./Audios4testing/Hello1.mp3",
        "./Audios4testing/Hello2.mp3",
        "./Audios4testing/Eliza1.mp3",
        "./Audios4testing/Eliza2.mp3",
        "./Audios4testing/Skywalker_en-AU-jimm.mp3",
        "./Audios4testing/Skywalker_en-AU-kylie.mp3",
        "./Audios4testing/Thunderbolt_en-AU-jimm.mp3",
        "./Audios4testing/Thunderbolt_en-AU-kylie.mp3",
        "./Audios4testing/Qausar_en-AU-jimm.mp3",
        "./Audios4testing/Qausar_en-AU-kylie.mp3"
    ]
    
    # Define wakeword groups for similarity testing
    wakeword_groups = [
        ["./Audios4testing/sam_1.wav", "./Audios4testing/sam_2.wav"],
        ["./Audios4testing/shambu_1.wav", "./Audios4testing/shambu_2.wav"],
        ["./Audios4testing/alexa_1.wav", "./Audios4testing/alexa_2.wav", 
         "./Audios4testing/alexa_3.wav", "./Audios4testing/alexa_4.wav", 
         "./Audios4testing/alexa_5.wav"],
        ["./Audios4testing/shiva_1.wav", "./Audios4testing/shiva_2.wav"],
        ["./Audios4testing/munez_1.wav", "./Audios4testing/munez_2.wav", 
         "./Audios4testing/munez_3.wav"],
        ["./Audios4testing/nigga1.wav", "./Audios4testing/nigga2.wav", 
         "./Audios4testing/nigga3.wav"],
        ["./Audios4testing/kallan_1.wav", "./Audios4testing/kallan_2.wav"],
        ["./Audios4testing/Hello1.mp3", "./Audios4testing/Hello2.mp3"],
        ["./Audios4testing/Eliza1.mp3", "./Audios4testing/Eliza2.mp3"],
        ["./Audios4testing/Skywalker_en-AU-jimm.mp3", "./Audios4testing/Skywalker_en-AU-kylie.mp3"],
        ["./Audios4testing/Thunderbolt_en-AU-jimm.mp3", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3"],
        ["./Audios4testing/Qausar_en-AU-jimm.mp3", "./Audios4testing/Qausar_en-AU-kylie.mp3"]
    ]
    
    # Generate pairs within each group
    file_pairs = []
    for group in wakeword_groups:
        for i in range(len(group)):
            for j in range(i+1, len(group)):
                file_pairs.append((group[i], group[j]))
    
    # Add cross-wakeword comparisons
    cross_pairs = [
        ("./Audios4testing/sam_1.wav", "./Audios4testing/shambu_1.wav"),
        ("./Audios4testing/sam_2.wav", "./Audios4testing/alexa_3.wav"),
        ("./Audios4testing/sam_1.wav", "./Audios4testing/shiva_2.wav"),
        ("./Audios4testing/sam_2.wav", "./Audios4testing/munez_1.wav"),
        ("./Audios4testing/shambu_1.wav", "./Audios4testing/alexa_1.wav"),
        ("./Audios4testing/shiva_1.wav", "./Audios4testing/munez_2.wav"),
        ("./Audios4testing/alexa_2.wav", "./Audios4testing/nigga3.wav"),
        ("./Audios4testing/munez_1.wav", "./Audios4testing/kallan_1.wav"),
        ("./Audios4testing/nigga1.wav", "./Audios4testing/Hello2.mp3"),
        ("./Audios4testing/kallan_2.wav", "./Audios4testing/Eliza1.mp3"),
        ("./Audios4testing/Hello1.mp3", "./Audios4testing/Skywalker_en-AU-jimm.mp3"),
        ("./Audios4testing/Eliza2.mp3", "./Audios4testing/Thunderbolt_en-AU-kylie.mp3")
    ]
    file_pairs.extend(cross_pairs)
    
    # Initialize evaluator
    output_dir = "./model_comparison_results"
    evaluator = MultiModelEvaluator(model_configs, output_dir=output_dir)
    
    # Run evaluation
    print(f"Starting evaluation of {len(test_files)} files across {len(model_configs)} models...")
    results_df, comparison_df = evaluator.evaluate_files(test_files, file_pairs)
    
    # Analyze and display results
    print("\nGenerating analysis and visualizations...")
    model_summary, file_summary, similarity_summary = evaluator.analyze_and_display(results_df, comparison_df)
    
    print(f"\nEvaluation complete! All results saved to {output_dir}")
    return model_summary, file_summary, similarity_summary


if __name__ == "__main__":
    run_multi_model_evaluation()