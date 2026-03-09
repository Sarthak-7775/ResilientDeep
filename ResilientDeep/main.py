# main.py
import argparse
import os
import tqdm

def run_attack_pipeline(input_base_dir, output_base_dir):
    """Loops through the dataset, applying compression and upscaling."""
    from src.data_pipeline.upscale import create_attack_image
    
    # Define subfolders
    subfolders = ["Celeb-real", "Celeb-synthesis"]
    
    for folder in subfolders:
        input_dir = os.path.join(input_base_dir, folder)
        output_dir = os.path.join(output_base_dir, folder)
        
        # Create output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(input_dir):
            print(f"Skipping {input_dir} (Not found)")
            continue
            
        images = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        print(f"\nProcessing {len(images)} images in {folder}...")
        
        for img_name in tqdm.tqdm(images, desc=f"Attacking {folder}"):
            in_path = os.path.join(input_dir, img_name)
            out_path = os.path.join(output_dir, img_name)
            
            # This function (from upscale.py) compresses then upscales the image
            create_attack_image(in_path, out_path)
            
    print(f"\nAttack pipeline complete. Data saved to: {output_base_dir}")

def main():
    parser = argparse.ArgumentParser(description="ResilientDeep Execution Engine")
    parser.add_argument('--mode', type=str, choices=['train', 'dashboard', 'attack'], required=True,
                        help="Choose what part of the prototype to run.")
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("Initializing Training Pipeline...")
        from src.training.train import train
        train()
        
    elif args.mode == 'dashboard':
        print("Launching Hardware Interface...")
        os.system("streamlit run dashboard/app.py")
        
    elif args.mode == 'attack':
        print("Executing Phase 1: Attack Pipeline (Compression & Upscaling)...")
        # Define paths for your sample dataset
        input_data = os.path.abspath("data/sample_dataset")
        output_data = os.path.abspath("data/sample_attacked")
        run_attack_pipeline(input_data, output_data)

if __name__ == "__main__":
    main()