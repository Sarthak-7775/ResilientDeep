# main.py
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="ResilientDeep Execution Engine")
    parser.add_argument('--mode', type=str, choices=['train', 'dashboard', 'attack'], required=True,
                        help="Choose what part of the prototype to run.")
    
    args = parser.parse_args()

    if args.mode == 'train':
        print("Initializing Training Pipeline...")
        # Import dynamically to avoid loading torch if we just want the dashboard
        from src.training.train import train
        train()
        
    elif args.mode == 'dashboard':
        print("Launching Hardware Interface...")
        os.system("streamlit run dashboard/app.py")
        
    elif args.mode == 'attack':
        print("Executing Phase 1: Attack Pipeline (Compression & Upscaling)...")
        from src.data_pipeline.upscale import create_attack_image
        # Example execution (you would wrap this in a loop for your whole dataset)
        print("Note: Implement your dataset traversal loop here utilizing upscale.py")

if __name__ == "__main__":
    main()